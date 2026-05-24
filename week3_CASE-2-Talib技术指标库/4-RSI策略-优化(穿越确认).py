# -*- coding: utf-8 -*-
"""
RSI 策略 -- 优化版：用 TA-Lib 计算 RSI + 穿越确认入场

核心改进：
  标准版（Backtrader 课程）：bt.indicators.RSI，RSI<30 买，RSI>70 卖
  优化版（本课）：talib.RSI 计算，等 RSI 从低位回升穿过 30 线再买

为什么需要"穿越确认"？
  标准版 RSI<30 立即买入的问题：
  在单边下跌行情中，RSI 可能从 30 继续跌到 15 甚至更低。
  如果 RSI<30 就买，相当于"接飞刀"——在下跌半山腰入场。

  优化版的做法：
  等 RSI 从低于 30 回升到 30 以上才买入，这确认了：
    1. 下跌动能已经衰竭（RSI 不再创新低）
    2. 反弹已经开始（RSI 已经回升）
  虽然会错过最低点，但大幅降低了抄底抄在半山腰的风险。

TA-Lib 在策略中的用法：
  rsi = talib.RSI(close_array, timeperiod=14)  # 返回 numpy 数组
  rsi[-1] 是当前 K 线的 RSI 值
  rsi[-2] 是前一根 K 线的 RSI 值

  Backtrader 中获取完整价格数组：
  close_array = np.array(self.data.close.get(size=len(self.data)), dtype=np.float64)

运行: python 4-RSI策略-优化(穿越确认).py
"""
import numpy as np
import talib
import backtrader as bt
from data_loader import run_and_report


class RSIStandard(bt.Strategy):
    """
    标准版 RSI 策略（与 Backtrader 课程一致）

    逻辑：
      - 无持仓时：RSI < 30（超卖）-> 买入
      - 有持仓时：RSI > 70（超买）-> 卖出

    问题：
      在下跌趋势中，RSI 可能在 30 以下停留很久，持续买入会导致严重亏损。
    """
    params = (('period', 14), ('oversold', 30), ('overbought', 70))

    def __init__(self):
        # Backtrader 内置 RSI 指标，自动管理数据长度
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)

    def next(self):
        # self.rsi[0] 是当前 K 线的 RSI 值
        if not self.position:
            if self.rsi[0] < self.p.oversold:
                self.buy()          # RSI 进入超卖区，买入
        elif self.rsi[0] > self.p.overbought:
            self.close()            # RSI 进入超买区，卖出


class RSIOptimized(bt.Strategy):
    """
    优化版 RSI 策略：talib.RSI + 穿越确认

    改用 talib 计算 RSI：
      标准版用 bt.indicators.RSI（内部自管理）
      优化版用 talib.RSI（需要手动传数组）

    关键改进：穿越确认入场
      rsi[-2] < 30 且 rsi[-1] >= 30
      即 RSI 从超卖区回升，说明下跌趋势可能已经反转
      比标准版的 rsi < 30 立即买入更安全

    出场不变：RSI > 70 卖出
    """
    params = (('period', 14), ('oversold', 30), ('overbought', 70))

    def _calc_rsi(self):
        """
        使用 TA-Lib 计算 RSI

        TA-Lib 需要完整的 numpy float64 数组作为输入。
        self.data.close.get(size=size) 从 Backtrader 获取指定长度的价格数据。

        返回:
            numpy.ndarray RSI 值数组，长度与数据长度一致
        """
        size = len(self.data)
        close = np.array(self.data.close.get(size=size), dtype=np.float64)
        return talib.RSI(close, timeperiod=self.p.period)

    def next(self):
        # 确保数据足够计算 RSI + 需要前一根做比较
        if len(self.data) < self.p.period + 2:
            return

        rsi = self._calc_rsi()

        if not self.position:
            # 穿越确认：RSI 从 <30 回升到 >=30
            # rsi[-2] 是上一根 K 线的值，rsi[-1] 是当前值
            if rsi[-2] < self.p.oversold and rsi[-1] >= self.p.oversold:
                self.buy()
        else:
            if rsi[-1] > self.p.overbought:
                self.close()


if __name__ == '__main__':
    # 测试 4 只不同风格的标的：
    # 600519.SH 贵州茅台 - 消费白马股，波动相对温和
    # 688981.SH 中芯国际 - 科技股，波动较大
    # 000001.SZ 平安银行 - 银行股，低波动
    # 513100.SH 纳指ETF - 美股指数基金，趋势性强
    stocks = [
        ('600519.SH', '贵州茅台'),
        ('688981.SH', '中芯国际'),
        ('000001.SZ', '平安银行'),
        ('513100.SH', '纳指ETF'),
    ]

    print("=" * 70)
    print("RSI策略-优化 (talib.RSI + 穿越确认入场)")
    print("=" * 70)
    print("\n标准版: RSI<30立即买入, RSI>70卖出")
    print("优化版: 等RSI从低位回升穿过30线再买, 确认止跌反弹")
    print("  减少'接飞刀'风险, 降低回撤, 提高夏普比率\n")

    for code, name in stocks:
        print(f"\n--- {name} ({code}) ---")
        print("[标准版]")
        r1 = run_and_report(RSIStandard, code, '2025-01-01', '2025-12-31',
                            label=f'{name} RSI标准', plot=True)
        print("[优化版]")
        r2 = run_and_report(RSIOptimized, code, '2025-01-01', '2025-12-31',
                            label=f'{name} RSI优化', plot=True)

        # 对比优化效果
        diff_ret = r2['total_return'] - r1['total_return']
        diff_dd = r2['max_drawdown'] - r1['max_drawdown']
        tags = []
        if abs(diff_ret) > 0.005:  # 收益变化超过0.5%才报告
            tags.append(f"收益{diff_ret*100:+.1f}%")
        if abs(diff_dd) > 0.005:   # 回撤变化超过0.5%才报告
            tags.append(f"回撤{diff_dd*100:+.1f}%")
        sh1 = r1.get('sharpe_ratio', 0)
        sh2 = r2.get('sharpe_ratio', 0)
        if sh2 > sh1 + 0.05:       # 夏普提升超过0.05才报告
            tags.append("夏普提升")
        if tags:
            print(f"  -> 变化: {', '.join(tags)}")
