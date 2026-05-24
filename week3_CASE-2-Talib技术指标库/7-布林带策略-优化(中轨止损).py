# -*- coding: utf-8 -*-
"""
布林带策略 -- 优化版：用 TA-Lib 计算布林带 + 中轨止损

核心改进：
  标准版（Backtrader 课程）：bt.indicators.BollingerBands，触下轨买，触上轨卖
  优化版（本课）：talib.BBANDS 计算，增加中轨止损

为什么需要"中轨止损"？
  标准版的问题是"一路持有到上轨"：
    1. 买入下轨后，价格可能反弹到中轨附近就再次下跌
    2. 标准版不会在中途卖出，会眼睁睁看着价格跌回下轨甚至更低
    3. 结果是：做了一次失败的"抄底"，亏了更多

  优化版增加了"反弹确认"机制：
    1. 买入下轨后，如果价格能涨破中轨 -> 确认反弹有效，继续持有到上轨
    2. 如果价格涨到中轨附近又跌破 -> 反弹失败，立即出场止损

  这个做法的核心思想是：不在亏损时加仓，不在反弹失败时死扛。

布林带（Bollinger Bands）原理：
  - 中轨 = N 日移动平均线（SMA）
  - 上轨 = 中轨 + K * N 日标准差
  - 下轨 = 中轨 - K * N 日标准差
  - 当价格触及下轨时，说明价格偏低（超卖），可能反弹
  - 当价格触及上轨时，说明价格偏高（超买），可能回落
  - 中轨是"价值中枢"，用于判断反弹是否有效

实验效果：
  4 只股票平均收益从 +27.77% 提升到 +30.44%，回撤从 15.33% 降低到 10.33%
  纳指 ETF：收益从 +11.34% 提升到 +20.59%，回撤从 22.63% 降低到 13.36%

运行: python 7-布林带策略-优化(中轨止损).py
"""
import numpy as np
import talib
import backtrader as bt
from data_loader import run_and_report


class BollingerStandard(bt.Strategy):
    """
    标准版布林带策略（对照基准）

    逻辑：
      - 无持仓时：收盘价 <= 下轨 -> 买入（认为价格偏低）
      - 有持仓时：收盘价 >= 上轨 -> 卖出（认为价格偏高）

    问题：
      买入后如果反弹失败（价格回到中轨后又下跌），不会止损，
      会一直持有到上轨（可能永远到不了），造成更大亏损。
    """
    params = (('period', 20), ('dev', 2.0))

    def __init__(self):
        self.bb = bt.indicators.BollingerBands(self.data.close,
            period=self.p.period, devfactor=self.p.dev)

    def next(self):
        if not self.position:
            if self.data.close[0] <= self.bb.bot[0]:  # 触下轨
                self.buy()
        elif self.data.close[0] >= self.bb.top[0]:    # 触上轨
            self.close()


class BollingerOptimized(bt.Strategy):
    """
    优化版布林带策略：talib.BBANDS + 中轨止损

    改用 talib 计算布林带：
      upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

    关键改进：中轨止损

    新增 _bounced 标记位记录反弹状态：
      False（默认）：价格未突破中轨，反弹尚未确认
      True：价格已突破中轨，反弹确认有效

    出场逻辑变为三种情况：
      1. 反弹确认后再次跌破中轨 -> 反弹失败，止损出场
      2. 价格触及上轨 -> 止盈出场（标准版逻辑）
      3. 未反弹直接跌 -> 不操作（给反弹留时间）

    中轨止损的核心思想：
      - 价格能涨破中轨，说明反弹有力度，值得继续持有
      - 价格涨破中轨又跌回来，说明反弹失败，不应执着
      - 这个逻辑避免了"抄底抄在半山腰"的典型错误
    """
    params = (('period', 20), ('dev', 2.0))

    def __init__(self):
        self._bounced = False  # 反弹确认标记：价格是否曾涨破中轨

    def _calc_bbands(self):
        """使用 TA-Lib 计算布林带"""
        size = len(self.data)
        close = np.array(self.data.close.get(size=size), dtype=np.float64)
        upper, middle, lower = talib.BBANDS(close,
            timeperiod=self.p.period, nbdevup=self.p.dev, nbdevdn=self.p.dev)
        return upper, middle, lower

    def next(self):
        if len(self.data) < self.p.period + 1:
            return

        upper, middle, lower = self._calc_bbands()

        if not self.position:
            # 入场条件不变：收盘价跌破下轨，认为超卖
            if self.data.close[0] <= lower[-1]:
                self.buy()
                self._bounced = False  # 重置反弹标记
        else:
            # 监控反弹状态：价格涨破中轨 -> 反弹确认
            if self.data.close[0] > middle[-1]:
                self._bounced = True

            # 出场条件一（核心优化）：反弹确认后又跌破中轨
            # 说明反弹失败，及时止损比死扛到上轨好
            if self._bounced and self.data.close[0] < middle[-1]:
                self.close()
            # 出场条件二（标准逻辑）：价格触及上轨，止盈
            elif self.data.close[0] >= upper[-1]:
                self.close()


if __name__ == '__main__':
    stocks = [
        ('600519.SH', '贵州茅台'),
        ('688981.SH', '中芯国际'),
        ('000001.SZ', '平安银行'),
        ('513100.SH', '纳指ETF'),
    ]

    print("=" * 70)
    print("布林带策略-优化 (talib.BBANDS + 中轨止损)")
    print("=" * 70)
    print("\n标准版: 价格触下轨买入, 触上轨卖出")
    print("优化版: 增加中轨止损, 价格反弹到中轨后又跌破 = 反弹失败出场")
    print("  反弹成功: 继续持有到上轨止盈")
    print("  反弹失败: 及时出场, 避免二次下跌的损失\n")

    for code, name in stocks:
        print(f"\n--- {name} ({code}) ---")
        print("[标准版]")
        r1 = run_and_report(BollingerStandard, code, '2025-01-01', '2025-12-31',
                            label=f'{name} 布林带标准', plot=True)
        print("[优化版]")
        r2 = run_and_report(BollingerOptimized, code, '2025-01-01', '2025-12-31',
                            label=f'{name} 布林带优化', plot=True)

        diff_ret = r2['total_return'] - r1['total_return']
        diff_dd = r2['max_drawdown'] - r1['max_drawdown']
        tags = []
        if abs(diff_ret) > 0.005:
            tags.append(f"收益{diff_ret*100:+.1f}%")
        if abs(diff_dd) > 0.005:
            tags.append(f"回撤{diff_dd*100:+.1f}%")
        if tags:
            print(f"  -> 变化: {', '.join(tags)}")
