# -*- coding: utf-8 -*-
"""
MACD 策略 -- 优化版：用 TA-Lib 计算 MACD + 成交量确认入场

核心改进：
  标准版（Backtrader 课程）：bt.indicators.MACD，金叉买，死叉卖
  优化版（本课）：talib.MACD + talib.SMA，增加成交量确认

为什么需要"成交量确认"？
  MACD 金叉的本质是短期均线上穿长期均线，但如果金叉时成交量没有放大：
    1. 说明市场缺乏资金推动，这个金叉可能是"虚假信号"
    2. 没有成交量支撑的上涨难以持续，很容易再次死叉
    3. 在缩量震荡行情中，金叉死叉频繁出现，按信号操作会频繁亏损

优化逻辑：
  成交量是价格的"燃料"。
  金叉 + 成交量放大 = 有资金支持的上涨趋势，信号更可靠。
  金叉 + 成交量萎缩 = 假突破的概率高，应过滤掉。

TA-Lib 在策略中的用法：
  macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
  vol_ma = talib.SMA(volume, timeperiod=20)

运行: python 5-MACD策略-优化(成交量确认).py
"""
import numpy as np
import talib
import backtrader as bt
from data_loader import run_and_report


class MACDStandard(bt.Strategy):
    """
    标准版 MACD 策略（与 Backtrader 课程一致）

    逻辑：
      - 无持仓时：MACD 金叉（DIF 上穿 DEA）-> 买入
      - 有持仓时：MACD 死叉（DIF 下穿 DEA）-> 卖出

    问题：
      在震荡行情中，MACD 频繁金叉死叉，导致频繁交易和亏损。
      缩量金叉往往是假信号。
    """
    params = (('fast', 12), ('slow', 26), ('signal', 9))

    def __init__(self):
        self.macd = bt.indicators.MACD(self.data.close,
            period_me1=self.p.fast, period_me2=self.p.slow, period_signal=self.p.signal)
        # CrossOver 指标：golden_cross > 0 表示金叉，< 0 表示死叉
        self.cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        if not self.position:
            if self.cross[0] > 0:   # 金叉
                self.buy()
        elif self.cross[0] < 0:     # 死叉
            self.close()


class MACDOptimized(bt.Strategy):
    """
    优化版 MACD 策略：talib.MACD + 成交量确认入场

    改用 talib 计算指标：
      talib.MACD(close) -> MACD 指标
      talib.SMA(volume, 20) -> 成交量 20 日均线

    成交量确认逻辑：
      金叉时，要求当日成交量 > vol_mult * 成交量 20 日均值
      默认 vol_mult = 0.9（即成交量不低于均量的 90%）
      这个阈值可以根据不同标的调整：
        换手率高的股票（如科技股）可以设高一些
        换手率低的股票（如银行股）可以设低一些

    TA-Lib 与 Backtrader 的混合使用：
      虽然本策略用 talib 计算指标，但回测框架仍用 Backtrader，
      这样可以同时享受 TA-Lib 的计算效率和 Backtrader 的回测便利性。
    """
    params = (('fast', 12), ('slow', 26), ('signal', 9),
              ('vol_period', 20), ('vol_mult', 0.9))

    def _calc(self):
        """
        使用 TA-Lib 一次性计算 MACD 和成交量均线

        TA-Lib 需要完整的 numpy float64 数组。
        用 get(size=size) 从 Backtrader 获取完整价格和成交量数据。

        返回:
            macd: DIF 线数组
            signal: DEA 线数组
            volume: 原始成交量数组
            vol_ma: 成交量 20 日移动平均数组
        """
        size = len(self.data)
        close = np.array(self.data.close.get(size=size), dtype=np.float64)
        volume = np.array(self.data.volume.get(size=size), dtype=np.float64)
        macd, signal, hist = talib.MACD(close,
            fastperiod=self.p.fast, slowperiod=self.p.slow, signalperiod=self.p.signal)
        vol_ma = talib.SMA(volume, timeperiod=self.p.vol_period)
        return macd, signal, volume, vol_ma

    def _is_golden_cross(self, macd, signal):
        """
        判断 MACD 是否金叉：DIF 上穿 DEA

        金叉 = 前一根 K 线 DIF <= DEA，当前 K 线 DIF > DEA
        需要至少 2 根 K 线的数据。
        """
        if len(macd) < 2 or np.isnan(macd[-1]) or np.isnan(macd[-2]):
            return False
        return macd[-2] <= signal[-2] and macd[-1] > signal[-1]

    def _is_dead_cross(self, macd, signal):
        """
        判断 MACD 是否死叉：DIF 下穿 DEA

        死叉 = 前一根 K 线 DIF >= DEA，当前 K 线 DIF < DEA
        """
        if len(macd) < 2 or np.isnan(macd[-1]) or np.isnan(macd[-2]):
            return False
        return macd[-2] >= signal[-2] and macd[-1] < signal[-1]

    def next(self):
        # 确保数据足够计算 MACD（至少需要 slow + signal 根 K 线）
        if len(self.data) < self.p.slow + self.p.signal:
            return

        macd, signal, volume, vol_ma = self._calc()

        if not self.position:
            if self._is_golden_cross(macd, signal):
                # 核心优化：成交量确认
                # 当日成交量 > 0.9 * 20日均量，说明有资金支持
                vol_ok = not np.isnan(vol_ma[-1]) and volume[-1] > vol_ma[-1] * self.p.vol_mult
                if vol_ok:
                    self.buy()
        else:
            if self._is_dead_cross(macd, signal):
                self.close()


if __name__ == '__main__':
    stocks = [
        ('600519.SH', '贵州茅台'),
        ('688981.SH', '中芯国际'),
        ('000001.SZ', '平安银行'),
        ('513100.SH', '纳指ETF'),
    ]

    print("=" * 70)
    print("MACD策略-优化 (talib.MACD + 成交量确认入场)")
    print("=" * 70)
    print("\n标准版: MACD金叉买入, 死叉卖出")
    print("优化版: 金叉 + 成交量>0.9倍20日均量时买入, 过滤缩量假信号")
    print("  talib.MACD 计算MACD指标, talib.SMA 计算成交量均线\n")

    all_std, all_opt = [], []
    for code, name in stocks:
        print(f"\n--- {name} ({code}) ---")
        print("[标准版]")
        r1 = run_and_report(MACDStandard, code, '2025-01-01', '2025-12-31',
                            label=f'{name} MACD标准', plot=True)
        print("[优化版]")
        r2 = run_and_report(MACDOptimized, code, '2025-01-01', '2025-12-31',
                            label=f'{name} MACD优化', plot=True)
        all_std.append(r1)
        all_opt.append(r2)

        diff_ret = r2['total_return'] - r1['total_return']
        diff_dd = r2['max_drawdown'] - r1['max_drawdown']
        tags = []
        if abs(diff_ret) > 0.005:
            tags.append(f"收益{diff_ret*100:+.1f}%")
        if abs(diff_dd) > 0.005:
            tags.append(f"回撤{diff_dd*100:+.1f}%")
        if tags:
            print(f"  -> 变化: {', '.join(tags)}")

    # 整体平均对比
    print(f"\n{'='*70}")
    print("平均对比")
    print(f"{'='*70}")
    avg = lambda lst, k: np.mean([r[k] for r in lst])
    print(f"  标准版: 平均收益 {avg(all_std,'total_return')*100:+.2f}%  平均回撤 {avg(all_std,'max_drawdown')*100:.2f}%")
    print(f"  优化版: 平均收益 {avg(all_opt,'total_return')*100:+.2f}%  平均回撤 {avg(all_opt,'max_drawdown')*100:.2f}%")
