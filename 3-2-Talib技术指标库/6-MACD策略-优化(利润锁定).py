# -*- coding: utf-8 -*-
"""
MACD 策略 -- 优化版（利润锁定）：用 TA-Lib 计算 MACD + 利润锁定出场

核心改进：
  标准版（Backtrader 课程）：bt.indicators.MACD，金叉买，死叉卖
  优化版（本课）：talib.MACD 计算，增加利润锁定出场

为什么需要"利润锁定"？
  MACD 死叉（卖出信号）存在严重的滞后问题：
    1. MACD 是滞后指标 — DIF 和 DEA 都是均线的派生指标
    2. 当死叉出现时，价格往往已经大幅下跌，利润大幅回吐
    3. 例如：股价涨了 20%，然后回调 5% 时死叉才出现，实际只锁住 15% 的利润

利润锁定的思路：
  与其等死叉出现（可能已经回吐很多利润），不如主动设置利润锁定规则：
    1. 盈利超过 profit_trigger%（默认 5%）后，开始监控回撤
    2. 从持仓期间最高点回落 trail_pct%（默认 3%）时，立即出场
  这叫"移动止盈"（Trailing Stop）——让利润奔跑的同时锁定大部分收益。

双保险出场：
  - 利润锁定触发 -> 主动出场，锁住利润
  - 死叉触发 -> 被动出场，防止亏损扩大
  两种条件任一满足就出场，确保盈利时能锁住，亏损时能止损。

运行: python 6-MACD策略-优化(利润锁定).py
"""
import numpy as np
import talib
import backtrader as bt
from data_loader import run_and_report


class MACDStandard(bt.Strategy):
    """
    标准版 MACD 策略（对照基准）

    逻辑与 5-MACD策略-优化(成交量确认).py 中的标准版一致。
    """
    params = (('fast', 12), ('slow', 26), ('signal', 9))

    def __init__(self):
        self.macd = bt.indicators.MACD(self.data.close,
            period_me1=self.p.fast, period_me2=self.p.slow, period_signal=self.p.signal)
        self.cross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        if not self.position:
            if self.cross[0] > 0:
                self.buy()
        elif self.cross[0] < 0:
            self.close()


class MACDProfitLock(bt.Strategy):
    """
    优化版 MACD 策略：talib.MACD + 利润锁定出场

    入场逻辑不变：MACD 金叉买入

    出场逻辑变为双保险：
      1. 利润锁定出场（主动）：盈利 > profit_trigger% 后，从高点回撤 trail_pct%
         就出场。这避免了死叉滞后导致的利润大幅回吐。
      2. 死叉出场（被动）：如果未触发利润锁定但出现死叉，仍然出场。
         这保证了在亏损或小赚时不会无限期持仓。

    params:
      profit_trigger: 触发利润锁定的最低盈利百分比（默认 5%）
      trail_pct: 从高点回撤多少百分比就出场（默认 3%）

    利润锁定的优点：
      - 盈利 10% 时：从高点回落 3% -> 锁住约 7% 的利润
      - 盈利 20% 时：从高点回落 3% -> 锁住约 17% 的利润
      - 让利润奔跑，同时锁定大部分收益
    """
    params = (('fast', 12), ('slow', 26), ('signal', 9),
              ('profit_trigger', 5.0), ('trail_pct', 3.0))

    def __init__(self):
        # 记录入场价格和持仓期间的最高价
        self._entry_price = 0.0   # 开仓价格
        self._peak_price = 0.0    # 持仓期间的最高价（用于利润锁定计算）

    def _calc_macd(self):
        """使用 TA-Lib 计算 MACD"""
        size = len(self.data)
        close = np.array(self.data.close.get(size=size), dtype=np.float64)
        macd, signal, hist = talib.MACD(close,
            fastperiod=self.p.fast, slowperiod=self.p.slow, signalperiod=self.p.signal)
        return macd, signal

    def _is_golden_cross(self, macd, signal):
        """判断 MACD 金叉：DIF 上穿 DEA"""
        if len(macd) < 2 or np.isnan(macd[-1]) or np.isnan(macd[-2]):
            return False
        return macd[-2] <= signal[-2] and macd[-1] > signal[-1]

    def _is_dead_cross(self, macd, signal):
        """判断 MACD 死叉：DIF 下穿 DEA"""
        if len(macd) < 2 or np.isnan(macd[-1]) or np.isnan(macd[-2]):
            return False
        return macd[-2] >= signal[-2] and macd[-1] < signal[-1]

    def next(self):
        if len(self.data) < self.p.slow + self.p.signal:
            return

        macd, signal = self._calc_macd()

        if not self.position:
            # 入场：MACD 金叉
            if self._is_golden_cross(macd, signal):
                self.buy()
                self._entry_price = self.data.close[0]  # 记录入场价
                self._peak_price = self.data.close[0]    # 初始最高价=入场价
        else:
            # 更新持仓期间的最高价
            self._peak_price = max(self._peak_price, self.data.close[0])

            # 计算盈利比例（基于最高价）
            gain = (self._peak_price - self._entry_price) / self._entry_price * 100
            # 计算从最高点的回撤比例
            drop = (self._peak_price - self.data.close[0]) / self._peak_price * 100

            # 利润锁定出场条件：
            #   盈利已超过 profit_trigger%，且从最高点回落超过 trail_pct%
            if gain >= self.p.profit_trigger and drop >= self.p.trail_pct:
                self.close()
            # 死叉出场（备选）：
            #   未触发利润锁定时，用死叉作为最后的出场信号
            elif self._is_dead_cross(macd, signal):
                self.close()


if __name__ == '__main__':
    stocks = [
        ('600519.SH', '贵州茅台'),
        ('688981.SH', '中芯国际'),
        ('000001.SZ', '平安银行'),
        ('513100.SH', '纳指ETF'),
    ]

    print("=" * 70)
    print("MACD策略-优化(利润锁定) (talib.MACD + 利润锁定出场)")
    print("=" * 70)
    print("\n标准版: MACD金叉买入, 死叉卖出")
    print("优化版: 同样金叉买入, 盈利>5%后从高点回撤>3%出场锁利")
    print("  死叉信号滞后, 利润锁定让盈利交易更早兑现利润\n")

    all_std, all_opt = [], []
    for code, name in stocks:
        print(f"\n--- {name} ({code}) ---")
        print("[标准版]")
        r1 = run_and_report(MACDStandard, code, '2025-01-01', '2025-12-31',
                            label=f'{name} MACD标准', plot=True)
        print("[优化版]")
        r2 = run_and_report(MACDProfitLock, code, '2025-01-01', '2025-12-31',
                            label=f'{name} MACD利润锁定', plot=True)
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

    print(f"\n{'='*70}")
    print("平均对比")
    print(f"{'='*70}")
    avg = lambda lst, k: np.mean([r[k] for r in lst])
    print(f"  标准版: 平均收益 {avg(all_std,'total_return')*100:+.2f}%  平均回撤 {avg(all_std,'max_drawdown')*100:.2f}%")
    print(f"  优化版: 平均收益 {avg(all_opt,'total_return')*100:+.2f}%  平均回撤 {avg(all_opt,'max_drawdown')*100:.2f}%")
