# -*- coding: utf-8 -*-
"""
MACD策略 - 趋势跟踪

类别: 趋势跟踪
逻辑: DIF上穿DEA(金叉) -> 买入; DIF下穿DEA(死叉) -> 卖出
参数: 短周期12, 长周期26, 信号线9

指标说明:
  MACD = 快线EMA(12) - 慢线EMA(26)     → 反映短期趋势与长期趋势的差值
  Signal = MACD的EMA(9)                 → 对MACD做平滑处理，形成信号线
  柱状图 = MACD - Signal                → 正值表示多头力量强，负值表示空头力量强

运行: python 2-MACD策略.py
"""
import backtrader as bt
from data_loader import run_and_report


class MACDStrategy(bt.Strategy):
    """MACD金叉/死叉策略

    交易信号:
      - 金叉 (MACD线上穿Signal线) → 买入，趋势转多
      - 死叉 (MACD线下穿Signal线) → 卖出，趋势转空
    """
    params = (
        ('short', 12),    # 快线周期 (短期EMA)
        ('long', 26),     # 慢线周期 (长期EMA)
        ('signal', 9),    # 信号线周期 (对MACD做平滑)
    )

    def __init__(self):
        # 计算MACD指标 (内置了 DIF/DEA/柱状图三条线)
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.short,     # 快线参数
            period_me2=self.p.long,      # 慢线参数
            period_signal=self.p.signal) # 信号线参数
        # 交叉信号: self.macd.macd = DIF线, self.macd.signal = DEA信号线
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        """每个交易日触发一次"""
        if not self.position:
            # 无持仓 + 金叉 (DIF上穿DEA) → 买入
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            # 有持仓 + 死叉 (DIF下穿DEA) → 卖出
            self.close()


if __name__ == '__main__':
    run_and_report(MACDStrategy, '600519.SH', '2025-01-01', '2025-12-31',
                   label='MACD策略', plot=True)
