# -*- coding: utf-8 -*-
"""
RSI策略 - 超买超卖

类别: 超买超卖
逻辑: RSI < 30(超卖) -> 买入; RSI > 70(超买) -> 卖出
参数: RSI周期14, 超卖线30, 超买线70

指标说明:
  RSI (相对强弱指标) = 100 - 100/(1 + RS)
  RS = N日内平均涨幅 / N日内平均跌幅
  RSI 范围 0~100，数值越高表示近期涨幅越强

运行: python 3-RSI策略.py
"""
import backtrader as bt
from data_loader import run_and_report


class RSIStrategy(bt.Strategy):
    """RSI超买超卖策略

    交易信号:
      - RSI < 超卖线(30) → 买入，市场过度恐慌，价格可能反弹
      - RSI > 超买线(70) → 卖出，市场过度乐观，价格可能回调
    """
    params = (
        ('period', 14),         # RSI计算周期 (通常用14日)
        ('oversold', 30),       # 超卖阈值 (RSI低于此值视为超卖)
        ('overbought', 70),     # 超买阈值 (RSI高于此值视为超买)
    )

    def __init__(self):
        # 计算14日RSI指标
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)

    def next(self):
        """每个交易日触发一次"""
        if not self.position:
            # 无持仓 + RSI进入超卖区 → 买入 (抄底)
            if self.rsi < self.p.oversold:
                self.buy()
        elif self.rsi > self.p.overbought:
            # 有持仓 + RSI进入超买区 → 卖出 (止盈)
            self.close()


if __name__ == '__main__':
    run_and_report(RSIStrategy, '600519.SH', '2025-01-01', '2025-12-31',
                   label='RSI策略', plot=True)
