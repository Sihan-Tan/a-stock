# -*- coding: utf-8 -*-
"""
乖离率策略 - 均值回归

类别: 均值回归
逻辑: 乖离率 < -6%(超跌) -> 买入; 乖离率 > 6%(超涨) -> 卖出
参数: 均线周期20, 买入阈值-6%, 卖出阈值6%

指标说明:
  乖离率 (BIAS) = (收盘价 - 均线) / 均线 × 100%
  正值表示价格在均线上方，负值表示在均线下方
  均值回归理论: 价格总是围绕均线波动，偏离过大时会回归

运行: python 5-乖离率策略.py
"""
import backtrader as bt
from data_loader import run_and_report


class BIASStrategy(bt.Strategy):
    """乖离率均值回归策略

    交易信号:
      - 乖离率 < 买入阈值(-6%) → 买入，价格远低于均线，有望反弹回归
      - 乖离率 > 卖出阈值(3%)  → 卖出，价格高于均线，止盈离场
    """
    params = (
        ('period', 20),             # 均线周期 (计算乖离率的基准均线)
        ('buy_threshold', -6.0),    # 买入阈值(%) (乖离率低于此值买入)
        ('sell_threshold', 3.0),    # 卖出阈值(%) (乖离率高于此值卖出)
    )

    def __init__(self):
        # 计算20日简单移动平均线作为基准
        self.sma = bt.indicators.SMA(self.data.close, period=self.p.period)

    def next(self):
        """每个交易日触发一次"""
        # 计算当日乖离率 = (收盘价 - 均线) / 均线 × 100
        bias = (self.data.close[0] - self.sma[0]) / self.sma[0] * 100

        if not self.position:
            # 无持仓 + 乖离率低于买入阈值 → 买入 (超跌抄底)
            if bias < self.p.buy_threshold:
                self.buy()
        elif bias > self.p.sell_threshold:
            # 有持仓 + 乖离率高于卖出阈值 → 卖出 (回归均线后止盈)
            self.close()


if __name__ == '__main__':
    run_and_report(BIASStrategy, '600519.SH', '2025-01-01', '2025-12-31',
                   label='乖离率策略', plot=True)
