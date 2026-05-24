# -*- coding: utf-8 -*-
"""
动量策略 - 动量因子

类别: 动量因子
逻辑: N日涨幅 > 5% -> 买入; N日跌幅 > 5% -> 卖出
参数: 观察周期20日, 阈值5%

指标说明:
  动量 (ROC, Rate of Change) = (当日收盘价 / N日前收盘价 - 1) × 100
  正值表示价格上涨，负值表示下跌
  动量效应: 近期上涨的股票倾向于继续上涨，下跌的倾向于继续下跌

运行: python 6-动量策略.py
"""
import backtrader as bt
from data_loader import run_and_report


class MomentumStrategy(bt.Strategy):
    """动量追逐策略

    交易信号:
      - N日涨幅 > 正阈值 → 买入，跟随上涨趋势
      - N日跌幅 > 负阈值 → 卖出，趋势反转止损
    """
    params = (
        ('period', 20),         # 观察周期 (计算过去N日的涨跌幅)
        ('threshold', 5.0),     # 涨跌阈值(%) (超过此幅度触发交易)
    )

    def __init__(self):
        # ROC100 = 涨跌幅百分比 (已乘以100，如5表示涨了5%)
        self.roc = bt.indicators.ROC100(self.data.close, period=self.p.period)

    def next(self):
        """每个交易日触发一次"""
        if not self.position:
            # 无持仓 + N日涨幅超过阈值 → 买入 (追涨，顺势而为)
            if self.roc[0] > self.p.threshold:
                self.buy()
        elif self.roc[0] < -self.p.threshold:
            # 有持仓 + N日跌幅超过阈值 → 卖出 (止损，趋势转弱)
            self.close()


if __name__ == '__main__':
    run_and_report(MomentumStrategy, '600519.SH', '2025-01-01', '2025-12-31',
                   label='动量策略', plot=True)
