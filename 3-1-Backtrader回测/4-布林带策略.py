# -*- coding: utf-8 -*-
"""
布林带策略 - 波动率

类别: 波动率
逻辑: 价格触及下轨(均线-2倍标准差) -> 买入; 触及上轨 -> 卖出
参数: 周期20, 标准差倍数2.0

指标说明:
  中轨 = N日移动平均线(SMA)
  上轨 = 中轨 + K × N日标准差
  下轨 = 中轨 - K × N日标准差
  价格在上下轨之间波动的概率约95%(K=2时)，突破轨道意味着异常波动

运行: python 4-布林带策略.py
"""
import backtrader as bt
from data_loader import run_and_report


class BollingerBandStrategy(bt.Strategy):
    """布林带突破策略

    交易信号:
      - 收盘价跌破下轨 → 买入，价格处于统计低位，有望回归中轨
      - 收盘价突破上轨 → 卖出，价格处于统计高位，锁定利润
    """
    params = (
        ('period', 20),         # 均线周期 (布林带默认20日)
        ('devfactor', 2.0),     # 标准差倍数 (2倍覆盖约95%的价格波动)
    )

    def __init__(self):
        # 计算布林带 (包含 mid/中线, top/上轨, bot/下轨)
        self.boll = bt.indicators.BollingerBands(
            self.data.close,
            period=self.p.period,
            devfactor=self.p.devfactor)

    def next(self):
        """每个交易日触发一次"""
        if not self.position:
            # 无持仓 + 收盘价跌破下轨 → 买入 (超跌反弹机会)
            if self.data.close[0] < self.boll.bot[0]:
                self.buy()
        elif self.data.close[0] > self.boll.top[0]:
            # 有持仓 + 收盘价突破上轨 → 卖出 (超涨止盈)
            self.close()


if __name__ == '__main__':
    run_and_report(BollingerBandStrategy, '600519.SH', '2025-01-01', '2025-12-31',
                   label='布林带策略', plot=True)
