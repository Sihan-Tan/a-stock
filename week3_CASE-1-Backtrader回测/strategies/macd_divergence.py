# -*- coding: utf-8 -*-
"""
MACD底背离策略 - 自定义策略示例

使用规范:
  1. 在 strategies/ 目录下创建 .py 文件
  2. 定义 STRATEGY_META 字典（策略元信息）
  3. 定义 Strategy 类，继承 backtrader.Strategy
  4. 运行 7-自定义策略.py 即可自动加载

策略逻辑:
  价格创N日新低但MACD柱未创新低（底背离），说明下跌动能衰竭，买入
  MACD死叉时卖出

底背离原理:
  价格低点不断下移(new low < old low)，但指标低点反而上移(higher low)
  → 下跌趋势的动能正在减弱，趋势即将反转向上
"""

import backtrader as bt

# ============================================================
# 策略元信息 - 必须定义，供系统自动注册和展示
# ============================================================
STRATEGY_META = {
    'name': 'MACD底背离',                                                            # 策略显示名称
    'category': 'custom',                                                            # 策略分类
    'desc': '价格创新低但MACD未创新低时买入，捕捉趋势反转的经典背离策略',              # 策略描述
    'params': {'lookback': 30, 'fast': 12, 'slow': 26, 'signal': 9},                # 可调参数
    'params_desc': '观察周期30日, MACD参数(12,26,9)',                                 # 参数说明
    'logic': '价格创N日新低且MACD未创新低(底背离) -> 买入; MACD死叉 -> 卖出',          # 交易逻辑
}


# ============================================================
# 策略类 - 必须命名为 Strategy，继承 bt.Strategy
# ============================================================
class Strategy(bt.Strategy):
    """MACD底背离策略

    买入条件 (同时满足):
      1. 价格在N日最低点附近 (当前最低价 ≤ N日最低价 × 1.01)
      2. MACD值高于N日最低值 (MACD未随价格创新低，形成背离)
      3. MACD金叉 (MACD线上穿Signal线，确认上涨动能)

    卖出条件:
      MACD死叉 (MACD线下穿Signal线，趋势转空)
    """
    params = (
        ('lookback', 30),     # 背离观察周期 (向前看N日的高低点)
        ('fast', 12),          # MACD快线周期
        ('slow', 26),          # MACD慢线周期
        ('signal', 9),         # MACD信号线周期
    )

    def __init__(self):
        # MACD指标 (包含 macd线 / signal线 / 柱状图)
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.fast,
            period_me2=self.p.slow,
            period_signal=self.p.signal)

        # N日内最低价 (用于判断价格是否在低位)
        self.price_lowest = bt.indicators.Lowest(
            self.data.low, period=self.p.lookback)

        # N日内MACD最低值 (用于判断MACD是否跟随价格创新低)
        self.macd_lowest = bt.indicators.Lowest(
            self.macd.macd, period=self.p.lookback)

    def next(self):
        """每个交易日触发一次"""
        if not self.position:
            # ---- 买入条件 ----
            # 条件1: 当日最低价接近N日最低价 (在1%误差范围内)
            at_price_low = self.data.low[0] <= self.price_lowest[0] * 1.01

            # 条件2: MACD值未随价格创新低 (> N日最低值的80%，留20%容差)
            #        这是底背离的核心: 价格低但MACD不低
            macd_higher = self.macd.macd[0] > self.macd_lowest[0] * 0.8

            # 条件3: MACD金叉确认 (macd线上穿signal线)
            golden_cross = self.macd.macd[0] > self.macd.signal[0]

            # 三个条件同时满足 → 买入
            if at_price_low and macd_higher and golden_cross:
                self.buy()
        else:
            # ---- 卖出条件 ----
            # MACD死叉: 当日macd线 < signal线 且 昨日macd线 >= signal线
            if (self.macd.macd[0] < self.macd.signal[0] and
                    self.macd.macd[-1] >= self.macd.signal[-1]):
                self.close()
