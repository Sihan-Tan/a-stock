# -*- coding: utf-8 -*-
"""
自适应市场状态策略 -- ADX 判断趋势/震荡，自动切换子策略

这是本课程中最复杂的策略，体现了"市场状态识别 + 子策略匹配"的先进思想。

核心问题：
  - 趋势策略（如 MACD）在震荡市中频繁产生假信号，反复止损
  - 震荡策略（如 RSI）在趋势市中过早离场，抓不住大行情
  - 没有任何一个单一策略能在所有市场环境下表现良好

解决方案：
  用 ADX（平均趋向指数）自动判断当前市场处于什么状态，
  然后切换到最适合该状态的子策略。

市场状态与策略匹配：
  ADX > 25（趋势市） -> MACD 趋势跟踪策略（顺势而为）
  ADX < 20（震荡市） -> RSI 均值回归策略（高抛低吸）
  20-25（过渡区）    -> 观望不操作（市场方向不明，减少交易频率）

ADX 指标说明：
  ADX 本身不指示方向（是涨是跌），只表示趋势的强度：
    - ADX > 25：趋势强劲（无论上涨还是下跌）
    - ADX < 20：趋势疲弱（市场在震荡整理）
    - ADX 上升：趋势在加强
    - ADX 下降：趋势在减弱

使用 TA-Lib 计算的指标：
  - talib.ADX(high, low, close) -> 判断市场状态
  - talib.MACD(close) -> 趋势子策略的入场/出场信号
  - talib.RSI(close) -> 震荡子策略的入场/出场信号
  - talib.ATR(high, low, close) -> 动态止损距离

运行: python 8-自适应策略.py
"""
import numpy as np
import talib
import backtrader as bt
from data_loader import run_and_report


class AdaptiveStrategy(bt.Strategy):
    """
    自适应市场状态策略

    核心思想：不预测市场，而是识别当前市场状态并做出最合适的应对。

    策略参数：
      adx_period: ADX 计算周期，默认 14
      adx_trend: ADX 趋势阈值，高于此值判断为趋势市，默认 25
      adx_range: ADX 震荡阈值，低于此值判断为震荡市，默认 20
      atr_period: ATR 计算周期，用于动态止损，默认 14
      atr_mult: ATR 倍数止损，默认 2.0
      macd_fast/macd_slow/macd_signal: MACD 标准参数 12/26/9
      rsi_period: RSI 计算周期，默认 14

    两种模式：
      1. 趋势模式 (trend mode):
         - 入场：MACD 金叉
         - 出场：MACD 死叉 或 跌破 ATR 止损线
         - 适用：ADX > 25 的强趋势行情

      2. 震荡模式 (range mode):
         - 入场：RSI < 30（超卖）
         - 出场：RSI > 70（超买） 或 跌破 ATR 止损线
         - 适用：ADX < 20 的震荡行情

    ATR 动态止损：
      两种模式都使用 ATR 跟踪止损。
      止损价 = max(入场ATR止损, 最新价 - atr_mult * ATR)
      随着价格上涨，止损线不断上移，锁住利润。
    """
    params = (
        ('adx_period', 14),
        ('adx_trend', 25), ('adx_range', 20),
        ('atr_period', 14), ('atr_mult', 2.0),
        ('macd_fast', 12), ('macd_slow', 26), ('macd_signal', 9),
        ('rsi_period', 14),
    )

    def __init__(self):
        self._stop_price = 0.0  # 动态止损价
        self._mode = None       # 当前模式：'trend' 或 'range'

    def _calc_indicators(self):
        """
        使用 TA-Lib 一次性计算所有需要的指标

        包括 ADX、MACD、RSI、ATR 四个指标。
        在 next() 中每次调用，因为数据长度在不断增长。

        返回:
            adx, macd, signal, rsi, atr 五个 numpy 数组
        """
        size = len(self.data)
        high = np.array(self.data.high.get(size=size), dtype=np.float64)
        low = np.array(self.data.low.get(size=size), dtype=np.float64)
        close = np.array(self.data.close.get(size=size), dtype=np.float64)

        adx = talib.ADX(high, low, close, timeperiod=self.p.adx_period)
        macd, signal, _ = talib.MACD(close,
            fastperiod=self.p.macd_fast, slowperiod=self.p.macd_slow, signalperiod=self.p.macd_signal)
        rsi = talib.RSI(close, timeperiod=self.p.rsi_period)
        atr = talib.ATR(high, low, close, timeperiod=self.p.atr_period)

        return adx, macd, signal, rsi, atr

    def _is_golden_cross(self, macd, signal):
        """判断 MACD 金叉"""
        if len(macd) < 2 or np.isnan(macd[-1]) or np.isnan(macd[-2]):
            return False
        return macd[-2] <= signal[-2] and macd[-1] > signal[-1]

    def _is_dead_cross(self, macd, signal):
        """判断 MACD 死叉"""
        if len(macd) < 2 or np.isnan(macd[-1]) or np.isnan(macd[-2]):
            return False
        return macd[-2] >= signal[-2] and macd[-1] < signal[-1]

    def next(self):
        # 确保数据足够计算 MACD
        if len(self.data) < self.p.macd_slow + self.p.macd_signal:
            return

        # 计算所有指标
        adx, macd, signal, rsi, atr = self._calc_indicators()

        # ---- 第1步：判断市场状态 ----
        adx_val = adx[-1]
        if np.isnan(adx_val):
            return
        if adx_val > self.p.adx_trend:
            regime = 'trend'    # 趋势市：ADX > 25
        elif adx_val < self.p.adx_range:
            regime = 'range'    # 震荡市：ADX < 20
        else:
            regime = 'neutral'  # 过渡区：20 <= ADX <= 25，观望

        # 获取 ATR 值，用于止损计算
        atr_val = atr[-1]
        if np.isnan(atr_val):
            atr_val = 0.0

        # ---- 第2步：入场逻辑 ----
        if not self.position:
            if regime == 'trend' and self._is_golden_cross(macd, signal):
                # 趋势市 + MACD 金叉 -> 趋势跟踪入场
                self.buy()
                # 初始止损价 = 入场价 - atr_mult * ATR
                self._stop_price = self.data.close[0] - self.p.atr_mult * atr_val
                self._mode = 'trend'  # 标记为趋势模式

            elif regime == 'range' and not np.isnan(rsi[-1]) and rsi[-1] < 30:
                # 震荡市 + RSI 超卖 -> 均值回归入场
                self.buy()
                self._stop_price = self.data.close[0] - self.p.atr_mult * atr_val
                self._mode = 'range'  # 标记为震荡模式

        # ---- 第3步：出场逻辑 ----
        else:
            # 动态调整止损价：用最高价计算新的止损，保留最大值（只上移不下移）
            new_stop = self.data.close[0] - self.p.atr_mult * atr_val
            self._stop_price = max(self._stop_price, new_stop)
            # 当前价是否跌破止损
            stop_hit = self.data.close[0] < self._stop_price

            if self._mode == 'trend':
                # 趋势模式出场：死叉 或 跌破止损
                if self._is_dead_cross(macd, signal) or stop_hit:
                    self.close()
                    self._mode = None
            elif self._mode == 'range':
                # 震荡模式出场：RSI 超买 或 跌破止损
                if (not np.isnan(rsi[-1]) and rsi[-1] > 70) or stop_hit:
                    self.close()
                    self._mode = None


# ============================================================
# 对照策略：纯 MACD 策略（不分市场环境）
# ============================================================

class PureMACDStrategy(bt.Strategy):
    """
    纯 MACD 策略（对照基准）

    不分趋势市还是震荡市，一律使用 MACD 金叉死叉信号。
    在震荡市中表现往往不佳（频繁假信号）。
    """
    params = (('fast', 12), ('slow', 26), ('signal', 9))

    def __init__(self):
        pass

    def _calc_macd(self):
        """用 TA-Lib 计算 MACD"""
        size = len(self.data)
        close = np.array(self.data.close.get(size=size), dtype=np.float64)
        macd, signal, _ = talib.MACD(close,
            fastperiod=self.p.fast, slowperiod=self.p.slow, signalperiod=self.p.signal)
        return macd, signal

    def _is_golden_cross(self, macd, signal):
        if len(macd) < 2 or np.isnan(macd[-1]) or np.isnan(macd[-2]):
            return False
        return macd[-2] <= signal[-2] and macd[-1] > signal[-1]

    def _is_dead_cross(self, macd, signal):
        if len(macd) < 2 or np.isnan(macd[-1]) or np.isnan(macd[-2]):
            return False
        return macd[-2] >= signal[-2] and macd[-1] < signal[-1]

    def next(self):
        if len(self.data) < self.p.slow + self.p.signal:
            return

        macd, signal = self._calc_macd()

        if not self.position:
            if self._is_golden_cross(macd, signal):
                self.buy()
        elif self._is_dead_cross(macd, signal):
            self.close()


if __name__ == '__main__':
    stock = '600519.SH'
    start = '2024-01-01'
    end = '2025-12-31'

    print("=" * 60)
    print("自适应策略 vs 纯MACD策略")
    print("=" * 60)

    print("\nADX市场状态判断:")
    print("  ADX > 25: 趋势市(用MACD跟踪趋势)")
    print("  ADX < 20: 震荡市(用RSI抄底逃顶)")
    print("  20-25:    过渡区(观望)\n")

    print("[纯MACD] 不分市场环境:")
    run_and_report(PureMACDStrategy, stock, start, end, label='纯MACD', plot=True)

    print("\n[自适应] ADX状态切换:")
    run_and_report(AdaptiveStrategy, stock, start, end, label='自适应策略', plot=True)
