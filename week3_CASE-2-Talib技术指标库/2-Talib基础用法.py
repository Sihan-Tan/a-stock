# -*- coding: utf-8 -*-
"""
TA-Lib 基础用法 -- 快速入门

本脚本演示 TA-Lib 的核心功能，适合初学者初步了解：
  1. 验证 TA-Lib 安装是否成功
  2. SMA / MACD / RSI / ATR 四个最常用指标的基本调用
  3. 每个指标都附有手算验证和原理说明
  4. 一览 TA-Lib 的 158 种函数分类

TA-Lib 的特点：
  - 完全由 C 语言实现，计算性能极高
  - 接口统一：所有函数都接收 numpy 数组，返回 numpy 数组
  - 涵盖 158 种技术指标，包括均线、动量、波动率、成交量、形态识别等
  - K 线形态识别（CDL_* 系列）是 TA-Lib 独有功能，Pandas/Backtrader 均不支持

运行: python 2-Talib基础用法.py
"""
import numpy as np
import talib

print("=" * 60)
print("TA-Lib 基础用法")
print("=" * 60)

# ============================================================
# 1. 安装验证
# ============================================================
print(f"\nTA-Lib 版本: {talib.__version__}")
print(f"可用函数数量: {len(talib.get_functions())}")  # 通常为 158

# 按类别分组展示，了解 TA-Lib 的功能全貌
groups = talib.get_function_groups()
print("\n指标类别:")
for group, funcs in groups.items():
    print(f"  {group:30s} {len(funcs)} 个")

# ============================================================
# 2. SMA 简单移动平均
# ============================================================
print("\n" + "-" * 60)
print("[SMA] 简单移动平均")
print("-" * 60)

# TA-Lib 的所有函数都要求输入为 numpy float64 数组
close = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=np.float64)
# SMA(5) = 最近 5 个收盘价的算术平均值
# 输出数组长度与输入相同，前 period-1 个值为 NaN
sma5 = talib.SMA(close, timeperiod=5)
print(f"收盘价:   {close}")
print(f"SMA(5):   {sma5}")

# 手算验证：SMA(5) 在第 5 个位置的值 = (10+11+12+13+14)/5 = 12
hand_calc = np.mean(close[0:5])
print(f"\n手算验证: (10+11+12+13+14)/5 = {hand_calc}")
print(f"TA-Lib:   {sma5[4]}")
print(f"一致: {abs(sma5[4] - hand_calc) < 1e-10}")

# ============================================================
# 3. MACD 指数平滑异同移动平均
# ============================================================
print("\n" + "-" * 60)
print("[MACD] 指数平滑异同移动平均")
print("-" * 60)

# 使用模拟股价数据演示 MACD
# MACD 需要至少 34 个数据点（slowperiod 26 + signalperiod 9 - 1）才能出值
# 这里用 50 个点确保完整显示
prices = np.array([
    1500, 1510, 1520, 1515, 1505, 1495, 1480, 1490, 1510, 1530,
    1550, 1540, 1535, 1545, 1560, 1570, 1565, 1555, 1540, 1530,
    1520, 1525, 1535, 1550, 1560, 1575, 1590, 1585, 1580, 1570,
    1565, 1575, 1590, 1600, 1610, 1605, 1595, 1600, 1615, 1620,
    1625, 1630, 1620, 1610, 1620, 1635, 1640, 1630, 1645, 1650,
], dtype=np.float64)

# MACD 返回三个数组：
#   macd (DIF)    = EMA(12) - EMA(26)
#   signal (DEA)  = DIF 的 EMA(9)
#   hist (柱状图) = DIF - DEA
macd, signal, hist = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)

print(f"数据长度: {len(prices)}")
print(f"MACD(DIF):  最后值 = {macd[-1]:.4f}")
print(f"Signal(DEA): 最后值 = {signal[-1]:.4f}")
print(f"Histogram:   最后值 = {hist[-1]:.4f}")

# 原理说明
print("\nMACD原理:")
print("  DIF = EMA(12) - EMA(26)")
print("  DEA = DIF 的 EMA(9)")
print("  柱状图 = DIF - DEA")
print("  金叉: DIF 上穿 DEA -> 买入信号（短期趋势强于长期趋势）")
print("  死叉: DIF 下穿 DEA -> 卖出信号（短期趋势弱于长期趋势）")

# ============================================================
# 4. RSI 相对强弱指标
# ============================================================
print("\n" + "-" * 60)
print("[RSI] 相对强弱指标")
print("-" * 60)

rsi = talib.RSI(prices, timeperiod=14)
print(f"RSI(14): 最后值 = {rsi[-1]:.2f}")

# RSI 原理说明
print("\nRSI原理:")
print("  RS = 平均涨幅 / 平均跌幅")
print("  RSI = 100 - 100/(1+RS)")
print("  RSI > 70: 超买区, 考虑卖出（上涨过猛，可能回调）")
print("  RSI < 30: 超卖区, 考虑买入（下跌过猛，可能反弹）")
print("  RSI = 50: 多空平衡")

# ============================================================
# 5. ATR 真实波幅均值
# ============================================================
print("\n" + "-" * 60)
print("[ATR] 真实波幅均值")
print("-" * 60)

# 模拟 high/low 数据：2% 的日内波动
high = prices * 1.02
low = prices * 0.98

atr = talib.ATR(high, low, prices, timeperiod=14)
print(f"ATR(14): 最后值 = {atr[-1]:.2f}")

print("\nATR原理:")
print("  真实波幅 (TR) = max(High-Low, |High-PrevClose|, |Low-PrevClose|)")
print("  真实波幅取三者最大值，捕捉跳空缺口的波动")
print("  ATR = TR 的 N 日移动平均")
print("  用途: 设置止损位（止损 = 入场价 - 2倍ATR）")

# ============================================================
# 6. 常用指标分类速查
# ============================================================
print("\n" + "-" * 60)
print("常用指标速查")
print("-" * 60)

# 这里列出了 TA-Lib 中最常用的指标，按类别分组
# 完整列表见 talib.get_function_groups()
common = {
    '均线': ['SMA', 'EMA', 'WMA', 'DEMA', 'TEMA'],
    '动量': ['RSI', 'MACD', 'MOM', 'CCI', 'ADX', 'STOCH'],
    '波动率': ['ATR', 'NATR', 'TRANGE'],
    '成交量': ['OBV', 'AD', 'ADOSC'],
    '形态识别': ['CDLHAMMER', 'CDLENGULFING', 'CDLMORNINGSTAR', 'CDLDOJI'],
    '统计': ['BETA', 'CORREL', 'LINEARREG', 'STDDEV'],
}
for cat, funcs in common.items():
    print(f"  {cat}: {', '.join(funcs)}")

print("\n" + "=" * 60)
print("TA-Lib 共 158 种指标, 覆盖技术分析的所有领域")
print("=" * 60)
