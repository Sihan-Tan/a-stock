# -*- coding: utf-8 -*-
"""
K 线形态识别 -- TA-Lib 独有能力

TA-Lib 内置 61 种 K 线形态识别函数（CDL_* 系列），
这是 Backtrader 和 Pandas 都没有的功能。

什么是 K 线形态？
  K 线形态是根据单根或多根 K 线的开盘价、收盘价、最高价、最低价
  之间的几何关系来判断市场情绪的特定模式。例如：
  - 锤子线：下影线长，实体小，预示底部反转
  - 吞没形态：大阳线包住前一根阴线，预示趋势反转

本脚本功能：
  1. 列出 TA-Lib 所有 61 种 CDL 函数
  2. 扫描茅台（600519.SH）2024-2025 年数据，统计每种形态的出现次数
  3. 解读重点形态的含义和用法

运行: python 3-K线形态识别.py
"""
import numpy as np
import talib
from data_loader import load_stock_data

print("=" * 60)
print("K线形态识别 - 扫描茅台(600519.SH)")
print("=" * 60)

# 加载数据并转换为 numpy float64 数组（TA-Lib 要求）
df = load_stock_data('600519.SH', '2024-01-01', '2025-12-31')
o = df['open'].values.astype(np.float64)     # 开盘价
h = df['high'].values.astype(np.float64)     # 最高价
l = df['low'].values.astype(np.float64)      # 最低价
c = df['close'].values.astype(np.float64)    # 收盘价

print(f"数据范围: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
print(f"交易日数: {len(df)}")

# ============================================================
# 1. 列出所有 CDL 函数（共 61 种）
# ============================================================
cdl_funcs = [f for f in talib.get_functions() if f.startswith('CDL')]
print(f"\nTA-Lib K线形态函数: {len(cdl_funcs)} 种")
print("完整列表:")
for i, f in enumerate(cdl_funcs):
    print(f"  {i+1:2d}. {f}")

# ============================================================
# 2. 逐个扫描，统计各形态在数据范围内出现的次数
# ============================================================
print(f"\n{'-'*60}")
print("扫描结果:")
print(f"{'-'*60}")

# 常用形态的中文名称映射，方便阅读
# 只映射了最常用的约 30 种，未映射的使用原名
PATTERN_NAMES = {
    'CDLHAMMER': '锤子线', 'CDLINVERTEDHAMMER': '倒锤子线',
    'CDLENGULFING': '吞没形态', 'CDLHARAMI': '孕线',
    'CDLMORNINGSTAR': '早晨之星', 'CDLEVENINGSTAR': '黄昏之星',
    'CDLDOJI': '十字星', 'CDLDRAGONFLYDOJI': '蜻蜓十字',
    'CDLGRAVESTONEDOJI': '墓碑十字', 'CDLHANGINGMAN': '吊人线',
    'CDLSHOOTINGSTAR': '射击之星', 'CDLDARKCLOUDCOVER': '乌云盖顶',
    'CDLPIERCING': '曙光初现', 'CDL3WHITESOLDIERS': '三白兵',
    'CDL3BLACKCROWS': '三黑鸦', 'CDLSPINNINGTOP': '纺锤线',
    'CDLMARUBOZU': '光头光脚', 'CDLKICKING': '反冲形态',
    'CDLBELTHOLD': '捉腰带线', 'CDLCLOSINGMARUBOZU': '收盘光头',
    'CDL3INSIDE': '三内部', 'CDL3OUTSIDE': '三外部',
    'CDLABANDONEDBABY': '弃婴', 'CDLADVANCEBLOCK': '前进受阻',
    'CDLCOUNTERATTACK': '反击线', 'CDLGAPSIDESIDEWHITE': '并列阳线',
    'CDLHIGHWAVE': '长脚十字', 'CDLLONGLINE': '长实体',
    'CDLSHORTLINE': '短实体', 'CDLSTALLEDPATTERN': '停顿形态',
}

bullish_signals = []  # 看涨形态列表
bearish_signals = []  # 看跌形态列表

for func_name in cdl_funcs:
    func = getattr(talib, func_name)   # 通过函数名获取函数对象
    # 所有 CDL 函数的参数签名一致：CDL*(open, high, low, close)
    # 返回值：>0 看涨，<0 看跌，=0 无信号
    # 绝对值大小表示信号强度（100=标准，200=强信号）
    result = func(o, h, l, c)

    bullish_count = np.sum(result > 0)   # 看涨信号出现次数
    bearish_count = np.sum(result < 0)   # 看跌信号出现次数

    cn_name = PATTERN_NAMES.get(func_name, func_name)  # 中文名，没有则用原名

    if bullish_count > 0:
        # 找到最近一次看涨信号的日期
        last_idx = np.where(result > 0)[0][-1]
        last_date = df.index[last_idx].strftime('%Y-%m-%d')
        bullish_signals.append((cn_name, func_name, bullish_count, last_date))

    if bearish_count > 0:
        # 找到最近一次看跌信号的日期
        last_idx = np.where(result < 0)[0][-1]
        last_date = df.index[last_idx].strftime('%Y-%m-%d')
        bearish_signals.append((cn_name, func_name, bearish_count, last_date))

# 按出现次数降序排列，显示看涨形态
print(f"\n  看涨形态 (共 {len(bullish_signals)} 种出现过):")
for cn, en, count, date in sorted(bullish_signals, key=lambda x: -x[2]):
    print(f"    {cn:12s} ({en:25s}) 出现 {count:3d} 次, 最近: {date}")

# 按出现次数降序排列，显示看跌形态
print(f"\n  看跌形态 (共 {len(bearish_signals)} 种出现过):")
for cn, en, count, date in sorted(bearish_signals, key=lambda x: -x[2]):
    print(f"    {cn:12s} ({en:25s}) 出现 {count:3d} 次, 最近: {date}")

# ============================================================
# 3. 重点形态解读 -- 理解每种形态背后的市场含义
# ============================================================
print(f"\n{'='*60}")
print("重点形态解读")
print("=" * 60)

key_patterns = [
    # 锤子线：下跌趋势末端出现，下影线很长（>=实体2倍），上影线很短或没有
    # 市场含义：空方将价格打压到低位，但多方强力拉回，说明下跌动能耗尽
    ('锤子线(Hammer)', '下跌趋势末端, 下影线长(>=实体2倍), 上影线短或无\n'
     '    含义: 空方力量衰竭, 可能反转向上'),
    # 吞没形态：当前K线实体完全包住前一根K线的实体
    # 看涨吞没：阴线之后出现大阳线，包住前一根阴线
    # 看跌吞没：阳线之后出现大阴线，包住前一根阳线
    ('吞没形态(Engulfing)', '当前K线实体完全包住前一根\n'
     '    看涨吞没: 阳包阴, 底部反转信号\n'
     '    看跌吞没: 阴包阳, 顶部反转信号'),
    # 早晨之星：三根K线组合——大阴线+小实体(十字星)+大阳线
    # 市场含义：下跌→犹豫→上涨，趋势由空转多
    ('早晨之星(Morning Star)', '三根K线组合: 大阴线 + 十字星 + 大阳线\n'
     '    含义: 底部反转的强信号'),
    # 十字星：开盘价=收盘价（或极为接近），表明多空力量暂时平衡
    # 在趋势末端出现十字星，往往预示趋势即将改变
    ('十字星(Doji)', '开盘价=收盘价(或极接近), 说明多空平衡\n'
     '    含义: 趋势可能即将改变'),
    # 射击之星：上升趋势末端出现，上影线很长，实体很小
    # 市场含义：多方将价格推高但无法维持，空方开始反击
    ('射击之星(Shooting Star)', '上升趋势末端, 上影线长, 实体小\n'
     '    含义: 多方力量衰竭, 可能反转向下'),
]

for name, desc in key_patterns:
    print(f"\n  {name}")
    print(f"    {desc}")

print(f"\n{'='*60}")
print("K线形态是TA-Lib的独有优势, 可用于:")
print("  1. 策略入场点精确化 (趋势+形态共振)")
print("  2. 选股扫描 (批量扫描出现特定形态的股票)")
print("  3. 特征工程 (作为机器学习模型的输入特征)")
print("=" * 60)
