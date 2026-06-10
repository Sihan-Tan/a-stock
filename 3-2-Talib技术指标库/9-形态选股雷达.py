# -*- coding: utf-8 -*-
"""
形态选股雷达 -- 全市场截面扫描：MACD 底背离 + K 线反转形态

本脚本是本课程最"实战"的应用，模拟真实量化投研中的"选股"环节。

使用场景：
  每天收盘后对全市场股票运行一次形态扫描，筛选出次日值得关注的股票。

扫描维度（双重共振）：
  1. MACD 底背离（动量维度）
     - 价格创新低但 MACD 未创新低 -> 空方力量衰竭
     - 说明下跌动能减弱，趋势可能反转
  2. K 线看涨反转形态（技术形态维度）
     - 吞没形态、锤子线、早晨之星等
     - 说明在该价格位置有较强的买入力量
  3. 双重共振 = 底背离 + K 线看涨形态
     - 同时在动量和形态两个维度发出买入信号
     - 信号可靠性远高于单一维度

TA-Lib 在本脚本的作用：
  - talib.MACD(close) -> 计算 MACD 指标，用于底背离检测
  - talib.CDLENGULFING(o, h, l, c) -> 吞没形态检测
  - talib.CDLHAMMER(o, h, l, c) -> 锤子线检测
  - ... 共 10 种看涨形态检测
  - talib.SMA(volume, 20) -> 计算成交量均线，用于量比分析

输出：
  - 按信号等级排序的候选池：A-双重共振 > B-底背离放量 > C-形态放量
  - 导出为 CSV 文件，方便次日开盘前查看

运行: python 9-形态选股雷达.py
"""
import numpy as np
import pandas as pd
import talib
import time
import os
import json
from openai import OpenAI
from db_config import execute_query, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from data_loader import get_instrument_names


# ============================================================
# 批量数据加载
# ============================================================

def batch_load_recent(days_back=120, end_date=None):
    """
    批量加载全市场近 N 个交易日的 K 线数据

    为什么需要批量加载？
      如果逐只使用 load_stock_data() 查询 5000 只股票，
      需要 5000 次 SQL 查询，耗时数十分钟。
      一条 SQL 一次性加载所有股票的全部数据，只需 1 次查询，
      耗时数秒，速度快几十倍。

    参数:
        days_back: 每只股票保留的最大交易日数
        end_date: 截止日期，None 则取数据库最新日期

    返回:
        dict {stock_code: DataFrame}，DataFrame 列为 open/high/low/close/volume
        只保留数据量 >= 60 条的股票（MACD 需要足够数据）
    """
    if end_date:
        ref_date = pd.Timestamp(end_date)
    else:
        # 取数据库中所有股票的最新交易日作为截止日期
        rows = execute_query("SELECT MAX(trade_date) AS latest FROM trade_stock_daily")
        if not rows or not rows[0]['latest']:
            return {}
        ref_date = pd.Timestamp(rows[0]['latest'])

    # 由于 A 股市场有周末和节假日，自然日需要预留更多余量
    # 250 个交易日约等于 365 个自然日，所以乘 1.8 的系数
    calendar_days = int(days_back * 1.8)
    start_date = (ref_date - pd.Timedelta(days=calendar_days)).strftime('%Y-%m-%d')
    end_str = ref_date.strftime('%Y-%m-%d')

    print(f"  SQL查询范围: {start_date} ~ {end_str}")

    # 一次性查询全市场数据，不按股票过滤，通过 ORDER BY 分组
    sql = """
        SELECT stock_code, trade_date, open_price, high_price, low_price, close_price, volume
        FROM trade_stock_daily
        WHERE trade_date >= %s AND trade_date <= %s
        ORDER BY stock_code, trade_date ASC
    """
    rows = execute_query(sql, [start_date, end_str])
    if not rows:
        return {}

    # 统一转换为 DataFrame
    df_all = pd.DataFrame(rows)
    df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])
    for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    # 按股票代码分组，每只保留最近 days_back 条数据
    # MACD(26) + signal(9) + 背离回看窗口(60) = 至少需要 60 条数据
    min_bars = 60
    result = {}
    for code, group in df_all.groupby('stock_code'):
        sub = group.set_index('trade_date').sort_index()
        sub = sub[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]
        sub.columns = ['open', 'high', 'low', 'close', 'volume']
        if len(sub) >= min_bars:
            result[code] = sub.tail(days_back)

    return result


# ============================================================
# MACD 底背离检测
# ============================================================

def detect_bottom_divergence(close, macd_line, lookback=60, recent_window=10):
    """
    检测 MACD 底背离（Bottom Divergence）

    底背离是技术分析中最可靠的底部反转信号之一。

    定义：
      价格创新低（或持平），但 MACD 线未创新低
      -> 说明下跌动能减弱，空方力量衰竭
      -> 后续大概率反转上涨

    算法实现：
      将回看区间分为"前段"（较早期）和"近段"（近期）：
        前段: index [n-lookback, n-recent_window)
        近段: index [n-recent_window, n)
      分别找到两段中的最低价位置，比较对应位置的 MACD 值。
      若近段价格更低但 MACD 更高 -> 底背离成立。

    参数:
        close: 收盘价数组（numpy array）
        macd_line: MACD 线（DIF）数组
        lookback: 总回看 K 线数（默认 60，约 3 个月）
        recent_window: 近段窗口（默认 10，近 2 周）

    返回:
        (bool, dict) -> (是否存在底背离, 详细信息字典)
        详细信息包括：前后段价格、前后段 MACD、时间间隔
    """
    n = len(close)
    if n < lookback:
        return False, {}

    # 近段：最后 recent_window 根 K 线（近期价格走势）
    recent_slice = close[n - recent_window: n]
    # 前段：lookback ~ recent_window 之前的 K 线（较早期的价格走势）
    prev_slice = close[n - lookback: n - recent_window]

    if len(prev_slice) == 0 or len(recent_slice) == 0:
        return False, {}

    # 分别找到各段价格最低点的位置（局部索引）
    recent_low_local = int(np.argmin(recent_slice))
    prev_low_local = int(np.argmin(prev_slice))

    # 将局部索引映射为全局索引
    idx_recent = n - recent_window + recent_low_local
    idx_prev = n - lookback + prev_low_local

    if np.isnan(macd_line[idx_recent]) or np.isnan(macd_line[idx_prev]):
        return False, {}

    # 核心条件：价格更低（或持平），但 MACD 更高
    price_lower = close[idx_recent] <= close[idx_prev]    # 近段价格 <= 前段价格
    macd_higher = macd_line[idx_recent] > macd_line[idx_prev]  # 近段 MACD > 前段 MACD

    if price_lower and macd_higher:
        return True, {
            'prev_price': round(float(close[idx_prev]), 2),          # 前段低点的价格
            'recent_price': round(float(close[idx_recent]), 2),      # 近段低点的价格
            'prev_macd': round(float(macd_line[idx_prev]), 4),      # 前段低点的 MACD
            'recent_macd': round(float(macd_line[idx_recent]), 4),  # 近段低点的 MACD
            'days_apart': idx_recent - idx_prev,                    # 两次低点的时间间隔
        }

    return False, {}


# ============================================================
# K 线看涨形态扫描
# ============================================================

# 选取了 10 种最常用、信号最可靠的看涨反转形态
# 这些形态在不同市场环境下都有较好的表现
BULLISH_PATTERNS = {
    'CDLENGULFING':      '看涨吞没',      # 阳包阴：大阳线完全包住前一根阴线
    'CDLHAMMER':         '锤子线',         # 下影线长，实体小，底部反转
    'CDLMORNINGSTAR':    '早晨之星',       # 阴线+十字星+阳线，强反转信号
    'CDLPIERCING':       '曙光初现',       # 阴线后低开高走，穿透前阴线实体一半
    'CDL3WHITESOLDIERS': '三白兵',         # 连续三根大阳线，上升趋势确立
    'CDLINVERTEDHAMMER': '倒锤子线',       # 上影线长，实体小，底部试盘
    'CDL3INSIDE':        '三内部上涨',     # 三根K线组合的内部上涨形态
    'CDL3OUTSIDE':       '三外部上涨',     # 三根K线组合的外部上涨形态
    'CDLHARAMI':         '看涨孕线',       # 小阳线在前大阴线内部，空方衰竭
    'CDLDRAGONFLYDOJI':  '蜻蜓十字',       # 下影线极长的十字星，强反转
}


def scan_bullish_patterns(o, h, l, c):
    """
    扫描最后一根 K 线是否出现看涨反转形态

    TA-Lib 的 CDL 函数返回值约定：
      > 0   = 看涨信号（100=标准强度，200=强信号）
      < 0   = 看跌信号
      = 0   = 无信号

    参数:
        o/h/l/c: 开盘价/最高价/最低价/收盘价数组

    返回:
        list of (中文名, 英文函数名, 信号强度)
    """
    found = []
    for func_name, cn_name in BULLISH_PATTERNS.items():
        func = getattr(talib, func_name)   # 通过名称获取函数对象
        result = func(o, h, l, c)           # 所有 CDL 函数参数签名一致
        last_val = result[-1]               # 最后一根 K 线的信号值
        if last_val > 0:                    # 出现看涨信号
            found.append((cn_name, func_name, int(last_val)))
    return found


# ============================================================
# 单只股票完整扫描
# ============================================================

def scan_one(df, lookback=60, recent_window=10):
    """
    对单只股票运行完整的形态扫描（MACD 底背离 + K 线形态）

    流程：
      1. 提取 numpy 数组（open/high/low/close/volume）
      2. 用 TA-Lib 计算 MACD
      3. 检测 MACD 底背离
      4. 扫描 K 线看涨形态
      5. 计算量比 = 当日成交量 / 20日均量

    参数:
        df: 单只股票的 DataFrame
        lookback: 底背离回看周期
        recent_window: 近段窗口

    返回:
        dict 包含扫描结果
    """
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)

    # 计算 MACD
    macd, signal, hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)

    # 检测底背离
    has_div, div_info = detect_bottom_divergence(c, macd, lookback, recent_window)

    # 扫描 K 线看涨形态
    patterns = scan_bullish_patterns(o, h, l, c)

    # 计算量比（成交量比率）= 当日成交量 / 20 日均量
    # 量比 > 1 表示放量，> 1.5 表示明显放量
    # 量比是确认信号有效性的重要辅助指标
    vol_ma = talib.SMA(v, timeperiod=20)
    if not np.isnan(vol_ma[-1]) and vol_ma[-1] > 0:
        vol_ratio = float(v[-1] / vol_ma[-1])
    else:
        vol_ratio = 0.0

    return {
        'has_divergence': has_div,
        'divergence_info': div_info,
        'bullish_patterns': patterns,
        'close': round(float(c[-1]), 2),
        'change_pct': round(float((c[-1] / c[-2] - 1) * 100), 2) if len(c) >= 2 else 0,
        'macd': round(float(macd[-1]), 4) if not np.isnan(macd[-1]) else 0,
        'vol_ratio': round(vol_ratio, 2),
        'combined': has_div and len(patterns) > 0,  # 双重共振标记
    }


# ============================================================
# 全市场扫描主函数
# ============================================================

def run_radar(end_date=None, lookback=60, recent_window=10):
    """
    形态选股雷达 -- 全市场截面扫描主函数

    流程：
      第 1 步：一次性加载全市场 K 线数据（批量 SQL，只查一次数据库）
      第 2 步：逐只运行 TA-Lib MACD + CDL 形态检测
      第 3 步：按信号强度排序输出：双重共振 > 单一信号

    参数:
        end_date: 扫描日期，None=数据库最新日期
        lookback: MACD 底背离回看周期（默认 60 个交易日）
        recent_window: 近段窗口（低点需在此范围内，默认 10）

    返回:
        dict 包含扫描结果和候选池
    """
    print("=" * 70)
    print("形态选股雷达 - 截面扫描")
    print("  MACD底背离 + K线看涨反转形态 -> 次日关注池")
    print("=" * 70)

    # ---- 第 1 步：批量加载数据 ----
    print("\n[1/3] 批量加载K线数据...")
    t0 = time.time()
    all_data = batch_load_recent(days_back=max(lookback + 60, 120), end_date=end_date)
    load_time = time.time() - t0
    print(f"  加载完成: {len(all_data)} 只标的, 耗时 {load_time:.1f}s")

    if not all_data:
        print("  没有可用数据, 请先运行 1-行情数据采集.py")
        return

    sample_code = next(iter(all_data))
    scan_date = all_data[sample_code].index[-1].strftime('%Y-%m-%d')
    print(f"  扫描日期: {scan_date}")

    # ---- 第 2 步：逐只扫描 ----
    print(f"\n[2/3] 运行TA-Lib形态扫描 (MACD底背离 + {len(BULLISH_PATTERNS)}种看涨K线形态)...")
    t0 = time.time()

    divergence_list = []   # 出现底背离的股票
    pattern_list = []      # 出现看涨形态的股票
    combined_list = []     # 双重共振的股票（底背离+形态）
    errors = 0

    for code, df in all_data.items():
        try:
            result = scan_one(df, lookback, recent_window)
            result['code'] = code

            if result['has_divergence']:
                divergence_list.append(result)
            if result['bullish_patterns']:
                pattern_list.append(result)
            if result['combined']:
                combined_list.append(result)
        except Exception:
            errors += 1

    scan_time = time.time() - t0
    print(f"  扫描完成: 耗时 {scan_time:.1f}s" +
          (f", 跳过异常 {errors} 只" if errors else ""))

    # ---- 查询股票名称 ----
    hit_codes = list(set(
        [r['code'] for r in divergence_list] +
        [r['code'] for r in pattern_list]
    ))
    names = get_instrument_names(hit_codes) if hit_codes else {}

    # ---- 第 3 步：输出结果 ----
    print(f"\n[3/3] 扫描结果 ({scan_date})")
    print("=" * 70)

    # (A) 双重共振 -- 最高优先级，两种信号互相验证
    print(f"\n{'*'*70}")
    print(f"  双重共振: MACD底背离 + 看涨K线形态  ({len(combined_list)} 只)")
    print(f"{'*'*70}")
    if combined_list:
        # 按量比降序排列（放量越大越值得关注）
        combined_list.sort(key=lambda x: x['vol_ratio'], reverse=True)
        print(f"{'代码':<14} {'名称':<10} {'收盘':>8} {'涨跌%':>7} {'量比':>6} {'形态':<20} {'背离信息'}")
        print("-" * 90)
        for r in combined_list:
            name = names.get(r['code'], r['code'])
            pat_str = ','.join(p[0] for p in r['bullish_patterns'])
            div = r['divergence_info']
            div_str = (f"价{div['prev_price']}->{div['recent_price']} "
                       f"MACD{div['prev_macd']}->{div['recent_macd']}")
            print(f"{r['code']:<14} {name:<10} {r['close']:>8.2f} "
                  f"{r['change_pct']:>+6.2f}% {r['vol_ratio']:>5.1f}x "
                  f"{pat_str:<20} {div_str}")
    else:
        print("  (无)")

    # (B) 仅 MACD 底背离（无 K 线形态验证）
    div_only = [r for r in divergence_list if not r['combined']]
    print(f"\n--- 仅MACD底背离 ({len(div_only)} 只, 显示前30) ---")
    if div_only:
        div_only.sort(key=lambda x: x['vol_ratio'], reverse=True)
        print(f"{'代码':<14} {'名称':<10} {'收盘':>8} {'涨跌%':>7} {'量比':>6} {'MACD':>10} {'间距'}")
        print("-" * 70)
        for r in div_only[:30]:
            name = names.get(r['code'], r['code'])
            div = r['divergence_info']
            print(f"{r['code']:<14} {name:<10} {r['close']:>8.2f} "
                  f"{r['change_pct']:>+6.2f}% {r['vol_ratio']:>5.1f}x "
                  f"{r['macd']:>10.4f} {div['days_apart']:>4}日")
        if len(div_only) > 30:
            print(f"  ... 还有 {len(div_only) - 30} 只")
    else:
        print("  (无)")

    # (C) 仅 K 线看涨形态（无底背离验证）
    pat_only = [r for r in pattern_list if not r['combined']]
    print(f"\n--- 仅看涨K线形态 ({len(pat_only)} 只, 显示前30) ---")
    if pat_only:
        pat_only.sort(key=lambda x: x['vol_ratio'], reverse=True)
        print(f"{'代码':<14} {'名称':<10} {'收盘':>8} {'涨跌%':>7} {'量比':>6} {'形态'}")
        print("-" * 60)
        for r in pat_only[:30]:
            name = names.get(r['code'], r['code'])
            pat_str = ','.join(p[0] for p in r['bullish_patterns'])
            print(f"{r['code']:<14} {name:<10} {r['close']:>8.2f} "
                  f"{r['change_pct']:>+6.2f}% {r['vol_ratio']:>5.1f}x "
                  f"{pat_str}")
        if len(pat_only) > 30:
            print(f"  ... 还有 {len(pat_only) - 30} 只")
    else:
        print("  (无)")

    # ---- 汇总统计 ----
    print(f"\n{'='*70}")
    print(f"扫描汇总 ({scan_date})")
    print(f"{'='*70}")
    print(f"  扫描标的:       {len(all_data)} 只")
    print(f"  MACD底背离:     {len(divergence_list)} 只 "
          f"({len(divergence_list)/len(all_data)*100:.1f}%)")
    print(f"  看涨K线形态:    {len(pattern_list)} 只 "
          f"({len(pattern_list)/len(all_data)*100:.1f}%)")
    print(f"  双重共振(重点): {len(combined_list)} 只 "
          f"({len(combined_list)/len(all_data)*100:.1f}%)")
    print(f"  总耗时:         {load_time + scan_time:.1f}s")

    # ---- 形态命中分布统计 ----
    # 了解哪些形态在当前市场环境下出现频率最高
    if pattern_list:
        pat_counter = {}
        for r in pattern_list:
            for cn, en, _ in r['bullish_patterns']:
                pat_counter[cn] = pat_counter.get(cn, 0) + 1
        print(f"\n  形态命中分布:")
        for cn, cnt in sorted(pat_counter.items(), key=lambda x: -x[1]):
            print(f"    {cn:<12} {cnt:>4} 只")

    return {
        'scan_date': scan_date,
        'total': len(all_data),
        'divergence': divergence_list,
        'patterns': pattern_list,
        'combined': combined_list,
        'names': names,
    }


# ============================================================
# 候选池导出
# ============================================================

def _is_tradable(r, name):
    """
    判断标的是否适合次日交易

    过滤掉不适合次日操作的情况：
      1. ST/*ST 股票：涨跌幅限制 5%，风险极高
      2. 当日涨停/跌停：次日可能延续极端走势，难以入场/出场
      3. 量比过低（<0.3）：流动性不足，买卖价差大

    参数:
        r: 扫描结果字典
        name: 股票名称

    返回:
        bool True=可交易，False=应过滤
    """
    if 'ST' in name or 'st' in name:
        return False        # 过滤 ST 股票
    if abs(r['change_pct']) >= 9.9:
        return False        # 过滤涨跌停（A 股涨跌幅限制 10%，ST 5%）
    if r['vol_ratio'] < 0.3:
        return False        # 过滤流动性不足
    return True


def _is_stock(code):
    """判断是否为个股（排除 ETF/LOF/REIT 等基金产品）"""
    if code.startswith(('51', '56', '58', '15', '16')):
        return False        # 以这些开头的代码通常是基金/ETF
    return True


def export_candidate_pool(scan_result):
    """
    将扫描结果导出为次日候选池 CSV 文件

    信号等级划分：
      A 级（双重共振）：底背离 + K 线看涨形态同时出现，信号最可靠
      B 级（底背离放量）：仅底背离，但成交量放大（量比 >= 1.0）
      C 级（形态放量）：仅 K 线看涨形态，但明显放量（量比 >= 1.5）

    筛选过滤：
      - 排除 ST 股票
      - 排除当日涨跌停
      - 排除量比过低的标的
      - 个股和 ETF 分开统计

    参数:
        scan_result: run_radar() 的返回结果

    输出:
        outputs/候选池_YYYY-MM-DD.csv
    """
    if not scan_result:
        return

    scan_date = scan_result['scan_date']
    names = scan_result.get('names', {})
    combined = scan_result['combined']
    divergence = scan_result['divergence']
    patterns = scan_result['patterns']

    os.makedirs('outputs', exist_ok=True)

    # 构建候选池记录
    rows = []

    # ---- A 级信号：双重共振 ----
    for r in combined:
        name = names.get(r['code'], r['code'])
        if not _is_tradable(r, name):
            continue
        div = r['divergence_info']
        rows.append({
            '信号等级': 'A-双重共振',
            '代码': r['code'],
            '名称': name,
            '类型': '个股' if _is_stock(r['code']) else 'ETF/基金',
            '收盘价': r['close'],
            '涨跌幅%': r['change_pct'],
            '量比': r['vol_ratio'],
            'K线形态': ','.join(p[0] for p in r['bullish_patterns']),
            'MACD': r['macd'],
            '背离前价': div['prev_price'],
            '背离近价': div['recent_price'],
            '背离前MACD': div['prev_macd'],
            '背离近MACD': div['recent_macd'],
            '背离间距日': div['days_apart'],
        })

    # ---- B 级信号：仅底背离 + 量比 >= 1.0 ----
    for r in divergence:
        if r['combined']:
            continue  # 已归入 A 级
        name = names.get(r['code'], r['code'])
        if not _is_tradable(r, name):
            continue
        if r['vol_ratio'] < 1.0:
            continue  # 需要放量确认
        div = r['divergence_info']
        rows.append({
            '信号等级': 'B-底背离放量',
            '代码': r['code'],
            '名称': name,
            '类型': '个股' if _is_stock(r['code']) else 'ETF/基金',
            '收盘价': r['close'],
            '涨跌幅%': r['change_pct'],
            '量比': r['vol_ratio'],
            'K线形态': '',
            'MACD': r['macd'],
            '背离前价': div['prev_price'],
            '背离近价': div['recent_price'],
            '背离前MACD': div['prev_macd'],
            '背离近MACD': div['recent_macd'],
            '背离间距日': div['days_apart'],
        })

    # ---- C 级信号：仅 K 线形态 + 量比 >= 1.5 ----
    for r in patterns:
        if r['combined']:
            continue  # 已归入 A 级
        name = names.get(r['code'], r['code'])
        if not _is_tradable(r, name):
            continue
        if r['vol_ratio'] < 1.5:
            continue  # 需要明显放量
        rows.append({
            '信号等级': 'C-形态放量',
            '代码': r['code'],
            '名称': name,
            '类型': '个股' if _is_stock(r['code']) else 'ETF/基金',
            '收盘价': r['close'],
            '涨跌幅%': r['change_pct'],
            '量比': r['vol_ratio'],
            'K线形态': ','.join(p[0] for p in r['bullish_patterns']),
            'MACD': r['macd'],
            '背离前价': '',
            '背离近价': '',
            '背离前MACD': '',
            '背离近MACD': '',
            '背离间距日': '',
        })

    if not rows:
        print("\n  过滤后无候选标的")
        return

    # 排序：按信号等级（A > B > C），同等级内按量比降序
    df = pd.DataFrame(rows)
    df = df.sort_values(['信号等级', '量比'], ascending=[True, False])

    # 导出 CSV（utf-8-sig 编码确保 Excel 打开不乱码）
    csv_path = os.path.join('outputs', f'候选池_{scan_date}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 打印分类摘要
    stocks_df = df[df['类型'] == '个股']
    etf_df = df[df['类型'] == 'ETF/基金']

    print(f"\n{'='*70}")
    print(f"次日候选池 ({scan_date} 收盘扫描)")
    print(f"{'='*70}")
    print(f"  已保存: {csv_path}")
    print(f"  总计: {len(df)} 只 (个股 {len(stocks_df)}, ETF/基金 {len(etf_df)})")

    for level in ['A-双重共振', 'B-底背离放量', 'C-形态放量']:
        sub = df[df['信号等级'] == level]
        if len(sub) == 0:
            continue
        print(f"\n  [{level}] {len(sub)} 只:")
        for _, row in sub.iterrows():
            tag = f"  {row['K线形态']}" if row['K线形态'] else ''
            print(f"    {row['代码']:<14} {row['名称']:<10} "
                  f"收盘{row['收盘价']:>8.2f}  {row['涨跌幅%']:>+6.2f}%  "
                  f"量比{row['量比']:>4.1f}x{tag}")

    return df


# ============================================================
# DeepSeek AI 二次筛选
# ============================================================

def _build_stock_lines(candidates, names):
    """将候选列表格式化为 DeepSeek prompt 用的文本行"""
    lines = []
    for i, (r, level) in enumerate(candidates, 1):
        code = r['code']
        name = names.get(code, code) if isinstance(names, dict) else names
        pat_str = ','.join(p[0] for p in r.get('bullish_patterns', []))
        div = r.get('divergence_info', {})
        div_str = ''
        if div:
            div_str = (f" | 背离: 前低价{div.get('prev_price','?')} MACD{div.get('prev_macd','?')} "
                       f"→ 近低价{div.get('recent_price','?')} MACD{div.get('recent_macd','?')} "
                       f"间隔{div.get('days_apart','?')}日")
        line = (f"{i}. [{level}] {code} {name} | "
                f"收盘{r['close']} | 涨跌{r['change_pct']:+.2f}% | "
                f"量比{r['vol_ratio']:.1f}x | MACD{r['macd']:.4f}"
                f"{' | 形态:' + pat_str if pat_str else ''}"
                f"{div_str}")
        lines.append(line)
    return '\n'.join(lines)


def _call_deepseek_api(stock_text, top_n):
    """调用 DeepSeek API 进行精选，返回解析后的 picks 列表"""
    prompt = f"""你是一位资深量化投资分析师。以下是一个A股形态选股雷达扫描后的候选股票池，每只股票已经通过了技术指标筛选（MACD底背离 / K线看涨反转形态 / 双重共振）。

请从中精选出最有投资价值的 {top_n} 只股票，综合考虑以下因素：

1. **信号强度**：A-双重共振 > B-仅底背离 > C-仅K线形态
2. **量价配合**：量比越大说明资金关注度越高，信号越可靠
3. **形态质量**：特定K线形态的历史胜率（如早晨之星、三白兵更可靠）
4. **风险收益比**：涨跌幅过大的可能已透支，适当回调的更安全
5. **背离质量**：背离间距越大、MACD改善越明显，反转信号越强

候选股票池：
{stock_text}

请严格按以下 JSON 格式返回，不要输出其他内容：
```json
[
  {{"code": "股票代码", "name": "股票名称", "confidence": 85, "reason": "理由(50字内)"}},
  ...
]
```

confidence 取值 0-100，按置信度从高到低排列，只返回恰好 {top_n} 条记录。"""

    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )
        completion = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        raw = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"  DeepSeek API 调用失败: {e}")
        return []

    # 解析 JSON（支持 markdown 代码块包裹）
    try:
        if '```' in raw:
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        picks = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  DeepSeek 返回格式异常，原始输出:\n{raw[:500]}")
        return []

    if not isinstance(picks, list):
        print(f"  DeepSeek 返回非列表格式: {type(picks)}")
        return []

    return picks


def _enrich_and_export(picks, candidates, scan_date):
    """补充完整技术指标、打印结果、导出 CSV"""
    candidate_map = {}
    for r, level in candidates:
        candidate_map[r['code']] = (r, level)

    print(f"\n  DeepSeek 最看好的 {len(picks)} 只:")
    print(f"  {'代码':<14} {'名称':<10} {'置信度':>6} {'理由'}")
    print(f"  {'-'*60}")
    enriched = []
    for p in picks:
        code = p.get('code', '?')
        name = p.get('name', '?')
        conf = p.get('confidence', 0)
        reason = p.get('reason', '')
        bar = '█' * (conf // 10)
        print(f"  {code:<14} {name:<10} {conf:>3}% {bar:<10} {reason}")

        row = {'代码': code, '名称': name, '置信度': conf, '理由': reason}
        if code in candidate_map:
            r, level = candidate_map[code]
            div = r.get('divergence_info', {})
            row.update({
                '信号等级': level,
                '收盘价': r['close'],
                '涨跌幅%': r['change_pct'],
                '量比': r['vol_ratio'],
                'MACD': r['macd'],
                'K线形态': ','.join(p[0] for p in r.get('bullish_patterns', [])),
                '背离前价': div.get('prev_price', '') if div else '',
                '背离近价': div.get('recent_price', '') if div else '',
                '背离前MACD': div.get('prev_macd', '') if div else '',
                '背离近MACD': div.get('recent_macd', '') if div else '',
                '背离间距日': div.get('days_apart', '') if div else '',
            })
        enriched.append(row)

    os.makedirs('outputs', exist_ok=True)
    csv_path = os.path.join('outputs', f'DeepSeek精选_{scan_date}.csv')
    df_picks = pd.DataFrame(enriched)
    cols_order = ['代码', '名称', '置信度', '理由', '信号等级', '收盘价', '涨跌幅%',
                  '量比', 'MACD', 'K线形态',
                  '背离前价', '背离近价', '背离前MACD', '背离近MACD', '背离间距日']
    df_picks = df_picks[[c for c in cols_order if c in df_picks.columns]]
    df_picks.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  已保存: {csv_path}")


def _run_deepseek_pipeline(candidates, names, scan_date, top_n):
    """核心流水线：格式化 -> 调 API -> 解析 -> 导出"""
    stock_text = _build_stock_lines(candidates, names)
    picks = _call_deepseek_api(stock_text, top_n)
    if picks:
        _enrich_and_export(picks, candidates, scan_date)
    return picks


def deepseek_filter(scan_result, top_n=10):
    """
    使用 DeepSeek 对扫描结果进行 AI 二次筛选，选出最看好的 top_n 只并给出置信度。

    参数:
        scan_result: run_radar() 的返回结果
        top_n: 最终保留的推荐数量（默认 10）

    返回:
        list[dict]: 推荐的股票列表，每项包含 code/name/confidence/reason
    """
    if not scan_result:
        return []

    # 从扫描结果收集所有候选（去重，A > B > C 优先级）
    seen = set()
    candidates = []
    for r in scan_result.get('combined', []):
        if r['code'] not in seen:
            candidates.append((r, 'A-双重共振(底背离+K线形态)'))
            seen.add(r['code'])
    for r in scan_result.get('divergence', []):
        if r['code'] not in seen:
            candidates.append((r, 'B-仅底背离'))
            seen.add(r['code'])
    for r in scan_result.get('patterns', []):
        if r['code'] not in seen:
            candidates.append((r, 'C-仅K线形态'))
            seen.add(r['code'])

    if not candidates:
        print("\n  候选池为空，跳过 AI 筛选")
        return []

    print(f"\n{'='*70}")
    print(f"DeepSeek AI 二次筛选")
    print(f"{'='*70}")
    print(f"  候选池: {len(candidates)} 只 -> 正在请求 DeepSeek 精选 top {top_n}...")

    if not DEEPSEEK_API_KEY:
        print("  未配置 DEEPSEEK_API_KEY，跳过 AI 筛选")
        return []

    names = scan_result.get('names', {})
    scan_date = scan_result['scan_date']
    return _run_deepseek_pipeline(candidates, names, scan_date, top_n)


def deepseek_filter_from_csv(csv_path, top_n=10):
    """
    从已导出的候选池 CSV 直接加载并运行 DeepSeek 精选，无需重新扫描。

    使用方式:
        python 9-形态选股雷达.py --csv outputs/候选池_2025-01-15.csv

    参数:
        csv_path: 候选池 CSV 文件路径
        top_n: 最终保留的推荐数量（默认 10）

    返回:
        list[dict]: 推荐的股票列表
    """
    if not os.path.exists(csv_path):
        print(f"  文件不存在: {csv_path}")
        return []

    print(f"\n{'='*70}")
    print(f"DeepSeek AI 二次筛选 (从CSV加载)")
    print(f"{'='*70}")
    print(f"  加载: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        print("  CSV 为空")
        return []

    # 从文件名推断扫描日期
    basename = os.path.splitext(os.path.basename(csv_path))[0]
    scan_date = basename.replace('候选池_', '')

    # 将 CSV 行转换为与 scan_result 兼容的候选格式
    candidates = []
    for _, row in df.iterrows():
        code = str(row.get('代码', ''))
        level = str(row.get('信号等级', ''))

        # 构建 bullish_patterns（与原始格式兼容的元组列表）
        patterns_str = str(row.get('K线形态', '')) if pd.notna(row.get('K线形态')) else ''
        bullish_patterns = []
        if patterns_str:
            for pn in patterns_str.split(','):
                pn = pn.strip()
                if pn:
                    bullish_patterns.append((pn, '', 100))

        # 构建 divergence_info
        div_info = {}
        if pd.notna(row.get('背离前价')) and row.get('背离前价') != '':
            div_info = {
                'prev_price': float(row['背离前价']),
                'recent_price': float(row['背离近价']),
                'prev_macd': float(row['背离前MACD']),
                'recent_macd': float(row['背离近MACD']),
                'days_apart': int(row['背离间距日']) if pd.notna(row.get('背离间距日')) and row.get('背离间距日') != '' else 0,
            }

        r = {
            'code': code,
            'close': float(row['收盘价']),
            'change_pct': float(row['涨跌幅%']),
            'vol_ratio': float(row['量比']),
            'macd': float(row['MACD']),
            'bullish_patterns': bullish_patterns,
            'divergence_info': div_info,
        }
        candidates.append((r, level))

    print(f"  候选池: {len(candidates)} 只 -> 正在请求 DeepSeek 精选 top {top_n}...")

    if not DEEPSEEK_API_KEY:
        print("  未配置 DEEPSEEK_API_KEY，跳过 AI 筛选")
        return []

    # 从 CSV 构建 names 字典
    names = {}
    for r, _ in candidates:
        name = str(df.loc[df['代码'] == r['code'], '名称'].values[0]) if r['code'] in df['代码'].values else r['code']
        names[r['code']] = name

    return _run_deepseek_pipeline(candidates, names, scan_date, top_n)


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    import sys

    if '--csv' in sys.argv:
        # 从已有候选池 CSV 直接精选，无需重新扫描
        idx = sys.argv.index('--csv')
        csv_path = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
        if not csv_path:
            print("用法: python 9-形态选股雷达.py --csv outputs/候选池_YYYY-MM-DD.csv")
            sys.exit(1)
        top_n = int(sys.argv[idx + 2]) if idx + 2 < len(sys.argv) and sys.argv[idx + 2].isdigit() else 10
        deepseek_filter_from_csv(csv_path, top_n=top_n)
    else:
        result = run_radar()
        if result:
            deepseek_filter(result, top_n=10)
            export_candidate_pool(result)
