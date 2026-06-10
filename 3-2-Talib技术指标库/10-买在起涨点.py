# -*- coding: utf-8 -*-
"""
买在起涨点选股 -- 均线金叉 + 多头排列 + 涨停放量 + 缩量回调 + 均线止跌

选股条件（1/2/3 必须满足 + 4/5/6/7 满足任意2条）：
  硬性:
    1. 近12个交易日有5日均线上穿过60日均线，形成金叉
    2. 10, 20日均线多头排列（MA10 > MA20）
    3. 近12个交易日有过涨停（不含当天）
  柔性(4选2):
    4. 近12个交易日有过放量，且出现过上涨日的量能是前一日的1.8倍以上
    5. 放量上涨后的下跌是缩量下跌
    6. 涨停后盘中不能低于涨停日最低价，且不能连续3天收盘价低于涨停日3%价位
    7. 涨停后不能连续2天收盘价低于30日均线

运行方式:
  python "10-买在起涨点.py"                          → 当天（数据库最新交易日）
  python "10-买在起涨点.py" 2025-01-15               → 指定日期
  python "10-买在起涨点.py" 2025-01-01 2025-01-15    → 日期区间
"""
import numpy as np
import pandas as pd
import talib
import time
import os
import sys
from db_config import execute_query
from data_loader import get_instrument_names


# ============================================================
# 辅助函数
# ============================================================

def _get_limit_pct(code):
    """根据股票代码前缀判断涨跌幅限制"""
    if code.startswith(('300', '301')):
        return 0.20   # 创业板
    elif code.startswith('688'):
        return 0.20   # 科创板
    elif code.startswith(('8', '4')):
        return 0.30   # 北交所
    else:
        return 0.10   # 主板


def _is_stock(code):
    """判断是否为个股（排除 ETF/LOF/REIT 等）"""
    return not code.startswith(('51', '56', '58', '15', '16'))


# ============================================================
# 批量数据加载
# ============================================================

def batch_load_recent(days_back=150, end_date=None, start_date=None):
    """
    批量加载全市场 K 线数据

    两种模式:
      - 单日模式 (start_date=None): 加载 end_date 往前 days_back 条数据
      - 区间模式 (start_date 指定):  加载 start_date 往前 60 条(均线所需) 到 end_date

    参数:
        days_back: 每只股票保留的最大交易日数（单日模式）
        end_date:  截止日期，None 则取数据库最新日期
        start_date: 起始日期（区间模式），指定后 days_back 仅用于预加载历史

    返回:
        dict {stock_code: DataFrame}
    """
    if end_date:
        ref_date = pd.Timestamp(end_date)
    else:
        rows = execute_query("SELECT MAX(trade_date) AS latest FROM trade_stock_daily")
        if not rows or not rows[0]['latest']:
            return {}
        ref_date = pd.Timestamp(rows[0]['latest'])

    if start_date:
        # 区间模式：需要 start_date 之前至少 70 条数据来算均线
        start_dt = pd.Timestamp(start_date)
        data_start = (start_dt - pd.Timedelta(days=130)).strftime('%Y-%m-%d')  # ~70 个交易日
        data_end = ref_date.strftime('%Y-%m-%d')
    else:
        # 单日模式：只需 end_date 往前 days_back 条
        calendar_days = int(days_back * 1.8)
        data_start = (ref_date - pd.Timedelta(days=calendar_days)).strftime('%Y-%m-%d')
        data_end = ref_date.strftime('%Y-%m-%d')

    print(f"  SQL查询范围: {data_start} ~ {data_end}")

    sql = """
        SELECT stock_code, trade_date, open_price, high_price, low_price, close_price, volume
        FROM trade_stock_daily
        WHERE trade_date >= %s AND trade_date <= %s
        ORDER BY stock_code, trade_date ASC
    """
    rows = execute_query(sql, [data_start, data_end])
    if not rows:
        return {}

    df_all = pd.DataFrame(rows)
    df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])
    for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    min_bars = 65  # 60日均线至少需要 60 条数据，留 5 条余量
    result = {}
    for code, group in df_all.groupby('stock_code'):
        sub = group.set_index('trade_date').sort_index()
        sub = sub[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]
        sub.columns = ['open', 'high', 'low', 'close', 'volume']
        if len(sub) >= min_bars:
            result[code] = sub.tail(days_back if not start_date else len(sub))

    return result


# ============================================================
# 单只股票选股条件检查（核心）
# ============================================================

def check_stock(df, code, target_idx=-1, flex_enabled=None, flex_min=2):
    """
    对单只股票在指定 K 线位置运行 7 个选股条件检查

    核心设计：所有条件以 target_idx 为"当前"来判定。
    target_idx=-1（默认）表示以最后一根 K 线为当前；
    传入其他负值可检查历史上某一天是否满足条件。

    参数:
        df:          单只股票的 DataFrame，列为 open/high/low/close/volume
        code:        股票代码
        target_idx:  目标 K 线索引（负值=从末尾倒数），-1 表示最后一天
        flex_enabled: 启用的柔性条件编号列表，默认 [4,5,6,7]
        flex_min:    柔性条件最少通过数，默认 2

    返回:
        (bool, dict) -> (是否全部满足, 条件明细字典)
    """
    if flex_enabled is None:
        flex_enabled = [4, 5, 6, 7]
    o = df['open'].values.astype(np.float64)
    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)

    n = len(c)
    if target_idx < 0:
        target_idx = n + target_idx  # 转为绝对索引

    # 截断到 target_idx（包含），后续用 -1 即代表 target_idx
    o = o[:target_idx + 1]
    h = h[:target_idx + 1]
    l = l[:target_idx + 1]
    c = c[:target_idx + 1]
    v = v[:target_idx + 1]
    n = len(c)

    detail = {}

    # ---- 计算所需均线 ----
    ma5 = talib.SMA(c, timeperiod=5)
    ma10 = talib.SMA(c, timeperiod=10)
    ma20 = talib.SMA(c, timeperiod=20)
    ma30 = talib.SMA(c, timeperiod=30)
    ma60 = talib.SMA(c, timeperiod=60)

    if np.isnan(ma60[-1]) or np.isnan(ma5[-1]):
        return False, {}

    # ================================================================
    # 条件 1: 近12个交易日有5日均线上穿过60日均线，形成金叉
    # ================================================================
    golden_cross = False
    cross_idx = -1
    for i in range(-12, 0):
        if i - 1 >= -n and not np.isnan(ma5[i]) and not np.isnan(ma60[i]) \
           and not np.isnan(ma5[i-1]) and not np.isnan(ma60[i-1]):
            if ma5[i-1] <= ma60[i-1] and ma5[i] > ma60[i]:
                golden_cross = True
                cross_idx = i
                break

    detail['cond1_golden_cross'] = golden_cross
    if golden_cross:
        detail['cross_days_ago'] = abs(cross_idx)

    # ================================================================
    # 条件 2: 10, 20日均线多头排列（MA10 > MA20）
    # ================================================================
    mas = [ma10[-1], ma20[-1]]
    bull_alignment = all(not np.isnan(x) for x in mas) and \
                     all(mas[i] > mas[i+1] for i in range(len(mas)-1))
    detail['cond2_bull_alignment'] = bull_alignment
    if bull_alignment:
        detail['ma5'] = round(float(ma5[-1]), 2)
        detail['ma10'] = round(float(ma10[-1]), 2)
        detail['ma20'] = round(float(ma20[-1]), 2)
        detail['ma30'] = round(float(ma30[-1]), 2)
        detail['ma60'] = round(float(ma60[-1]), 2)

    # ================================================================
    # 条件 3: 近12个交易日有过涨停（不含当天，涨停日必须是过去）
    #   range(-2, -13, -1): 从昨天往12天前找 → 涨停日至少在1天前
    #   当天涨停的不算"起涨点"——我们要的是涨停后回调、止跌的买点
    # ================================================================
    limit_pct = _get_limit_pct(code)
    limit_threshold = limit_pct * 0.95

    has_limit_up = False
    limit_up_idx = -1
    for i in range(-2, -13, -1):
        if i - 1 >= -n and c[i-1] > 0:
            change = (c[i] / c[i-1] - 1)
            if change >= limit_threshold:
                has_limit_up = True
                limit_up_idx = i
                break

    detail['cond3_limit_up'] = has_limit_up
    if has_limit_up:
        detail['limit_up_days_ago'] = abs(limit_up_idx)
        detail['limit_up_change'] = round(float(c[limit_up_idx] / c[limit_up_idx-1] - 1) * 100, 1)

    # ================================================================
    # 条件 4: 近12个交易日有过放量，且上涨日的量能是前一日的1.8倍以上
    #   range(-1, -13, -1): 从当前往12天前遍历 → 找到的第一个就是"最近"的
    #   c[i] > o[i]:        收阳，确保是上涨日而非放量下跌
    #   v[i]/v[i-1]:        量比 ≥ 阈值 → 大资金介入的痕迹
    #   vol_exp_idx = None: 用 None 做哨兵，避免和合法的负索引 -1 冲突
    # ================================================================
    vol_exp_idx = None
    for i in range(-1, -13, -1):
        if i - 1 >= -n and v[i-1] > 0:
            is_up_day = c[i] > o[i]
            vol_ratio = v[i] / v[i-1]
            if is_up_day and vol_ratio >= 2:
                vol_exp_idx = i
                break

    detail['cond4_vol_expand'] = vol_exp_idx is not None
    if vol_exp_idx is not None:
        detail['vol_expand_days_ago'] = abs(vol_exp_idx)
        detail['vol_expand_ratio'] = round(float(v[vol_exp_idx] / v[vol_exp_idx-1]), 1)

    # ================================================================
    # 条件 5: 放量上涨后的下跌是缩量下跌
    #   vol_exp_idx < -1: 放量上涨日之后至少还有一根K线才需要检查（None < -1 为 False，安全）
    #   post_slice:       放量上涨日之后 → 当前（不含放量日当天）
    #   down_days_mask:   筛选出收盘价 < 放量日收盘价的"下跌日"
    #   down_volumes:     所有下跌日的成交量
    #   核心判断: 下跌日的量全部 < 放量日量 × 0.85 → 卖压枯竭，回调健康
    #   如果放量日后没有下跌日（一直在涨），条件不满足（未出现回调买点）
    # ================================================================
    cond5_ok = False
    if vol_exp_idx is not None and vol_exp_idx < -1:
        post_slice = slice(vol_exp_idx + 1, None)
        post_close = c[post_slice]
        post_vol = v[post_slice]

        down_days_mask = post_close < c[vol_exp_idx]
        if down_days_mask.any():
            down_volumes = post_vol[down_days_mask]
            expand_vol = v[vol_exp_idx]
            # 每一根下跌K线的量都必须 < 放量日量 × 0.85
            all_shrinking = np.all(down_volumes < expand_vol * 0.85)
            cond5_ok = all_shrinking

    detail['cond5_shrink_decline'] = cond5_ok

    # ================================================================
    # 条件 6: 涨停后盘中不能低于涨停日最低价，且不能连续3天收盘价低于涨停日3%价位
    #  涨停日之后:
    #    (a) 任意一天的盘中最低价 >= 涨停日最低价（不能破涨停日低点）
    #    (b) 收盘价低于涨停日收盘价*0.97 的连续天数 < 3（回调不深且不久）
    # ================================================================
    cond6_ok = False
    if has_limit_up and limit_up_idx < -1:
        # 涨停日之后至少有一根 K 线才需要检查
        limit_low = l[limit_up_idx]
        limit_close = c[limit_up_idx]
        threshold_97 = limit_close * 0.97

        post_lows = l[limit_up_idx + 1:]
        post_closes = c[limit_up_idx + 1:]

        # (a) 盘中最低价不低于涨停日最低价
        low_not_breached = bool(np.all(post_lows >= limit_low))

        # (b) 收盘价低于 97% 价位的最长连续天数 < 3
        below_97 = post_closes < threshold_97
        max_consecutive = 0
        streak = 0
        for v in below_97:
            if v:
                streak += 1
                max_consecutive = max(max_consecutive, streak)
            else:
                streak = 0
        no_3day_below = max_consecutive < 3

        cond6_ok = low_not_breached and no_3day_below
        detail['limit_up_low'] = round(float(limit_low), 2)
        detail['limit_up_close'] = round(float(limit_close), 2)
        detail['threshold_97pct'] = round(float(threshold_97), 2)
        detail['low_breached'] = not low_not_breached
        detail['max_below_97_days'] = max_consecutive
    elif has_limit_up:
        # 涨停日就是当天，无后续K线可检查，默认通过
        cond6_ok = True
        detail['limit_up_low'] = round(float(l[limit_up_idx]), 2)
        detail['limit_up_close'] = round(float(c[limit_up_idx]), 2)
        detail['threshold_97pct'] = round(float(c[limit_up_idx] * 0.97), 2)
        detail['low_breached'] = False
        detail['max_below_97_days'] = 0

    detail['cond6_limit_up_support'] = cond6_ok

    # ================================================================
    # 条件 7: 涨停后不能连续2天收盘价低于30日均线
    # ================================================================
    cond7_ok = False
    if has_limit_up and limit_up_idx < -1:
        post_closes = c[limit_up_idx + 1:]
        post_ma30 = ma30[limit_up_idx + 1:]
        below_ma30 = post_closes < post_ma30
        max_below = 0
        streak = 0
        for v in below_ma30:
            if bool(v) and not np.isnan(post_ma30[streak]):
                streak += 1
                max_below = max(max_below, streak)
            else:
                streak = 0
        cond7_ok = max_below < 2
        detail['max_below_ma30_days'] = max_below
    elif has_limit_up:
        cond7_ok = True
        detail['max_below_ma30_days'] = 0
    else:
        cond7_ok = c[-1] >= ma30[-1]
        detail['max_below_ma30_days'] = 0 if cond7_ok else 1

    detail['cond7_above_ma30'] = cond7_ok

    # ---- 汇总判断 ----
    # 条件 1, 2, 3 必须全部满足（硬性条件）
    core_ok = golden_cross and bull_alignment and has_limit_up
    # 条件 4, 5, 6, 7 按参数启用的条件中，满足 flex_min 条即可
    flex_map = {
        4: vol_exp_idx is not None,
        5: cond5_ok,
        6: cond6_ok,
        7: cond7_ok,
    }
    flexible = [flex_map[c] for c in flex_enabled]
    flexible_count = sum(flexible)
    passed = core_ok and flexible_count >= flex_min

    if passed:
        detail['close'] = round(float(c[-1]), 2)
        detail['change_pct'] = round(float((c[-1] / c[-2] - 1) * 100), 2) if n >= 2 else 0
        vol_ma20 = talib.SMA(v, timeperiod=20)
        detail['vol_ratio'] = round(float(v[-1] / vol_ma20[-1]), 1) if not np.isnan(vol_ma20[-1]) and vol_ma20[-1] > 0 else 0
        detail['flexible_passed'] = flexible_count

    return passed, detail


# ============================================================
# 选股扫描（单日 / 区间两种模式）
# ============================================================

def _calc_score(r):
    """综合评分：均线发散 + 量比 + 涨停强度"""
    score = 0
    if r.get('cond2_bull_alignment'):
        try:
            spread = (r['ma5'] - r['ma60']) / r['ma60'] * 100
            score += min(spread, 20)
        except Exception:
            pass
    score += r.get('vol_ratio', 0) * 2
    if r.get('cond3_limit_up'):
        score += min(r.get('limit_up_change', 0), 10)
    return round(score, 1)


def _scan_single_day(all_data, scan_date_str, flex_enabled=None, flex_min=2):
    """单日模式：检查每只股票在最后一天是否满足条件"""
    passed_list = []
    detail_list = []
    errors = 0
    total = len(all_data)

    for idx, (code, df) in enumerate(all_data.items(), 1):
        if idx % 1000 == 0:
            print(f"  进度: {idx}/{total} ({idx/total*100:.0f}%), 已选出 {len(passed_list)} 只")

        try:
            passed, detail = check_stock(df, code, flex_enabled=flex_enabled, flex_min=flex_min)
            if passed:
                detail['code'] = code
                detail['match_date'] = scan_date_str
                detail['score'] = _calc_score(detail)
                passed_list.append(detail)
            if detail:
                detail_list.append(detail)
        except Exception:
            errors += 1

    return passed_list, detail_list, errors


def _scan_date_range(all_data, start_date, end_date, flex_enabled=None, flex_min=2):
    """
    区间模式：逐日检查每只股票，找出区间内所有满足条件的 (股票, 日期)

    返回:
        passed_list: [{...code, match_date, ...}, ...] 按日期+评分排序
    """
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    passed_list = []
    errors = 0
    total = len(all_data)
    total_checks = 0

    for idx, (code, df) in enumerate(all_data.items(), 1):
        if idx % 500 == 0:
            print(f"  进度: {idx}/{total} ({idx/total*100:.0f}%), "
                  f"已匹配 {len(passed_list)} 条, 检查 {total_checks} 次")

        try:
            date_mask = (df.index >= start_dt) & (df.index <= end_dt)
            target_dates = df.index[date_mask]

            if len(target_dates) == 0:
                continue

            date_to_iloc = {d: i for i, d in enumerate(df.index)}

            for match_dt in target_dates:
                abs_idx = date_to_iloc[match_dt]
                rel_idx = abs_idx - len(df)

                total_checks += 1
                passed, detail = check_stock(df, code, target_idx=rel_idx, flex_enabled=flex_enabled, flex_min=flex_min)
                if passed:
                    detail['code'] = code
                    detail['match_date'] = match_dt.strftime('%Y-%m-%d')
                    detail['score'] = _calc_score(detail)
                    passed_list.append(detail)
        except Exception:
            errors += 1

    print(f"  总检查次数: {total_checks}, 匹配 {len(passed_list)} 条")
    return passed_list, [], errors


def run_screener(start_date=None, end_date=None, flex_enabled=None, flex_min=2):
    """
    买在起涨点 -- 全市场选股扫描

    两种模式:
      - 单日模式: start_date=None, end_date=指定日期或None(最新)
      - 区间模式: start_date 和 end_date 同时指定

    参数:
        start_date: 起始日期 'YYYY-MM-DD'，None=单日模式
        end_date:   截止日期 'YYYY-MM-DD'，None=数据库最新日期

    返回:
        dict 包含扫描结果
    """
    is_range_mode = start_date is not None

    print("=" * 70)
    if is_range_mode:
        print(f"买在起涨点选股 - 区间扫描")
        print(f"  日期范围: {start_date} ~ {end_date or '(最新)'}")
    else:
        print(f"买在起涨点选股 - 单日扫描")
        print(f"  目标日期: {end_date or '(最新)'}")
    fe = flex_enabled or [4, 5, 6, 7]
    print(f"  硬性(1/2/3) + 柔性({fe} {len(fe)}选{flex_min})")
    print("=" * 70)

    # ---- 第 1 步：批量加载数据 ----
    print("\n[1/3] 批量加载K线数据...")
    t0 = time.time()
    all_data = batch_load_recent(days_back=150, end_date=end_date, start_date=start_date)
    load_time = time.time() - t0
    print(f"  加载完成: {len(all_data)} 只标的, 耗时 {load_time:.1f}s")

    if not all_data:
        print("  没有可用数据, 请先运行 1-行情数据采集.py")
        return

    # ---- 第 2 步：扫描 ----
    if is_range_mode:
        scan_label = f"{start_date}_{end_date or 'latest'}"
        scan_date_str = f"{start_date} ~ {end_date or 'latest'}"
        print(f"\n[2/3] 运行区间选股扫描 ({scan_date_str})...")
        t0 = time.time()
        passed_list, detail_list, errors = _scan_date_range(all_data, start_date, end_date or None, flex_enabled=flex_enabled, flex_min=flex_min)
    else:
        sample_code = next(iter(all_data))
        scan_date_str = all_data[sample_code].index[-1].strftime('%Y-%m-%d')
        scan_label = scan_date_str
        print(f"\n[2/3] 运行选股条件检查 (目标日期: {scan_date_str})...")
        t0 = time.time()
        passed_list, detail_list, errors = _scan_single_day(all_data, scan_date_str, flex_enabled=flex_enabled, flex_min=flex_min)

    scan_time = time.time() - t0
    print(f"  扫描完成: 耗时 {scan_time:.1f}s" +
          (f", 跳过异常 {errors} 只" if errors else ""))

    total = len(all_data)

    # ---- 第 3 步：输出结果 ----
    print(f"\n[3/3] 选股结果 ({scan_date_str})")
    print("=" * 70)

    if not passed_list:
        print("\n  当前市场无符合条件的股票")
        if not is_range_mode and detail_list:
            _print_condition_stats(detail_list, total)
        return {
            'scan_date': scan_date_str,
            'total': total,
            'passed': [],
        }

    # 排序：区间模式按日期+评分，单日模式按评分
    if is_range_mode:
        passed_list.sort(key=lambda x: (x['match_date'], -x['score']))
    else:
        passed_list.sort(key=lambda x: x['score'], reverse=True)

    # ---- 打印结果 ----
    _print_results(passed_list, is_range_mode)

    # ---- 输出统计 ----
    print(f"\n{'='*70}")
    print(f"扫描汇总 ({scan_date_str})")
    print(f"{'='*70}")
    print(f"  扫描标的:     {total} 只")
    if is_range_mode:
        # 去重统计：区间内有多少只不同的股票出现过信号
        unique_codes = len(set(r['code'] for r in passed_list))
        print(f"  信号总数:     {len(passed_list)} 条")
        print(f"  去重标的:     {unique_codes} 只")
    else:
        print(f"  符合7条件:    {len(passed_list)} 只 ({len(passed_list)/total*100:.2f}%)")
    print(f"  总耗时:       {load_time + scan_time:.1f}s")

    # 打印各条件命中率（仅单日模式有意义）
    if not is_range_mode and detail_list:
        _print_condition_stats(detail_list, total)

    # ---- 导出 CSV ----
    _export_csv(passed_list, scan_label, is_range_mode)

    return {
        'scan_date': scan_date_str,
        'total': total,
        'passed': passed_list,
    }


# ============================================================
# 输出辅助函数
# ============================================================

def _print_results(passed_list, is_range_mode=False):
    """打印选股结果表格"""
    header_date = " 日期" if is_range_mode else ""
    print(f"\n  符合条件的股票 (共 {len(passed_list)} 条):")
    header = (f"  {header_date:<12} {'代码':<14} {'收盘':>8} {'涨跌%':>7} {'量比':>6} "
              f"{'柔性':>4} {'评分':>6} "
              f"{'MA5':>8} {'MA10':>8} {'MA20':>8} {'MA30':>8} {'MA60':>8} "
              f"{'涨停低':>8} {'金叉(日前)':>10} {'涨停(日前%))':>13} "
              f"{'破涨停低':>8} {'连<97%':>7}")
    print(header)
    print(f"  {'-'*len(header)}")

    for r in passed_list:
        date_col = f"{r.get('match_date', ''):<12} " if is_range_mode else ""
        print(f"  {date_col}{r['code']:<14} {r['close']:>8.2f} {r['change_pct']:>+6.2f}% "
              f"{r['vol_ratio']:>5.1f}x "
              f"{r.get('flexible_passed', 0):>3}/4 "
              f"{r['score']:>5.1f} "
              f"{r.get('ma5', 0):>8.2f} {r.get('ma10', 0):>8.2f} "
              f"{r.get('ma20', 0):>8.2f} {r.get('ma30', 0):>8.2f} "
              f"{r.get('ma60', 0):>8.2f} "
              f"{r.get('limit_up_low', 0):>8.2f} "
              f"{r.get('cross_days_ago', '?'):>5}日前 "
              f"{r.get('limit_up_change', 0):>+5.1f}%({r.get('limit_up_days_ago', '?'):>2}日前) "
              f"{'是' if r.get('low_breached') else '否':>8} "
              f"{r.get('max_below_97_days', 0):>5}日")


def _print_condition_stats(detail_list, total):
    """打印各条件的单项命中率统计"""
    core_conds = [
        ('cond1_golden_cross',   '硬性1: 12日内金叉60日'),
        ('cond2_bull_alignment', '硬性2: 10/20多头排列'),
        ('cond3_limit_up',       '硬性3: 近12日有涨停'),
    ]
    flex_conds = [
        ('cond4_vol_expand',     '柔性4: 放量上涨1.8倍'),
        ('cond5_shrink_decline', '柔性5: 缩量下跌'),
        ('cond6_limit_up_support', '柔性6: 涨停后支撑有效'),
        ('cond7_above_ma30',     '柔性7: 不连续2日低于MA30'),
    ]
    print(f"\n  各条件单项命中率 (共 {len(detail_list)} 只有效数据):")
    print(f"  [硬性条件]")
    for key, label in core_conds:
        cnt = sum(1 for d in detail_list if d.get(key))
        print(f"    {label:<24} {cnt:>5} 只 ({cnt/total*100:>5.1f}%)")
    print(f"  [柔性条件]")
    for key, label in flex_conds:
        cnt = sum(1 for d in detail_list if d.get(key))
        print(f"    {label:<24} {cnt:>5} 只 ({cnt/total*100:>5.1f}%)")


def _export_csv(passed_list, scan_label, is_range_mode=False):
    """导出选股结果 CSV"""
    os.makedirs('outputs', exist_ok=True)

    rows = []
    for r in passed_list:
        row = {
            '代码': r['code'],
            '名称': r['code'],
            '类型': '个股' if _is_stock(r['code']) else 'ETF/基金',
            '收盘价': r['close'],
            '涨跌幅%': r['change_pct'],
            '量比': r['vol_ratio'],
            '柔性通过(4/5/6/7)': f"{r.get('flexible_passed', 0)}/4",
            '综合评分': r['score'],
            'MA5': r.get('ma5', ''),
            'MA10': r.get('ma10', ''),
            'MA20': r.get('ma20', ''),
            'MA30': r.get('ma30', ''),
            'MA60': r.get('ma60', ''),
            '涨停日最低': r.get('limit_up_low', ''),
            '涨停日收盘': r.get('limit_up_close', ''),
            '97%阈值价位': r.get('threshold_97pct', ''),
            '破涨停日低点': '是' if r.get('low_breached') else '否',
            '收盘连续低于97%天数': r.get('max_below_97_days', ''),
            '金叉天数': r.get('cross_days_ago', ''),
            '涨停涨幅%': r.get('limit_up_change', ''),
            '涨停天数': r.get('limit_up_days_ago', ''),
            '放量倍数': r.get('vol_expand_ratio', ''),
            '放量天数': r.get('vol_expand_days_ago', ''),
        }
        if is_range_mode:
            row['匹配日期'] = r.get('match_date', '')
        rows.append(row)

    df = pd.DataFrame(rows)

    # 调整列顺序：区间模式下把匹配日期放前面
    if is_range_mode and '匹配日期' in df.columns:
        cols = ['匹配日期'] + [c for c in df.columns if c != '匹配日期']
        df = df[cols]

    prefix = '买在起涨点_区间' if is_range_mode else '买在起涨点'
    csv_path = os.path.join('outputs', f'{prefix}_{scan_label}.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n  已保存: {csv_path}")

    # 补充股票名称后重新保存
    codes = list(set(r['code'] for r in passed_list))
    names = get_instrument_names(codes) if codes else {}
    if names:
        for row in rows:
            row['名称'] = names.get(row['代码'], row['代码'])
        df = pd.DataFrame(rows)
        if is_range_mode and '匹配日期' in df.columns:
            cols = ['匹配日期'] + [c for c in df.columns if c != '匹配日期']
            df = df[cols]
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')


# ============================================================
# 入口
# ============================================================

if __name__ == '__main__':
    # 解析位置参数（日期）
    pos_args = [a for a in sys.argv[1:] if not a.startswith('--')]
    # 解析命名参数
    flex_enabled = None
    flex_min = 2
    for a in sys.argv[1:]:
        if a.startswith('--flex='):
            flex_enabled = [int(x.strip()) for x in a.split('=', 1)[1].split(',')]
        elif a.startswith('--flex-min='):
            flex_min = int(a.split('=', 1)[1])

    if len(pos_args) == 0:
        run_screener(flex_enabled=flex_enabled, flex_min=flex_min)
    elif len(pos_args) == 1:
        run_screener(end_date=pos_args[0], flex_enabled=flex_enabled, flex_min=flex_min)
    elif len(pos_args) >= 2:
        run_screener(start_date=pos_args[0], end_date=pos_args[1], flex_enabled=flex_enabled, flex_min=flex_min)
