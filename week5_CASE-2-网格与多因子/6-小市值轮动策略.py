# -*- coding: utf-8 -*-
"""
小市值轮动策略 - 市值因子 + 月度轮动
====================================

核心思路:
---------
A股有一个长期有效的"小市值效应" (Size Effect):
  小市值股票组的长期收益率 > 大市值股票组的长期收益率
  (学术上称为 SMB: Small Minus Big, Fama-French 三因子模型之一)

本脚本实现:
-----------
1. 用流通市值 (或总市值) 作为核心排序因子
2. 每月末选出市值最小的 N 只股票
3. 等权重配置, 持有至下月末
4. 增加动量 + 波动率过滤, 排除"垃圾小票"

过滤筛选逻辑:
  1. 纯小市值: 只看市值排名
  2. 增强版 (市值 + 动量 + 波动率):
     - 先按市值选候选池 (pool_size = 50)
     - 排除 20 日动量 < -20% 的暴跌股
     - 排除波动率 > 5% 的极端波动股
     - 在剩余股票中按"低市值 + 正动量 + 低波动"打分

注意:
  本脚本通过 close * avg_volume 估算"相对市值",
  实际生产环境中应该使用 trade_stock_financial 中的总市值/流通市值字段。

对比:
  - 纯小市值策略 (只看市值)
  - 增强版 (市值 + 动量 + 波动率过滤)

运行方式:
  python 6-小市值轮动策略.py
"""
import numpy as np
import pandas as pd
import talib
import time
from db_config import execute_query, INITIAL_CASH


# ============================================================
# 数据加载
# ============================================================

def batch_load_daily(start_date, end_date, min_bars=60):
    """批量加载日K线, 返回 dict {code: DataFrame}"""
    sql = """
        SELECT stock_code, trade_date, open_price, high_price, low_price,
               close_price, volume
        FROM trade_stock_daily
        WHERE trade_date >= %s AND trade_date <= %s
        ORDER BY stock_code, trade_date ASC
    """
    rows = execute_query(sql, [start_date, end_date])
    if not rows:
        return {}

    df_all = pd.DataFrame(rows)
    df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])
    for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    result = {}
    for code, group in df_all.groupby('stock_code'):
        sub = group.set_index('trade_date').sort_index()
        sub = sub[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]
        sub.columns = ['open', 'high', 'low', 'close', 'volume']
        if len(sub) >= min_bars:
            result[code] = sub
    return result


def load_market_cap(date_str):
    """
    从 trade_stock_financial 加载最近一期的总市值。

    如果财务表没有数据, 用 close * volume 估算相对市值排名。

    参数:
        date_str: 日期字符串, 用于筛选 report_date <= 该日期

    返回值:
        dict {stock_code: total_assets}
    """
    sql = """
        SELECT stock_code, total_assets
        FROM trade_stock_financial
        WHERE report_date <= %s
        ORDER BY report_date DESC
    """
    rows = execute_query(sql, [date_str])
    if rows:
        df = pd.DataFrame(rows)
        df['total_assets'] = pd.to_numeric(df['total_assets'], errors='coerce')
        # 每个股票取最新一期
        df = df.groupby('stock_code').first().reset_index()
        return dict(zip(df['stock_code'], df['total_assets']))
    return {}


# ============================================================
# 小市值排序
# ============================================================

def rank_by_market_cap(all_data, calc_date, top_n=20):
    """
    按"相对市值"排序选出市值最小的股票。

    使用 close_price * avg_volume_20d 作为市值代理指标。
    (真实场景应使用流通市值, 但由于数据限制, 使用此近似方法)

    参数:
        all_data: dict {code: DataFrame}
        calc_date: 计算日期 (pd.Timestamp)
        top_n: 选股数量

    返回值:
        list[str], 小市值股票代码列表
    """
    cap_dict = {}
    for code, df in all_data.items():
        sub = df[df.index <= calc_date]
        if len(sub) < 20:
            continue
        close = float(sub['close'].iloc[-1])
        avg_vol = float(sub['volume'].iloc[-20:].mean())
        if close > 0 and avg_vol > 0:
            # 估算市值 ≈ 价格 × 日均成交量
            # 严格来说应该是 price * 流通股数, 但这里用成交量作为活跃度的代理
            cap_dict[code] = close * avg_vol

    if not cap_dict:
        return []

    # 按市值从小到大排序
    sorted_caps = sorted(cap_dict.items(), key=lambda x: x[1])
    return [code for code, _ in sorted_caps[:top_n]]


def rank_enhanced(all_data, calc_date, top_n=20, pool_size=50):
    """
    增强版小市值选股 (市值 + 动量 + 波动率过滤)。

    两步法:
      1. 市值筛选: 选出市值最小的 pool_size 只股票作为候选池
      2. 因子打分: 在候选池中按"低市值 + 正动量 + 低波动"综合评分

    过滤条件:
      - 排除 20 日动量 < -20% (近期暴跌, 可能有未释放的风险)
      - 排除波动率 > 5% (极端波动, 不确定性高)

    打分公式:
      score = 市值排名分(0.5) + 动量分(0.3) + 波动率分(0.2)

    参数:
        all_data: dict {code: DataFrame}
        calc_date: 计算日期
        top_n: 最终选股数量
        pool_size: 候选池大小 (默认50)

    返回值:
        list[str], 选出的股票代码列表
    """
    # ---- 第一步: 市值排名选候选池 ----
    cap_dict = {}
    for code, df in all_data.items():
        sub = df[df.index <= calc_date]
        if len(sub) < 60:
            continue
        close = float(sub['close'].iloc[-1])
        avg_vol = float(sub['volume'].iloc[-20:].mean())
        if close > 0 and avg_vol > 0:
            cap_dict[code] = close * avg_vol

    if len(cap_dict) < pool_size:
        pool_size = len(cap_dict) // 2

    # 按市值从小到大排序, 取最小的 pool_size 只
    sorted_caps = sorted(cap_dict.items(), key=lambda x: x[1])
    candidates = [code for code, _ in sorted_caps[:pool_size]]

    # ---- 第二步: 技术因子过滤 + 打分 ----
    scores = {}
    for code in candidates:
        df = all_data[code]
        sub = df[df.index <= calc_date]
        if len(sub) < 60:
            continue

        c = sub['close'].values.astype(np.float64)
        h = sub['high'].values.astype(np.float64)
        l = sub['low'].values.astype(np.float64)

        try:
            roc_20 = talib.ROC(c, timeperiod=20)        # 20日动量
            atr = talib.ATR(h, l, c, timeperiod=14)     # ATR波动率

            m20 = float(roc_20[-1]) if not np.isnan(roc_20[-1]) else 0
            vol = float(atr[-1] / c[-1]) if not np.isnan(atr[-1]) and c[-1] > 0 else 0

            # 过滤: 排除暴跌股 (动量 < -20%)
            if m20 < -20:
                continue
            # 过滤: 排除极端波动股 (波动率 > 5%)
            if vol > 0.05:
                continue

            # 综合打分: 低市值优先 + 正动量 + 低波动
            cap_rank = candidates.index(code) / len(candidates)  # 0~1, 越小市值越小
            # score = 50%市值 + 30%动量 + 20%低波动
            score = (1 - cap_rank) * 0.5 + min(m20, 30) / 30 * 0.3 + (1 - min(vol, 0.05) / 0.05) * 0.2
            scores[code] = score
        except Exception:
            continue

    # 按得分从高到低排序
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [code for code, _ in sorted_scores[:top_n]]


# ============================================================
# 月度轮动回测
# ============================================================

def rotation_backtest(all_data, select_fn, top_n=10, label=''):
    """
    轮动策略回测 (通用框架)。

    每月末调用 select_fn 选股, 等权持有至下月末。

    参数:
        all_data: dict {code: DataFrame}
        select_fn: 选股函数, 签名: select_fn(all_data, calc_date, top_n) -> [codes]
        top_n: 选股数量
        label: 策略名称 (用于显示)

    返回值:
        dict {nav_log, rebalance_log, final_nav}
    """
    cash = INITIAL_CASH

    # 交易日并集
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)

    # 月末日期
    rebalance_dates = []
    for i, d in enumerate(all_dates):
        if i + 1 < len(all_dates) and all_dates[i + 1].month != d.month:
            rebalance_dates.append(d)

    if len(rebalance_dates) < 2:
        return None

    nav = cash
    nav_log = [{'date': rebalance_dates[0], 'nav': nav}]
    rb_log = []

    for i in range(len(rebalance_dates) - 1):
        rb_date = rebalance_dates[i]
        next_rb = rebalance_dates[i + 1]

        selected = select_fn(all_data, rb_date, top_n)
        if not selected:
            nav_log.append({'date': next_rb, 'nav': nav})
            continue

        # 等权持有计算收益
        rets = []
        valid_stocks = []
        for code in selected:
            if code not in all_data:
                continue
            df = all_data[code]
            if rb_date not in df.index or next_rb not in df.index:
                continue
            c1 = float(df.loc[rb_date, 'close'])
            c2 = float(df.loc[next_rb, 'close'])
            if c1 > 0:
                rets.append(c2 / c1 - 1)
                valid_stocks.append(code)

        port_ret = np.mean(rets) if rets else 0
        nav *= (1 + port_ret)

        nav_log.append({'date': next_rb, 'nav': nav})
        rb_log.append({
            'date': rb_date,
            'stocks': valid_stocks,
            'return': port_ret,
        })

    return {'nav_log': nav_log, 'rebalance_log': rb_log, 'final_nav': nav}


def calc_metrics(result):
    """
    计算轮动策略的绩效指标。

    与脚本5的 calc_backtest_metrics 逻辑相同。
    """
    navs = [x['nav'] for x in result['nav_log']]
    dates = [x['date'] for x in result['nav_log']]
    rets = [r['return'] for r in result['rebalance_log']]

    total_ret = navs[-1] / INITIAL_CASH - 1
    days = (dates[-1] - dates[0]).days
    years = days / 365.25 if days > 0 else 1
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else total_ret

    peak = navs[0]
    max_dd = 0
    for v in navs:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

    if rets:
        std_m = np.std(rets)
        sharpe = (np.mean(rets) * 12 - 0.02) / (std_m * np.sqrt(12)) if std_m > 0 else 0
        win_rate = sum(1 for r in rets if r > 0) / len(rets)
    else:
        sharpe = 0
        win_rate = 0

    calmar = ann_ret / max_dd if max_dd > 0 else 0

    return {
        'total_return': total_ret,
        'annual_return': ann_ret,
        'max_drawdown': max_dd,
        'sharpe': round(sharpe, 4),
        'calmar': round(calmar, 4),
        'win_rate': win_rate,
        'periods': len(rets),
    }


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    start_date = '2024-01-01'
    end_date = '2025-12-31'

    print("=" * 70)
    print("小市值轮动策略")
    print("=" * 70)
    print("\n策略逻辑:")
    print("  每月末选出市值最小的N只股票, 等权持有至下月末")
    print("  增强版: 在小市值候选池中, 加入动量+波动率过滤")

    # ---- 1. 加载数据 ----
    print(f"\n[1] 加载数据...")
    t0 = time.time()
    all_data = batch_load_daily(start_date, end_date, min_bars=60)
    print(f"    {len(all_data)} 只标的, 耗时 {time.time()-t0:.1f}s")

    if len(all_data) < 20:
        print("  标的数量不足")
        exit()

    # ---- 2. 纯小市值策略 ----
    print(f"\n{'=' * 70}")
    print("[2] 纯小市值策略 (Top-10)")
    print(f"{'=' * 70}")

    r_pure = rotation_backtest(
        all_data, rank_by_market_cap, top_n=10, label='纯小市值'
    )
    if r_pure:
        m_pure = calc_metrics(r_pure)
        print(f"  总收益: {m_pure['total_return']*100:+.2f}% | "
              f"年化: {m_pure['annual_return']*100:+.2f}% | "
              f"最大回撤: {m_pure['max_drawdown']*100:.2f}%")
        print(f"  夏普: {m_pure['sharpe']:.2f} | "
              f"卡玛: {m_pure['calmar']:.2f} | "
              f"月度胜率: {m_pure['win_rate']*100:.1f}% | "
              f"调仓: {m_pure['periods']}期")

        print(f"\n  调仓记录(前5期):")
        for rb in r_pure['rebalance_log'][:5]:
            stocks_str = ', '.join(rb['stocks'][:4])
            if len(rb['stocks']) > 4:
                stocks_str += f' ...+{len(rb["stocks"])-4}'
            print(f"    {rb['date'].strftime('%Y-%m-%d')} | "
                  f"收益={rb['return']*100:+.1f}% | {stocks_str}")

    # ---- 3. 增强版小市值 ----
    print(f"\n{'=' * 70}")
    print("[3] 增强版小市值 (市值+动量+波动率过滤, Top-10)")
    print(f"{'=' * 70}")

    r_enhanced = rotation_backtest(
        all_data, rank_enhanced, top_n=10, label='增强小市值'
    )
    if r_enhanced:
        m_enhanced = calc_metrics(r_enhanced)
        print(f"  总收益: {m_enhanced['total_return']*100:+.2f}% | "
              f"年化: {m_enhanced['annual_return']*100:+.2f}% | "
              f"最大回撤: {m_enhanced['max_drawdown']*100:.2f}%")
        print(f"  夏普: {m_enhanced['sharpe']:.2f} | "
              f"卡玛: {m_enhanced['calmar']:.2f} | "
              f"月度胜率: {m_enhanced['win_rate']*100:.1f}% | "
              f"调仓: {m_enhanced['periods']}期")

    # ---- 4. 不同 Top-N 对比 ----
    print(f"\n{'=' * 70}")
    print("[4] 不同 Top-N 对比 (增强版)")
    print(f"{'=' * 70}")

    all_metrics = {}
    for n in [5, 10, 20]:
        r = rotation_backtest(all_data, rank_enhanced, top_n=n)
        if r:
            all_metrics[n] = calc_metrics(r)

    if all_metrics:
        print(f"\n  {'Top-N':<10} {'总收益':>12} {'年化':>12} {'最大回撤':>12} {'夏普':>8} {'胜率':>8}")
        print(f"  {'-' * 64}")
        for n, m in sorted(all_metrics.items()):
            print(f"  {'Top-'+str(n):<10} {m['total_return']*100:>+11.2f}% "
                  f"{m['annual_return']*100:>+11.2f}% "
                  f"{m['max_drawdown']*100:>11.2f}% "
                  f"{m['sharpe']:>8.2f} "
                  f"{m['win_rate']*100:>7.1f}%")

    # ---- 5. 汇总对比 ----
    if r_pure and r_enhanced:
        print(f"\n{'=' * 70}")
        print("纯小市值 vs 增强版 对比")
        print(f"{'=' * 70}")
        print(f"  {'指标':<16} {'纯小市值':>14} {'增强版':>14}")
        print(f"  {'-' * 46}")
        print(f"  {'总收益':<16} {m_pure['total_return']*100:>+13.2f}% {m_enhanced['total_return']*100:>+13.2f}%")
        print(f"  {'年化收益':<16} {m_pure['annual_return']*100:>+13.2f}% {m_enhanced['annual_return']*100:>+13.2f}%")
        print(f"  {'最大回撤':<16} {m_pure['max_drawdown']*100:>13.2f}% {m_enhanced['max_drawdown']*100:>13.2f}%")
        print(f"  {'夏普比率':<16} {m_pure['sharpe']:>14.2f} {m_enhanced['sharpe']:>14.2f}")
        print(f"  {'卡玛比率':<16} {m_pure['calmar']:>14.2f} {m_enhanced['calmar']:>14.2f}")
        print(f"  {'月度胜率':<16} {m_pure['win_rate']*100:>13.1f}% {m_enhanced['win_rate']*100:>13.1f}%")

    print("\n关键发现:")
    print("  - 小市值效应在A股长期有效, 但单独使用风险大")
    print("  - 纯小市值容易选到'垃圾股': 暴跌股、高波动股")
    print("  - 增强版: 动量过滤排除趋势恶化的小票, 波动率过滤排除极端波动")
    print("  - 市值+动量+波动率 = 最基础的多因子选股框架")
    print("  - 下一步: 将因子选股和中枢网格融合")
