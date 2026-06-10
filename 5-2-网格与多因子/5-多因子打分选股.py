# -*- coding: utf-8 -*-
"""
多因子打分选股策略 - 综合打分 + 月度调仓
==========================================

核心思路:
---------
脚本4 证明了因子的有效性 (IC 分析和分层回测确认因子能预测收益),
本脚本将多个因子组合成一个完整的选股策略。

选股流程 (每月末执行):
--------------
1. 对全市场所有股票计算 8 个技术因子 (TA-Lib)
2. 横截面排名打分 (每个因子排名归一化 0~1)
3. 按权重加权求和, 得到综合得分
4. 选出 Top-N 股票构建投资组合
5. 等权重配置, 持有至下月末
6. 下月重新打分调仓

回测方法说明:
-------------
因为是多标的轮动回测, 不能再用 Backtrader 的单标的回测框架。
改用自行实现的月度调仓回测框架:

  每月末:
    1. 计算因子得分
    2. 选出 Top-N 股票
    3. 等权重买入
  下月末:
    1. 卖出所有持仓
    2. 重新选股
    3. 再平衡

运行方式:
  python 5-多因子打分选股.py
"""
import numpy as np
import pandas as pd
import time
import os
from factor_engine import batch_calc_factors, score_stocks, select_top_stocks, print_factor_report
from db_config import execute_query, INITIAL_CASH


# ============================================================
# 批量数据加载 (复用脚本4的逻辑)
# ============================================================

def batch_load_daily(start_date, end_date, min_bars=120):
    """
    批量加载日K线, 返回 dict {code: DataFrame}。

    从 trade_stock_daily 表读取数据, 按股票代码分组,
    过滤掉K线数量不足的股票。

    参数:
        start_date: 开始日期
        end_date: 结束日期
        min_bars: 最少K线数 (默认120, 约半年数据)

    返回值:
        dict {stock_code: DataFrame}, DataFrame 含 open/high/low/close/volume
    """
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


# ============================================================
# 月度调仓回测
# ============================================================

def monthly_rebalance_backtest(all_data, top_n=10, initial_cash=None):
    """
    多因子月度调仓回测 (核心回测函数)。

    回测流程:
      每月末:
        1. 计算所有股票截至当日的因子值
        2. 使用 factor_engine.score_stocks() 综合打分
        3. 选出得分最高的 Top-N 只股票
        4. 等权重配置 (简单平均)
        5. 持有至下月末, 计算组合收益

    参数:
        all_data: dict {code: DataFrame}, 全市场K线数据
        top_n: 每期选股数量
        initial_cash: 初始资金 (默认使用 db_config.INITIAL_CASH)

    返回值:
        dict, 包含:
          - nav_log: 净值记录 [{date, nav}, ...]
          - rebalance_log: 调仓记录 [{date, stocks, return, nav}, ...]
          - final_nav: 最终净值

    注意:
      这是一个简化版回测, 未考虑:
        - 交易成本 (手续费/滑点)
        - 涨跌停无法买入/卖出的限制
        - 资金管理和仓位分配
    """
    cash = initial_cash or INITIAL_CASH

    # 取所有交易日期
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)

    # 找到每月最后一个交易日 (调仓日)
    rebalance_dates = []
    for i, d in enumerate(all_dates):
        if i + 1 < len(all_dates) and all_dates[i + 1].month != d.month:
            rebalance_dates.append(d)

    if len(rebalance_dates) < 2:
        print("  调仓期数不足")
        return None

    nav = cash
    nav_log = [{'date': rebalance_dates[0], 'nav': nav}]
    rebalance_log = []
    holding = {}

    for i in range(len(rebalance_dates) - 1):
        rb_date = rebalance_dates[i]       # 调仓日 (月末)
        next_rb = rebalance_dates[i + 1]   # 下个调仓日

        # 对截至 rb_date 的数据计算因子
        # 这是关键: 只用调仓日之前的数据, 避免未来信息泄露
        period_data = {}
        for code, df in all_data.items():
            sub = df[df.index <= rb_date]
            if len(sub) >= 60:
                period_data[code] = sub

        if len(period_data) < top_n * 2:
            nav_log.append({'date': next_rb, 'nav': nav})
            continue

        factor_df = batch_calc_factors(period_data, calc_date=rb_date)
        if len(factor_df) < top_n:
            nav_log.append({'date': next_rb, 'nav': nav})
            continue

        # 因子打分 + 选股
        scored, top_codes = select_top_stocks(factor_df, top_n=top_n)

        # 计算 rb_date ~ next_rb 的区间收益
        returns = []
        selected_names = []
        for code in top_codes:
            if code not in all_data:
                continue
            df = all_data[code]
            if rb_date not in df.index or next_rb not in df.index:
                continue
            c1 = float(df.loc[rb_date, 'close'])     # 调仓日收盘价 (买入价)
            c2 = float(df.loc[next_rb, 'close'])     # 下月调仓日收盘价 (卖出价)
            if c1 > 0:
                ret = c2 / c1 - 1
                returns.append(ret)
                selected_names.append(code)

        # 等权组合收益
        if returns:
            port_return = np.mean(returns)
            nav *= (1 + port_return)
        else:
            port_return = 0

        nav_log.append({'date': next_rb, 'nav': nav})
        rebalance_log.append({
            'date': rb_date,
            'stocks': selected_names,
            'return': port_return,
            'nav': nav,
        })

    return {
        'nav_log': nav_log,
        'rebalance_log': rebalance_log,
        'final_nav': nav,
    }


def calc_backtest_metrics(result, initial_cash=None):
    """
    计算月度调仓回测的绩效指标。

    与 Backtrader 版本的指标计算方式类似, 但处理的是月度收益序列。
    夏普比率用月收益 * 12 年化。

    参数:
        result: monthly_rebalance_backtest() 的返回值
        initial_cash: 初始资金

    返回值:
        dict, 包含 total_return / annual_return / max_drawdown / sharpe / calmar / win_rate / periods
    """
    cash = initial_cash or INITIAL_CASH
    nav_log = result['nav_log']

    if len(nav_log) < 2:
        return {}

    navs = [x['nav'] for x in nav_log]
    dates = [x['date'] for x in nav_log]

    total_return = navs[-1] / cash - 1
    trading_days = (dates[-1] - dates[0]).days
    years = trading_days / 365.25 if trading_days > 0 else 1
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else total_return

    # 最大回撤 (从峰值到谷值的最大跌幅)
    peak = navs[0]
    max_dd = 0
    for v in navs:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (peak - v) / peak
            max_dd = max(max_dd, dd)

    # 月度收益统计
    monthly_rets = [r['return'] for r in result['rebalance_log']]
    if monthly_rets:
        avg_month = np.mean(monthly_rets)
        std_month = np.std(monthly_rets)
        # 夏普 = (月均收益 * 12 - 无风险利率) / (月波动 * sqrt(12))
        sharpe = (avg_month * 12 - 0.02) / (std_month * np.sqrt(12)) if std_month > 0 else 0
        win_rate = sum(1 for r in monthly_rets if r > 0) / len(monthly_rets)
    else:
        sharpe = 0
        win_rate = 0

    calmar = annual_return / max_dd if max_dd > 0 else 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_dd,
        'sharpe': round(sharpe, 4),
        'calmar': round(calmar, 4),
        'win_rate': win_rate,
        'periods': len(monthly_rets),
        'years': round(years, 2),
    }


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    start_date = '2024-01-01'
    end_date = '2025-12-31'

    print("=" * 70)
    print("多因子打分选股策略 - 月度调仓")
    print("=" * 70)
    print("\n选股流程: 8因子打分 -> Top-N选股 -> 等权配置 -> 月度调仓")

    # ---- 1. 加载数据 ----
    print(f"\n[1] 加载数据...")
    t0 = time.time()
    all_data = batch_load_daily(start_date, end_date, min_bars=60)
    print(f"    {len(all_data)} 只标的, 耗时 {time.time()-t0:.1f}s")

    if len(all_data) < 20:
        print("  标的数量不足, 请确保数据库中有足够的股票数据")
        exit()

    # ---- 2. 展示当前截面因子 ----
    print(f"\n[2] 当前截面因子打分 (最新交易日)")
    factor_df = batch_calc_factors(all_data)
    scored, top_codes = select_top_stocks(factor_df, top_n=10)
    print_factor_report(scored, top_n=10, title='多因子打分排名 Top-10')

    # ---- 3. 不同 Top-N 的回测对比 ----
    print(f"\n{'=' * 70}")
    print(f"[3] 月度调仓回测")
    print(f"{'=' * 70}")

    top_n_list = [5, 10, 20]  # 对比不同选股数量的表现
    bt_results = {}

    for top_n in top_n_list:
        print(f"\n  --- Top-{top_n} 策略 ---")
        result = monthly_rebalance_backtest(all_data, top_n=top_n)
        if result:
            m = calc_backtest_metrics(result)
            bt_results[top_n] = m

            print(f"  总收益: {m['total_return']*100:+.2f}% | "
                  f"年化: {m['annual_return']*100:+.2f}% | "
                  f"最大回撤: {m['max_drawdown']*100:.2f}%")
            print(f"  夏普: {m['sharpe']:.2f} | "
                  f"卡玛: {m['calmar']:.2f} | "
                  f"月度胜率: {m['win_rate']*100:.1f}% | "
                  f"调仓: {m['periods']}期")

            # 打印调仓记录 (前3期)
            for rb in result['rebalance_log'][:3]:
                stocks_str = ', '.join(rb['stocks'][:5])
                if len(rb['stocks']) > 5:
                    stocks_str += f' ...+{len(rb["stocks"])-5}'
                print(f"    {rb['date'].strftime('%Y-%m-%d')} | "
                      f"收益={rb['return']*100:+.1f}% | "
                      f"持仓: {stocks_str}")
            if len(result['rebalance_log']) > 3:
                print(f"    ... 共 {len(result['rebalance_log'])} 期调仓")

    # ---- 4. 汇总对比 ----
    if bt_results:
        print(f"\n{'=' * 70}")
        print("Top-N 策略汇总对比")
        print(f"{'=' * 70}")
        print(f"  {'Top-N':<10} {'总收益':>12} {'年化':>12} {'最大回撤':>12} {'夏普':>8} {'胜率':>8}")
        print(f"  {'-' * 64}")
        for n, m in sorted(bt_results.items()):
            print(f"  {'Top-'+str(n):<10} {m['total_return']*100:>+11.2f}% "
                  f"{m['annual_return']*100:>+11.2f}% "
                  f"{m['max_drawdown']*100:>11.2f}% "
                  f"{m['sharpe']:>8.2f} "
                  f"{m['win_rate']*100:>7.1f}%")

    print("\n关键发现:")
    print("  - 多因子打分 = 综合多个维度的信息, 比单因子更稳健")
    print("  - Top-N 越小: 集中度越高, 弹性越大, 但波动也大")
    print("  - Top-N 越大: 分散化效果好, 收益更稳")
    print("  - 下一步: 结合小市值因子做轮动策略")
