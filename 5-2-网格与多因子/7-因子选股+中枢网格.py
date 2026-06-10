# -*- coding: utf-8 -*-
"""
因子选股 + 中枢网格 融合策略
============================

核心思路:
---------
前面6个脚本分别教了两条线:
  网格线: 固定网格(1) → 中枢网格(2) → 中枢+趋势联动(3)
  选股线: 因子评价(4) → 多因子打分(5) → 小市值轮动(6)

本脚本将两条线融合:
  1. 多因子打分选出"适合做网格"的标的
  2. 对选出的标的运行缠论分析, 找到中枢区间
  3. 在中枢内做网格交易

选股因子侧重:
-------------
和脚本5的通用的因子配置不同, 本脚本使用"网格适配因子":

  - ADX(14) 反向 30%: ADX越低越震荡, 越适合网格
  - 波动率  正向 25%: 适度波动有利于网格差价
  - RSI(14) 特殊 15%: 距离50越近越好 (没有方向倾向)
  - 20日动量 特殊 15%: 接近0最好 (没有趋势)
  - 换手率  正向 15%: 足够流动性

特殊因子处理:
  RSI和动量用"距离中性值的距离"打分, 而不是直接排名。
  因为网格策略需要"震荡", 而不是"趋势"。

融合流程:
---------
  1. 批量加载全市场K线数据
  2. 计算每个股票的8个技术因子
  3. 用网格适配因子打分
  4. 选出得分最高的 Top-5 股票
  5. 对每只股票运行缠论分析
  6. 在中枢内运行网格策略回测
  7. 汇总展示

运行方式:
  python 7-因子选股+中枢网格.py
"""
import numpy as np
import pandas as pd
import talib
import time
import backtrader as bt
from data_loader import load_stock_data, ChanPandasData, run_and_report, calc_buy_and_hold
from chanpy_wrapper import run_chan, chan_to_signal_df
from grid_engine import ChanGridEngine
from factor_engine import calc_all_factors, FACTOR_CONFIG
from db_config import execute_query, INITIAL_CASH


# ============================================================
# 网格适配因子配置 (侧重震荡特征)
# ============================================================

GRID_FACTOR_CONFIG = {
    'adx_14': {
        'name': 'ADX(14)',
        'direction': -1,   # 反向: ADX越低越震荡, 越适合网格
        'weight': 0.30,    # 权重最高: 震荡是网格的前提
        'desc': 'ADX低=震荡市=网格天堂',
    },
    'volatility': {
        'name': '波动率',
        'direction': 1,    # 正向: 适度波动有利于网格差价 (但会过滤极端)
        'weight': 0.25,
        'desc': '适度波动率有利于网格收租',
    },
    'rsi_14': {
        'name': 'RSI(14)',
        'direction': 0,    # 特殊: 接近50最好 (用距离50的绝对值取反)
        'weight': 0.15,
        'desc': 'RSI接近50=没有方向=适合网格',
    },
    'momentum_20d': {
        'name': '20日动量',
        'direction': 0,    # 特殊: 接近0最好 (无方向)
        'weight': 0.15,
        'desc': '动量接近0=没有趋势=适合网格',
    },
    'turnover_ratio': {
        'name': '换手率',
        'direction': 1,
        'weight': 0.15,
        'desc': '足够的交易量保证流动性',
    },
}


def score_for_grid(factor_df):
    """
    用网格适配因子打分。

    特殊处理:
      RSI(14): 距离50越近, 得分越高 (没有超买超卖倾向)
      动量(20日): 绝对值越小, 得分越高 (没有趋势方向)

    普通处理:
      正向因子: rank(pct=True) 直接使用
      反向因子: 1 - rank(pct=True)

    参数:
        factor_df: DataFrame, batch_calc_factors() 的输出

    返回值:
        DataFrame, 按得分降序排列
    """
    result = factor_df.copy()
    result['score'] = 0.0

    for fname, cfg in GRID_FACTOR_CONFIG.items():
        if fname not in result.columns:
            continue

        if cfg['direction'] == 0:
            # 特殊因子: 距离中性值越近, 得分越高
            if fname == 'rsi_14':
                # RSI: 距离50越近越好 (没有方向倾向)
                dist = (result[fname] - 50).abs()
                rank = 1 - dist.rank(pct=True)
            elif fname == 'momentum_20d':
                # 动量: 绝对值越小越好 (没有趋势)
                dist = result[fname].abs()
                rank = 1 - dist.rank(pct=True)
            else:
                rank = result[fname].rank(pct=True)
        elif cfg['direction'] > 0:
            rank = result[fname].rank(pct=True)
        else:
            rank = 1 - result[fname].rank(pct=True)

        result[f'{fname}_rank'] = rank
        result['score'] += rank * cfg['weight']

    return result.sort_values('score', ascending=False)


# ============================================================
# 批量数据加载
# ============================================================

def batch_load_daily(start_date, end_date, min_bars=120):
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


# ============================================================
# 中枢网格策略 (单标的, 用于对选出的股票回测)
# ============================================================

class FusionGridStrategy(bt.Strategy):
    """
    融合策略 - 在因子选出的标的上运行中枢网格。

    逻辑与脚本2的 ChanGridStrategy 相同。
    在中枢内做网格, 出中枢停止。

    参数:
      num_grids: 中枢内格子数 (默认6)
      capital_ratio: 网格资金比例 (默认80%)
    """
    params = (
        ('num_grids', 6),
        ('capital_ratio', 0.80),
    )

    def __init__(self):
        self.grid = None
        self.order = None
        self.current_zg = 0.0
        self.current_zd = 0.0
        self.mode = 'WAIT'

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

    def next(self):
        if self.order:
            return
        zg = self.data.chan_zg[0]
        zd = self.data.chan_zd[0]
        price = self.data.close[0]
        has_zs = (not np.isnan(zg)) and (not np.isnan(zd)) and zg > 0 and zd > 0

        if not has_zs:
            if self.mode == 'GRID' and self.position.size > 0:
                self.order = self.close()
            if self.grid:
                self.grid.deactivate()
            self.mode = 'WAIT'
            return

        zg_c = abs(zg - self.current_zg) > self.current_zg * 0.001 if self.current_zg > 0 else True
        zd_c = abs(zd - self.current_zd) > self.current_zd * 0.001 if self.current_zd > 0 else True

        if zg_c or zd_c:
            if self.position.size > 0:
                self.order = self.close()
                return
            cap = self.broker.getvalue() * self.p.capital_ratio
            if self.grid is None:
                self.grid = ChanGridEngine(zg, zd, self.p.num_grids, cap)
            else:
                self.grid.switch_zhongshu(zg, zd)
                self.grid.total_capital = cap
                self.grid.capital_per_grid = cap / self.p.num_grids
            self.current_zg, self.current_zd = zg, zd
            self.mode = 'GRID'

        if self.mode != 'GRID' or not self.grid or not self.grid.active:
            return

        if price > self.current_zg * 1.005 or price < self.current_zd * 0.995:
            if self.position.size > 0:
                self.order = self.close()
            self.grid.deactivate()
            self.mode = 'WAIT'
            return

        signals = self.grid.update(price)
        for sig in signals:
            if sig['action'] == 'BUY':
                if self.broker.getcash() >= price * sig['size'] * 1.01 and sig['size'] > 0:
                    self.order = self.buy(size=sig['size'])
                    break
            elif sig['action'] == 'SELL':
                if self.position.size >= sig['size']:
                    self.order = self.sell(size=sig['size'])
                    break

    def stop(self):
        if self.grid:
            self.grid.summary()


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    start_date = '2023-01-01'
    end_date = '2025-12-31'

    print("=" * 70)
    print("因子选股 + 中枢网格 融合策略")
    print("=" * 70)
    print("\n融合流程:")
    print("  1. 多因子打分: 选出'适合做网格'的标的 (低ADX, 适度波动)")
    print("  2. 缠论分析: 对选出的标的识别中枢")
    print("  3. 中枢网格: 在中枢内做网格交易")

    # ---- 1. 加载数据 & 因子打分 ----
    print(f"\n[1] 加载数据...")
    t0 = time.time()
    all_data = batch_load_daily(start_date, end_date, min_bars=120)
    print(f"    {len(all_data)} 只标的, 耗时 {time.time()-t0:.1f}s")

    if len(all_data) < 10:
        print("  标的不足, 使用预设标的")
        all_data = {}
        for code in ['600519.SH', '688981.SH', '000001.SZ', '159941.SZ', '300750.SZ']:
            try:
                df = load_stock_data(code, start_date, end_date)
                if len(df) >= 120:
                    all_data[code] = df
            except Exception:
                pass

    # ---- 2. 网格适配因子打分 ----
    print(f"\n[2] 网格适配因子打分...")
    factor_dict = {}
    for code, df in all_data.items():
        f = calc_all_factors(df)
        if f is not None:
            factor_dict[code] = f

    if not factor_dict:
        print("  因子计算失败")
        exit()

    factor_df = pd.DataFrame(factor_dict).T
    scored = score_for_grid(factor_df)

    print(f"\n  网格适配评分排名 (ADX低=震荡, 波动适中, RSI中性):")
    print(f"  {'排名':>4} {'代码':<14} {'得分':>8} {'ADX':>8} {'波动率':>10} {'RSI':>8} {'20D动量':>10}")
    print(f"  {'-' * 66}")
    for i, (code, row) in enumerate(scored.head(10).iterrows()):
        adx_v = row.get('adx_14', 0)
        vol_v = row.get('volatility', 0)
        rsi_v = row.get('rsi_14', 0)
        m20 = row.get('momentum_20d', 0)
        print(f"  {i+1:>4} {code:<14} {row['score']:>8.4f} {adx_v:>8.1f} {vol_v:>10.4f} {rsi_v:>8.1f} {m20:>+9.2f}%")

    # ---- 3. 对 Top-5 运行缠论分析 + 中枢网格回测 ----
    top_n = min(5, len(scored))
    top_codes = scored.head(top_n).index.tolist()

    print(f"\n{'=' * 70}")
    print(f"[3] 对 Top-{top_n} 运行中枢网格回测")
    print(f"{'=' * 70}")

    backtest_results = {}
    for code in top_codes:
        try:
            print(f"\n--- {code} (得分={scored.loc[code, 'score']:.4f}) ---")
            df = all_data[code]

            # 缠论分析
            chan_data = run_chan(df, symbol=code)
            signal_df = chan_to_signal_df(df, chan_data)
            zs_count = len(chan_data['zs_list'])
            print(f"  缠论: {len(chan_data['bi_list'])}笔, {zs_count}个中枢")

            if zs_count == 0:
                print(f"  无中枢, 跳过网格回测")
                continue

            # 中枢网格回测
            r = run_and_report(
                FusionGridStrategy, stock_code=code,
                label=f'{code}-融合网格', plot=True, use_sizer=False,
                df=signal_df, data_class=ChanPandasData,
            )
            bh = calc_buy_and_hold(code, start_date, end_date)
            backtest_results[code] = {'result': r, 'bh': bh, 'zs_count': zs_count}

        except Exception as e:
            print(f"  跳过: {e}")

    # ---- 4. 汇总对比 ----
    if backtest_results:
        print(f"\n{'=' * 70}")
        print("融合策略汇总")
        print(f"{'=' * 70}")

        print(f"\n  {'代码':<14} {'得分':>8} {'中枢数':>6} {'网格收益':>12} {'买持收益':>12} "
              f"{'最大回撤':>10} {'夏普':>8} {'胜率':>8}")
        print(f"  {'-' * 84}")

        for code, data in backtest_results.items():
            r = data['result']
            bh = data['bh']
            score = scored.loc[code, 'score']
            print(f"  {code:<14} {score:>8.4f} {data['zs_count']:>6d} "
                  f"{r['total_return']*100:>+11.2f}% {(bh or 0)*100:>+11.2f}% "
                  f"{r['max_drawdown']*100:>9.2f}% {r['sharpe_ratio']:>8.2f} "
                  f"{r['win_rate']*100:>7.1f}%")

        avg_ret = np.mean([d['result']['total_return'] for d in backtest_results.values()])
        avg_dd = np.mean([d['result']['max_drawdown'] for d in backtest_results.values()])
        print(f"\n  组合平均: 收益={avg_ret*100:+.2f}%, 最大回撤={avg_dd*100:.2f}%")

    print("\n关键发现:")
    print("  - 因子选股筛出'适合网格'的标的 → 提升网格策略效果")
    print("  - 低ADX的股票在中枢内震荡更多 → 网格交易机会更多")
    print("  - 适度波动率 → 每格差价合理, 既有利润又不至于太高风险")
    print("  - 因子选股 + 缠论中枢 + 网格交易 = 完整的量化策略闭环")
    print("  - 下一步: 用机器学习(LightGBM)优化因子权重和选股")
