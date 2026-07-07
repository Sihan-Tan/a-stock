# -*- coding: utf-8 -*-
# 21-CASE-C 多因子: 分层回测 + IC 时序
"""
LayeredBacktest -- 分层回测器

什么是分层回测?
    1. 在每个调仓日, 把所有股票按 alpha 分成 N 层 (默认 5 层)
    2. 第 1 层 = alpha 最低的 20%, 第 5 层 = alpha 最高的 20%
    3. 每层等权持有到下个调仓日, 算累计收益
    4. 看 5 层的收益曲线: 如果"层数越高收益越高", 因子就有效
    5. 多空收益 = 第 5 层 - 第 1 层 = 因子的纯 alpha

关键指标:
    - 多空累计收益: 越大因子越强
    - 多空波动: 越小越稳
    - IC 均值: 横截面相关系数, 越大因子越准
    - IR (IC/IC.std): IC 是否稳定, > 0.5 优秀

本模块产出:
    - 5 层累计收益曲线 (DataFrame)
    - IC 时序 (Series)
    - 综合指标表 (DataFrame)

"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

# 让 python 直接跑也能 import (而不是必须 -m multi_factor.layered_backtest)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from factor_lib import calc_factors_for_one
from preprocessor import preprocess_factors
from synthesizer import calc_ic, calc_ir, ic_weighted_synthesis


def _calc_factor_snapshot(prices_panel: Dict[str, pd.DataFrame],
                          end_idx: int) -> pd.DataFrame:
    """
    在某个时点 (end_idx) 截取每只股票的历史 K 线, 算因子矩阵
    
    prices_panel: {stock_code: DataFrame, 含 close/volume/amount}
    end_idx:      截至第几行 (不含)
    """
    rows = {}
    for code, df in prices_panel.items():
        if len(df) <= end_idx + 1 or end_idx < 130:
            continue
        sub = df.iloc[: end_idx + 1]
        f = calc_factors_for_one(sub)
        if f:
            rows[code] = f
    return pd.DataFrame.from_dict(rows, orient="index")


def _next_period_returns(prices_panel: Dict[str, pd.DataFrame],
                         end_idx: int, period: int) -> pd.Series:
    """计算 [end_idx -> end_idx+period] 的累计收益"""
    rets = {}
    for code, df in prices_panel.items():
        if end_idx + period >= len(df):
            continue
        p_now = df["close"].iloc[end_idx]
        p_future = df["close"].iloc[end_idx + period]
        if p_now > 0:
            rets[code] = p_future / p_now - 1.0
    return pd.Series(rets)


def run_single_factor_ic(prices_panel: Dict[str, pd.DataFrame],
                         industry_map: Dict[str, str],
                         rebal_period: int = 21,
                         min_warmup: int = 130) -> pd.DataFrame:
    """
    单因子 IC 测试: 对每个因子分别测 IC 时序, 看到底哪个因子真有效

    返回: DataFrame, index=因子名, columns=[ic_mean, ic_std, ic_ir, ic_positive_ratio]
    """
    first_code = next(iter(prices_panel))
    n = len(prices_panel[first_code].index)
    rebal_dates = list(range(min_warmup, n - rebal_period, rebal_period))

    # {factor_name: [ic_t0, ic_t1, ...]}
    ic_records = {}

    for end_idx in rebal_dates:
        factor_df = _calc_factor_snapshot(prices_panel, end_idx)
        if len(factor_df) < 30:
            continue

        # 预处理 (含中性化)
        factor_processed = preprocess_factors(factor_df, industry_map=industry_map,
                                              neutralize=True)

        # 拿下期收益
        future_ret = _next_period_returns(prices_panel, end_idx, rebal_period)

        # 每个因子单独算 IC
        for col in factor_processed.columns:
            ic = calc_ic(factor_processed[col], future_ret)
            ic_records.setdefault(col, []).append(ic)

    rows = {}
    for col, ic_list in ic_records.items():
        ic_arr = np.array([x for x in ic_list if not np.isnan(x)])
        if len(ic_arr) == 0:
            continue
        rows[col] = {
            "ic_mean":            float(ic_arr.mean()),
            "ic_std":             float(ic_arr.std(ddof=1)) if len(ic_arr) > 1 else 0.0,
            "ic_ir":              (float(ic_arr.mean() / ic_arr.std(ddof=1))
                                   if len(ic_arr) > 1 and ic_arr.std(ddof=1) > 0 else 0.0),
            "ic_positive_ratio":  float((ic_arr > 0).mean()),
            "samples":            len(ic_arr),
        }
    return pd.DataFrame.from_dict(rows, orient="index").sort_values("ic_ir", ascending=False)


def run_layered_backtest(prices_panel: Dict[str, pd.DataFrame],
                         industry_map: Dict[str, str],
                         rebal_period: int = 21,
                         n_layers: int = 5,
                         min_warmup: int = 130,
                         weight_method: str = "equal",
                         ic_lookback: int = 6,
                         top_n_list: List[int] = None,
                         benchmark_returns: pd.Series = None) -> Dict:
    """
    分层回测主流程

    参数:
        prices_panel:      {stock_code: DataFrame (含 close/volume/amount, 升序)}
        industry_map:      {stock_code: industry}
        rebal_period:      调仓周期 (日, 默认 21 = 月度)
        n_layers:          分多少层 (默认 5)
        min_warmup:        前 N 个交易日不参与 (因子要 warm up)
        weight_method:     "equal" -- 等权合成
                           "ic_weighted" -- 用前 ic_lookback 期单因子 IC 滚动均值作为权重
                           (这是 walk-forward, 第 1~ic_lookback 期没权重就退化为等权)
        ic_lookback:       IC 加权时用前几期 IC 估权重 (默认 6 = 半年)
        top_n_list:        Top-N 集中度回测的 N 列表, 默认 [5, 10, 20]
                           对每个 N 输出"按 alpha 排名前 N 等权持有"的累计收益
        benchmark_returns: 基准 (如沪深 300) 在每个调仓日的下期收益, index=date_str
                           为 None 时不输出基准

    返回:
        {
            "layer_returns":  DataFrame,    # 每个调仓日各层的下期收益
            "layer_cumret":   DataFrame,    # 各层累计收益
            "ic_series":      Series,       # 每个调仓日的 IC
            "topn_returns":   DataFrame,    # 每个调仓日 Top-N 组合下期收益
            "topn_cumret":    DataFrame,    # Top-N 累计收益曲线
            "single_factor_ic": DataFrame,  # 单因子 IC 时序 (用于 IC 加权时的 walk-forward)
            "metrics":        dict,         # 综合指标
        }
    """
    if top_n_list is None:
        top_n_list = [5, 10, 20]

    # 用第一只股票的索引作为时间基准
    first_code = next(iter(prices_panel))
    base_index = prices_panel[first_code].index
    n = len(base_index)

    if n < min_warmup + rebal_period * 2:
        raise ValueError(f"数据不足, 至少需要 {min_warmup + rebal_period * 2} 行")

    # 调仓日: 每隔 rebal_period 一个
    rebal_dates = list(range(min_warmup, n - rebal_period, rebal_period))
    print(f"[BACKTEST] 数据 {n} 行, 调仓 {len(rebal_dates)} 次, 每次间隔 {rebal_period} 日, "
          f"合成方式={weight_method}")

    layer_returns = []     # 每行: [date, layer1_ret, ..., layerN_ret]
    ic_records = []
    topn_records = []      # 每行: {date, top5, top10, top20, ...}
    single_ic_records = {}  # {factor_name: [ic_t0, ic_t1, ...]} 每期单因子 IC, 用于 IC 加权

    for i, end_idx in enumerate(rebal_dates):
        # 1) 算因子矩阵
        factor_df = _calc_factor_snapshot(prices_panel, end_idx)
        if len(factor_df) < n_layers * 5:
            continue

        # 2) 预处理 (去极值 + Z-score + 行业中性化)
        factor_processed = preprocess_factors(factor_df, industry_map=industry_map,
                                              neutralize=True)

        # 3) 拿下期收益 (放在合成 alpha 之前, 因为 IC 加权要先记录单因子 IC)
        future_ret = _next_period_returns(prices_panel, end_idx, rebal_period)

        # 3.1) 先算每个因子的当期 IC, 累积起来供 IC 加权用 (walk-forward)
        single_ic_now = {}
        for col in factor_processed.columns:
            ic_one = calc_ic(factor_processed[col], future_ret)
            single_ic_records.setdefault(col, []).append(ic_one)
            single_ic_now[col] = ic_one

        # 3.2) 合成 alpha
        if weight_method == "ic_weighted":
            # 用过去 ic_lookback 期的 IC 滚动均值作为权重 (严格 walk-forward, 不含当期)
            past_ic = {}
            for col, hist in single_ic_records.items():
                # 当期已经追加, 排除掉, 只用前 ic_lookback 期
                hist_no_now = hist[:-1]
                if len(hist_no_now) >= 1:
                    recent = [x for x in hist_no_now[-ic_lookback:] if not np.isnan(x)]
                    if recent:
                        past_ic[col] = float(np.mean(recent))
            if past_ic and any(v != 0 for v in past_ic.values()):
                alpha = ic_weighted_synthesis(factor_processed, past_ic).dropna()
            else:
                # 前 ic_lookback 期没有历史 IC, 退化为等权 (避免冷启动空跑)
                alpha = factor_processed.mean(axis=1).dropna()
        else:
            alpha = factor_processed.mean(axis=1).dropna()
        if len(alpha) < n_layers * 5:
            continue

        # 4) 算 IC (合成后的 alpha vs 下期收益)
        ic = calc_ic(alpha, future_ret)
        date_str = str(base_index[end_idx])[:10]
        ic_records.append({"date": date_str, "ic": ic})

        # 5) 分 N 层, 各层等权持有的下期收益
        common = alpha.index.intersection(future_ret.index)
        if len(common) < n_layers * 5:
            continue
        df = pd.DataFrame({"alpha": alpha.loc[common],
                           "ret": future_ret.loc[common]})
        df["layer"] = pd.qcut(df["alpha"], n_layers,
                              labels=range(1, n_layers + 1),
                              duplicates="drop")
        layer_means = df.groupby("layer", observed=True)["ret"].mean()
        layer_dict = {f"L{i}": layer_means.get(i, 0) for i in range(1, n_layers + 1)}
        layer_dict["date"] = date_str
        layer_dict["long_short"] = layer_means.get(n_layers, 0) - layer_means.get(1, 0)
        layer_returns.append(layer_dict)

        # 6) Top-N 集中度回测 -- 按 alpha 排名取前 N 等权
        topn_dict = {"date": date_str}
        df_sorted = df.sort_values("alpha", ascending=False)
        for k in top_n_list:
            if k <= len(df_sorted):
                topn_dict[f"Top{k}"] = df_sorted["ret"].iloc[:k].mean()
            else:
                topn_dict[f"Top{k}"] = np.nan
        topn_records.append(topn_dict)

        if (i + 1) % 5 == 0:
            print(f"  ... 进度 {i+1}/{len(rebal_dates)}: date={date_str}, IC={ic:.3f}")

    # 汇总
    layer_returns_df = pd.DataFrame(layer_returns).set_index("date")
    layer_cumret = (1 + layer_returns_df.drop(columns=["long_short"])).cumprod()
    ic_series = pd.DataFrame(ic_records).set_index("date")["ic"]

    topn_returns_df = pd.DataFrame(topn_records).set_index("date") if topn_records else pd.DataFrame()
    topn_cumret = (1 + topn_returns_df).cumprod() if not topn_returns_df.empty else pd.DataFrame()

    # 单因子 IC 时序 (用于复盘 IC 加权依据)
    single_ic_df = pd.DataFrame(single_ic_records, index=[r["date"] for r in ic_records])

    # 指标
    metrics = {
        "weight_method":        weight_method,
        "rebal_count":          len(layer_returns),
        "ic_mean":              ic_series.mean(),
        "ic_std":               ic_series.std(),
        "ic_ir":                ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
        "ic_positive_ratio":    (ic_series > 0).mean(),
        "long_short_total_ret": (1 + layer_returns_df["long_short"]).prod() - 1,
        "long_short_avg_ret":   layer_returns_df["long_short"].mean(),
        "L5_total_ret":         layer_cumret["L5"].iloc[-1] - 1,
        "L1_total_ret":         layer_cumret["L1"].iloc[-1] - 1,
        "L5_minus_L1_total":    layer_cumret["L5"].iloc[-1] - layer_cumret["L1"].iloc[-1],
    }

    # Top-N 组合的累计收益指标
    if not topn_cumret.empty:
        for col in topn_cumret.columns:
            metrics[f"{col}_total_ret"] = topn_cumret[col].iloc[-1] - 1
            metrics[f"{col}_avg_ret"]   = topn_returns_df[col].mean()
            # 简单回撤 = 累计净值的最大回撤
            cum = topn_cumret[col]
            metrics[f"{col}_max_dd"] = float((cum / cum.cummax() - 1).min())

    # 基准对照
    if benchmark_returns is not None and not benchmark_returns.empty:
        bench_aligned = benchmark_returns.reindex(layer_returns_df.index)
        metrics["bench_total_ret"] = (1 + bench_aligned.fillna(0)).prod() - 1
        metrics["bench_avg_ret"]   = bench_aligned.mean()
        for col in topn_cumret.columns:
            metrics[f"{col}_excess_vs_bench"] = (
                metrics[f"{col}_total_ret"] - metrics["bench_total_ret"]
            )

    return {
        "layer_returns":     layer_returns_df,
        "layer_cumret":      layer_cumret,
        "ic_series":         ic_series,
        "topn_returns":      topn_returns_df,
        "topn_cumret":       topn_cumret,
        "single_factor_ic":  single_ic_df,
        "metrics":           metrics,
    }


# ============================================================
# 基准: 沪深 300 在每个调仓日的下期收益
# ============================================================

def calc_benchmark_returns(prices_panel_or_index_df,
                           rebal_period: int = 21,
                           min_warmup: int = 130) -> pd.Series:
    """
    用同一调仓周期, 算基准 (一般是沪深 300 指数) 的下期收益序列

    参数:
        prices_panel_or_index_df:
            - DataFrame: 已经是单一基准指数的日线 (有 close 列, index 是日期)
            - dict {code: df}: 用全样本等权当 "穷人版基准" (退而求其次)
        rebal_period:  跟主回测一致
        min_warmup:    跟主回测一致

    返回: Series, index=调仓日字符串 yyyy-mm-dd, value=下期累计收益
    """
    if isinstance(prices_panel_or_index_df, pd.DataFrame):
        # 单一基准指数
        df = prices_panel_or_index_df
        n = len(df)
        rebal_dates = list(range(min_warmup, n - rebal_period, rebal_period))
        rows = {}
        for end_idx in rebal_dates:
            p_now = df["close"].iloc[end_idx]
            p_future = df["close"].iloc[end_idx + rebal_period]
            if p_now > 0:
                date_str = str(df.index[end_idx])[:10]
                rows[date_str] = p_future / p_now - 1.0
        return pd.Series(rows, name="bench")
    else:
        # dict 模式: 用样本等权作为穷人版基准
        panel = prices_panel_or_index_df
        first_code = next(iter(panel))
        base_index = panel[first_code].index
        n = len(base_index)
        rebal_dates = list(range(min_warmup, n - rebal_period, rebal_period))
        rows = {}
        for end_idx in rebal_dates:
            future_ret = _next_period_returns(panel, end_idx, rebal_period)
            if len(future_ret) > 0:
                date_str = str(base_index[end_idx])[:10]
                rows[date_str] = float(future_ret.mean())
        return pd.Series(rows, name="bench")


# ============================================================
# CLI
# ============================================================

def _try_load_benchmark(rebal_period: int, lookback: int, min_warmup: int) -> pd.Series:
    """
    尝试拉沪深 300 指数 (000300.SH) 的下期收益序列, 拿不到就返回空 Series
    """
    try:
        from xtquant import xtdata
        xtdata.connect()
        try:
            xtdata.download_history_data("000300.SH", period="1d",
                                         start_time="20230101")
        except Exception:
            pass
        data = xtdata.get_market_data_ex(
            field_list=["close"], stock_list=["000300.SH"], period="1d",
            count=lookback, dividend_type="none",
        )
        df = data.get("000300.SH")
        if df is None or len(df) < min_warmup + rebal_period * 2:
            return pd.Series(dtype=float)
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        return calc_benchmark_returns(df, rebal_period=rebal_period,
                                      min_warmup=min_warmup)
    except Exception as e:
        print(f"  [WARN] 沪深 300 基准拉取失败: {e}")
        return pd.Series(dtype=float)


def _print_topn_summary(label: str, result: Dict, bench_total: float = None):
    """打印 Top-N 集中度对比表"""
    m = result["metrics"]
    topn_cols = [c for c in result["topn_cumret"].columns]
    if not topn_cols:
        return
    print(f"\n[{label} -- Top-N 集中度对比]")
    print(f"  {'组合':<8} {'累计收益':>10} {'每期均值':>10} {'最大回撤':>10}"
          + ("  " + "超额(vs沪深300)" if bench_total is not None else ""))
    for col in topn_cols:
        line = (f"  {col:<8} "
                f"{m.get(f'{col}_total_ret', 0):>+9.2%} "
                f"{m.get(f'{col}_avg_ret', 0):>+9.2%} "
                f"{m.get(f'{col}_max_dd', 0):>+9.2%}")
        if bench_total is not None:
            excess = m.get(f"{col}_total_ret", 0) - bench_total
            line += f"     {excess:>+8.2%}"
        print(line)


def main():
    """端到端 demo: 拉沪深 300 -> 单因子IC -> 等权回测 -> IC加权回测 -> Top-N -> 基准对照"""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="分层回测 demo")
    parser.add_argument("--max-stocks", type=int, default=80,
                        help="只用前 N 只股票, 默认 80 (节省时间, 全量 300 太慢)")
    parser.add_argument("--lookback", type=int, default=400,
                        help="拉多少日 K 线, 默认 400")
    parser.add_argument("--rebal", type=int, default=21,
                        help="调仓周期 (日), 默认 21 = 月度")
    parser.add_argument("--ic-lookback", type=int, default=6,
                        help="IC 加权用前几期 IC 估权重, 默认 6")
    parser.add_argument("--no-benchmark", action="store_true",
                        help="不跑沪深 300 基准对照")
    args = parser.parse_args()

    # 让 .py 文件能直接 python 跑 (不通过 -m)
    THIS_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(THIS_DIR.parent))

    from stock_pool import get_csi300, filter_tradable, get_industry_map
    from xtquant import xtdata
    xtdata.connect()

    print(f"\n{'='*70}")
    print(f"  21 章 CASE-C: 多因子分层回测 + IC 加权 + Top-N 集中度")
    print(f"{'='*70}\n")

    print(f"[1/5] 拉沪深 300 + 过滤 ...")
    codes = get_csi300()
    codes = filter_tradable(codes)[:args.max_stocks]
    print(f"  最终参与回测: {len(codes)} 只")

    print(f"[2/5] 拉行业映射 ...")
    industry_map = get_industry_map(codes)
    industry_count = pd.Series(industry_map).value_counts().head(5)
    print(f"  Top 5 行业分布: {industry_count.to_dict()}")

    print(f"[3/5] 拉 K 线 ({args.lookback} 日) ...")
    for code in codes:
        try:
            xtdata.download_history_data(code, period="1d", start_time="20230101")
        except Exception:
            pass

    data = xtdata.get_market_data_ex(
        field_list=["close", "volume", "amount"],
        stock_list=codes, period="1d", count=args.lookback,
        dividend_type="back",
    )
    prices_panel = {}
    for code in codes:
        df = data.get(code)
        if df is not None and len(df) > 200:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            prices_panel[code] = df

    print(f"  实际有效股票: {len(prices_panel)}")

    # 基准: 沪深 300 指数下期收益
    bench_returns = pd.Series(dtype=float)
    if not args.no_benchmark:
        print(f"\n[3.5/5] 拉沪深 300 指数作为基准 ...")
        bench_returns = _try_load_benchmark(args.rebal, args.lookback, min_warmup=130)
        if not bench_returns.empty:
            bench_total = (1 + bench_returns).prod() - 1
            print(f"  沪深 300 同期累计收益: {bench_total:+.2%} "
                  f"({len(bench_returns)} 个调仓周期)")

    print(f"\n[4/5] 单因子 IC 诊断 (诊断每个因子的有效性) ...")
    single_ic = run_single_factor_ic(prices_panel, industry_map,
                                     rebal_period=args.rebal)
    print(f"\n[单因子 IC 排名 (按 IR 降序)]")
    print(single_ic.round(4).to_string())
    print(f"\n  解读: IR > 0.5 = 优秀因子 | IR > 0.2 = 可用 | < 0.1 = 噪音")

    print(f"\n[5a] 等权合成 + 分层回测 ...")
    result_eq = run_layered_backtest(prices_panel, industry_map,
                                     rebal_period=args.rebal, n_layers=5,
                                     weight_method="equal",
                                     top_n_list=[5, 10, 20],
                                     benchmark_returns=bench_returns
                                     if not bench_returns.empty else None)

    print(f"\n[5b] IC 加权合成 + 分层回测 (walk-forward, ic_lookback={args.ic_lookback}) ...")
    result_ic = run_layered_backtest(prices_panel, industry_map,
                                     rebal_period=args.rebal, n_layers=5,
                                     weight_method="ic_weighted",
                                     ic_lookback=args.ic_lookback,
                                     top_n_list=[5, 10, 20],
                                     benchmark_returns=bench_returns
                                     if not bench_returns.empty else None)

    # ============== 等权 vs IC 加权 对比 ==============
    print(f"\n{'='*70}")
    print(f"  等权 vs IC 加权 -- 同样的因子, 不同的合成方式")
    print(f"{'='*70}\n")

    m_eq, m_ic = result_eq["metrics"], result_ic["metrics"]
    bench_total = m_eq.get("bench_total_ret")

    print(f"  {'指标':<22} {'等权合成':>12} {'IC 加权':>12}  {'变化':>10}")
    rows = [
        ("IC 均值",            "ic_mean",              "{:+.4f}"),
        ("IC IR",              "ic_ir",                "{:+.4f}"),
        ("IC 正比例",          "ic_positive_ratio",    "{:.1%}"),
        ("L5 累计收益",        "L5_total_ret",         "{:+.2%}"),
        ("L1 累计收益",        "L1_total_ret",         "{:+.2%}"),
        ("L5-L1 多空收益",     "L5_minus_L1_total",    "{:+.2%}"),
        ("Top10 累计收益",     "Top10_total_ret",      "{:+.2%}"),
        ("Top10 最大回撤",     "Top10_max_dd",         "{:+.2%}"),
    ]
    for label, key, fmt in rows:
        v_eq = m_eq.get(key, 0)
        v_ic = m_ic.get(key, 0)
        delta = v_ic - v_eq
        print(f"  {label:<22} {fmt.format(v_eq):>12} {fmt.format(v_ic):>12}  "
              f"{('{:+.4f}' if 'IC' in label and '%' not in fmt else '{:+.2%}').format(delta):>10}")

    # Top-N 集中度对比 (两种合成各列一次)
    _print_topn_summary("等权合成", result_eq, bench_total)
    _print_topn_summary("IC 加权",  result_ic, bench_total)

    print(f"\n{'='*70}")
    print(f"  各层累计收益曲线 -- IC 加权方案 (最后 8 期)")
    print(f"{'='*70}")
    print(result_ic["layer_cumret"].tail(8).round(4).to_string())

    print(f"\n[一句话洞察]")
    if m_eq["L5_minus_L1_total"] < 0 and m_ic["L5_minus_L1_total"] > m_eq["L5_minus_L1_total"]:
        print(f"  等权合成 L5-L1={m_eq['L5_minus_L1_total']:+.2%} (失效),")
        print(f"  IC 加权后 L5-L1={m_ic['L5_minus_L1_total']:+.2%}, 改善 {m_ic['L5_minus_L1_total'] - m_eq['L5_minus_L1_total']:+.2%}")
        print(f"  -- 这说明: 反向因子 + 正向因子互相抵消是等权的死穴, IC 加权能自动给反向因子负权重救活策略.")
    elif m_ic["ic_ir"] > m_eq["ic_ir"]:
        print(f"  IC 加权把 IR 从 {m_eq['ic_ir']:+.3f} 提到 {m_ic['ic_ir']:+.3f}, 验证'让有效因子主导'的工业方案有效.")
    else:
        print(f"  本数据集上两种合成差不多, 可能因为 (1) 因子之间高度同向 (2) 样本期太短.")


if __name__ == "__main__":
    main()
