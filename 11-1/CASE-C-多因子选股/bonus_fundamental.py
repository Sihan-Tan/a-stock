# -*- coding: utf-8 -*-
# 21-CASE-C 多因子: BONUS 实验 -- 加财务因子能不能跑赢沪深 300?
"""
BonusFundamental -- 给"纯技术因子"加上财务因子, 看能不能在沪深 300 上挖出 alpha

财务因子设计 (4 个, 都是 "越大越好" 方向):
    1. ROE              -- 净资产收益率, 巴菲特最看重的指标
    2. NetProfit_YoY    -- 净利润同比增速 (本期 / 去年同期 - 1)
    3. GrossMargin      -- 毛利率, 反映行业护城河
    4. NegDebtRatio     -- 负的资产负债率 (低杠杆好), 取负号统一方向

时间对齐 (核心难点 -- 避免未来信息泄露):
    A 股财报发布滞后:
        Q1 (3-31)  通常 4 月底发  -> 实际滞后 ~30 天
        Q2 (6-30)  通常 8 月底发  -> 实际滞后 ~60 天
        Q3 (9-30)  通常 10 月底发 -> 实际滞后 ~30 天
        Q4 (12-31) 通常次年 4 月底发 -> 实际滞后 ~120 天 (!!)
    最大滞后是 Q4 年报 = 120 天. 用 lag_days=60 会在 1-4 月误用年报 -> 未来函数!
    本脚本默认 lag_days=120 天, 即"财报截止日 + 120 天"才允许使用,
    覆盖所有报告期的实际发布滞后. 严苛但安全.

    历史教训: 第一版用 60 天, 纯财务策略 Top-5 跑出 +31%, 看似惊艳;
    用 120 天严控未来函数后, 策略效果会明显回归到真实水平 (这就是教学意义所在).

数据来源 (重要!):
    - K 线: xtdata (miniQMT 实时拉)
    - 财务: data/csi300_fundamental.csv
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

# 复用主 pipeline 的工具
from preprocessor import preprocess_factors
from synthesizer import calc_ic, ic_weighted_synthesis
from stock_pool import filter_tradable, get_industry_map
from layered_backtest import _calc_factor_snapshot, _next_period_returns, calc_benchmark_returns


# csv 列名 -> 内部因子名 (统一为"越大越好"方向)
CSV_FACTOR_MAP = {
    "roe":          "ROE",
    "gross_margin": "GrossMargin",
    "debt_ratio":   "NegDebtRatio",   # 负向: csv 是原值, 加载时取负
    # NetProfit_YoY 不是 csv 直接列, 由 net_profit 算 YoY 派生
}


# ============================================================
# 1. 从 csv 加载财务因子 (按报告期面板)
# ============================================================

def load_fundamental_csv(csv_path: str | Path) -> Dict[str, pd.DataFrame]:
    """
    加载老师导出的 csv, 切分成每只股票的因子时间序列

    csv 列 (export_financial_to_csv.py 产出):
        stock_code, report_date,
        roe, net_profit, gross_margin, debt_ratio,
        net_margin, operating_cashflow, total_equity

    返回:
        {stock_code: DataFrame(index=report_date "yyyy-mm-dd",
                               columns=[ROE, NetProfit_YoY, GrossMargin, NegDebtRatio])}

    NetProfit_YoY 计算规则 (核心防雷点):
        增速 = (本期净利润 - 去年同期净利润) / |去年同期净利润|
        - 用绝对值做分母, 避免去年亏损时分母为负导致符号反转
        - 仅在 |去年同期| > 100 万时计算, 防止小分母放大噪声
        - 钳位到 [-200%, +500%], 避免极端值压垮 Z-score
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"财务 csv 不存在: {csv_path}\n"
            f"请先跑 export_financial_to_csv.py 生成 (需要 wucai_trade MySQL)"
        )

    print(f"[FUND] 读 csv: {csv_path}")
    df_raw = pd.read_csv(csv_path)
    df_raw["report_date"] = pd.to_datetime(df_raw["report_date"]).dt.strftime("%Y-%m-%d")
    print(f"[FUND] 原始 {len(df_raw)} 行 / {df_raw['stock_code'].nunique()} 只股票")

    panel: Dict[str, pd.DataFrame] = {}
    for code, sub in df_raw.groupby("stock_code"):
        sub = sub.sort_values("report_date").set_index("report_date")

        out = pd.DataFrame(index=sub.index)
        out["ROE"] = sub["roe"].astype(float)
        out["GrossMargin"] = sub["gross_margin"].astype(float)
        out["NegDebtRatio"] = -sub["debt_ratio"].astype(float)

        out["NetProfit_YoY"] = np.nan
        np_series = sub["net_profit"].astype(float)
        for cur_p in sub.index:
            try:
                cur_dt = pd.to_datetime(cur_p)
                last_p = (cur_dt - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
                if last_p in sub.index:
                    cur_np = np_series.loc[cur_p]
                    last_np = np_series.loc[last_p]
                    if pd.notna(cur_np) and pd.notna(last_np) and abs(last_np) > 1e6:
                        yoy = (cur_np - last_np) / abs(last_np)
                        out.at[cur_p, "NetProfit_YoY"] = float(np.clip(yoy * 100, -200, 500))
            except Exception:
                continue

        panel[code] = out

    print(f"[FUND] 切分: {len(panel)} 只 -> 因子面板 (4 个因子 x 报告期)")
    return panel


def fundamental_as_of(panel: Dict[str, pd.DataFrame],
                      as_of_date: str,
                      lag_days: int = 120) -> pd.DataFrame:
    """
    截至 as_of_date 时点 (含), 返回每只股票最新可用的财务因子值

    可用规则: report_date + lag_days <= as_of_date  (避免未来信息泄露)

    lag_days 取值:
        - 120 天 (默认, 安全):
            覆盖 A 股最坏情况的 Q4 年报发布 (12-31 截止 -> 次年 4 月底发).
            适合"严格防未来函数"的回测.
        - 60 天 (激进, 危险):
            只覆盖 Q1/Q2/Q3 季报, Q4 年报会被提前用 -> 未来函数.
            诊断脚本 _diag_lookahead.py 可以验证哪些 as-of 日期会"提前看到"年报.

    返回: DataFrame(index=stock_code, columns=4 个财务因子)
    """
    cutoff = pd.to_datetime(as_of_date) - pd.Timedelta(days=lag_days)
    cutoff_str = cutoff.strftime("%Y-%m-%d")
    rows = {}
    for code, df in panel.items():
        usable = df[df.index <= cutoff_str]
        if len(usable) == 0:
            continue
        rows[code] = usable.iloc[-1].to_dict()
    return pd.DataFrame.from_dict(rows, orient="index")


def load_csi300_codes_from_file(codes_txt: str | Path) -> List[str]:
    """从导出脚本产出的代码列表 (一行一个) 加载沪深 300, 不依赖 xtdata"""
    codes_txt = Path(codes_txt)
    if not codes_txt.exists():
        return []
    return [c.strip() for c in codes_txt.read_text(encoding="utf-8").splitlines() if c.strip()]


# ============================================================
# 2. 增强版分层回测 -- 技术因子 + 财务因子联合
# ============================================================

def run_combined_backtest(prices_panel: Dict[str, pd.DataFrame],
                          industry_map: Dict[str, str],
                          fundamental_panel: Dict[str, pd.DataFrame] = None,
                          mode: str = "tech_only",
                          rebal_period: int = 21,
                          n_layers: int = 5,
                          min_warmup: int = 130,
                          ic_lookback: int = 6,
                          top_n_list: List[int] = None,
                          benchmark_returns: pd.Series = None,
                          fund_lag_days: int = 120) -> dict:
    """
    支持 3 种 mode:
        - "tech_only"     : 只用 10 个技术因子
        - "fund_only"     : 只用 4 个财务因子
        - "tech_plus_fund": 14 个因子合并 (技术 + 财务)

    其余参数同主 layered_backtest, 输出格式也对齐, 方便横向比较.
    """
    if top_n_list is None:
        top_n_list = [5, 10]

    first_code = next(iter(prices_panel))
    base_index = prices_panel[first_code].index
    n = len(base_index)

    if n < min_warmup + rebal_period * 2:
        raise ValueError(f"数据不足, 至少需要 {min_warmup + rebal_period * 2} 行")

    rebal_dates = list(range(min_warmup, n - rebal_period, rebal_period))
    print(f"[BACKTEST/{mode}] 数据 {n} 行, 调仓 {len(rebal_dates)} 次")

    layer_returns = []
    ic_records = []
    topn_records = []
    single_ic_records = {}

    for i, end_idx in enumerate(rebal_dates):
        # 技术因子
        tech_factors = _calc_factor_snapshot(prices_panel, end_idx)
        if len(tech_factors) < n_layers * 3:
            continue

        # 财务因子 (按 as-of 时点查表, 默认严格 120 天滞后)
        as_of_str = str(base_index[end_idx])[:10]
        if mode in ("fund_only", "tech_plus_fund") and fundamental_panel:
            fund_factors = fundamental_as_of(fundamental_panel, as_of_str,
                                             lag_days=fund_lag_days)
            if len(fund_factors) == 0:
                continue
        else:
            fund_factors = pd.DataFrame()

        # 选择因子矩阵
        if mode == "tech_only":
            factor_df = tech_factors
        elif mode == "fund_only":
            common = tech_factors.index.intersection(fund_factors.index)
            factor_df = fund_factors.loc[common]
        else:  # tech_plus_fund
            common = tech_factors.index.intersection(fund_factors.index)
            factor_df = pd.concat([tech_factors.loc[common], fund_factors.loc[common]], axis=1)

        if len(factor_df) < n_layers * 3:
            continue

        # 预处理 (去极值 + Z-score + 行业中性化)
        factor_processed = preprocess_factors(factor_df, industry_map=industry_map,
                                              neutralize=True)

        # 下期收益
        future_ret = _next_period_returns(prices_panel, end_idx, rebal_period)

        # 单因子 IC (供 IC 加权 walk-forward)
        for col in factor_processed.columns:
            ic_one = calc_ic(factor_processed[col], future_ret)
            single_ic_records.setdefault(col, []).append(ic_one)

        # IC 加权合成 (前 ic_lookback 期没历史时退化为等权)
        past_ic = {}
        for col, hist in single_ic_records.items():
            hist_no_now = hist[:-1]
            recent = [x for x in hist_no_now[-ic_lookback:] if not np.isnan(x)]
            if recent:
                past_ic[col] = float(np.mean(recent))
        if past_ic and any(v != 0 for v in past_ic.values()):
            alpha = ic_weighted_synthesis(factor_processed, past_ic).dropna()
        else:
            alpha = factor_processed.mean(axis=1).dropna()

        if len(alpha) < n_layers * 3:
            continue

        # 合成因子的 IC
        ic = calc_ic(alpha, future_ret)
        ic_records.append({"date": as_of_str, "ic": ic})

        # 分层
        common_idx = alpha.index.intersection(future_ret.index)
        df_step = pd.DataFrame({"alpha": alpha.loc[common_idx],
                                "ret": future_ret.loc[common_idx]})
        try:
            df_step["layer"] = pd.qcut(df_step["alpha"], n_layers,
                                       labels=range(1, n_layers + 1),
                                       duplicates="drop")
        except Exception:
            continue
        layer_means = df_step.groupby("layer", observed=True)["ret"].mean()
        layer_dict = {f"L{j}": layer_means.get(j, 0) for j in range(1, n_layers + 1)}
        layer_dict["date"] = as_of_str
        layer_dict["long_short"] = layer_means.get(n_layers, 0) - layer_means.get(1, 0)
        layer_returns.append(layer_dict)

        # Top-N 集中度: 同时记录顺向 (Top, 买 alpha 最高) 和反向 (Bot, 买 alpha 最低)
        # 反向是为了验证"沪深 300 是反转市场"的假说 -- 主回测里 L1 跑赢 L5
        topn_dict = {"date": as_of_str}
        df_sorted = df_step.sort_values("alpha", ascending=False)
        for k in top_n_list:
            if k <= len(df_sorted):
                topn_dict[f"Top{k}"] = df_sorted["ret"].iloc[:k].mean()
                topn_dict[f"Bot{k}"] = df_sorted["ret"].iloc[-k:].mean()
            else:
                topn_dict[f"Top{k}"] = np.nan
                topn_dict[f"Bot{k}"] = np.nan
        topn_records.append(topn_dict)

    # ---- 汇总 ----
    layer_returns_df = pd.DataFrame(layer_returns).set_index("date")
    layer_cumret = (1 + layer_returns_df.drop(columns=["long_short"])).cumprod()
    ic_series = pd.DataFrame(ic_records).set_index("date")["ic"]
    topn_returns_df = pd.DataFrame(topn_records).set_index("date") if topn_records else pd.DataFrame()
    topn_cumret = (1 + topn_returns_df).cumprod() if not topn_returns_df.empty else pd.DataFrame()

    metrics = {
        "mode":              mode,
        "rebal_count":       len(layer_returns),
        "ic_mean":           float(ic_series.mean()) if len(ic_series) > 0 else 0,
        "ic_ir":             float(ic_series.mean() / ic_series.std()) if ic_series.std() > 0 else 0,
        "L5_total_ret":      float(layer_cumret["L5"].iloc[-1] - 1) if "L5" in layer_cumret.columns else 0,
        "L1_total_ret":      float(layer_cumret["L1"].iloc[-1] - 1) if "L1" in layer_cumret.columns else 0,
        "L5_minus_L1":       float(layer_cumret["L5"].iloc[-1] - layer_cumret["L1"].iloc[-1])
                              if "L5" in layer_cumret.columns and "L1" in layer_cumret.columns else 0,
    }
    for col in topn_cumret.columns:
        metrics[f"{col}_total_ret"] = float(topn_cumret[col].iloc[-1] - 1)
        cum = topn_cumret[col]
        metrics[f"{col}_max_dd"] = float((cum / cum.cummax() - 1).min())

    if benchmark_returns is not None and not benchmark_returns.empty:
        # 对齐用 index 交集 (reindex 在某些 dtype 不一致时会全 NaN, 这里更稳)
        bench_idx = pd.Index([str(x)[:10] for x in benchmark_returns.index])
        layer_idx = pd.Index([str(x)[:10] for x in layer_returns_df.index])
        common = bench_idx.intersection(layer_idx)
        if len(common) > 0:
            bench_clean = pd.Series(benchmark_returns.values, index=bench_idx).loc[common]
            metrics["bench_total_ret"] = float((1 + bench_clean.fillna(0)).prod() - 1)
        else:
            # 退而求其次: 用 bench 自己的全区间累计 (调仓日期可能不一致)
            metrics["bench_total_ret"] = float((1 + benchmark_returns.fillna(0)).prod() - 1)
        for col in topn_cumret.columns:
            metrics[f"{col}_excess"] = metrics[f"{col}_total_ret"] - metrics["bench_total_ret"]

    return {
        "layer_cumret":  layer_cumret,
        "topn_cumret":   topn_cumret,
        "ic_series":     ic_series,
        "metrics":       metrics,
    }


# ============================================================
# 3. 主入口 -- 三方案对比
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="技术 + 财务 联合多因子 (BONUS 实验)")
    parser.add_argument("--max-stocks", type=int, default=30,
                        help="只用前 N 只股票, 默认 30 (散户友好)")
    parser.add_argument("--lookback", type=int, default=400,
                        help="拉多少日 K 线, 默认 400")
    parser.add_argument("--rebal", type=int, default=21,
                        help="调仓周期 (日), 默认 21 = 月度")
    parser.add_argument("--ic-lookback", type=int, default=6,
                        help="IC 加权用前几期 IC, 默认 6")
    parser.add_argument("--fund-csv", default="data/csi300_fundamental.csv",
                        help="财务 csv 路径, 默认 data/csi300_fundamental.csv")
    parser.add_argument("--codes-txt", default="data/csi300_codes.txt",
                        help="股票池快照 (没有就走 xtdata 实时拉)")
    parser.add_argument("--fund-lag", type=int, default=120,
                        help="财报使用滞后天数, 默认 120 (覆盖 Q4 年报). "
                             "改成 60 会引入未来函数 (用于教学对比)")
    args = parser.parse_args()

    print(f"\n{'='*72}")
    print(f"  21-CASE-C BONUS: 技术因子 vs 财务因子 vs 技术+财务 (沪深 300 实验)")
    print(f"{'='*72}\n")

    # 1. 选股池 (优先用快照, 否则 xtdata)
    print(f"[1/5] 加载沪深 300 股票池 ...")
    codes = load_csi300_codes_from_file(args.codes_txt)
    if codes:
        print(f"  从快照 {args.codes_txt} 加载 {len(codes)} 只")
    else:
        from stock_pool import get_csi300
        codes = get_csi300()
        print(f"  从 xtdata 实时拉 {len(codes)} 只")

    codes = filter_tradable(codes)[:args.max_stocks]
    print(f"  过滤后参与回测: {len(codes)} 只")

    # 2. 行业映射 + K 线 (xtdata)
    from xtquant import xtdata
    xtdata.connect()

    print(f"\n[2/5] 行业映射 ...")
    industry_map = get_industry_map(codes)

    print(f"\n[3/5] 拉 K 线 ({args.lookback} 日) ...")
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
    print(f"  有效 K 线: {len(prices_panel)} 只")

    # 3. 财务数据 (csv)
    print(f"\n[4/5] 加载财务因子 (csv) ...")
    fund_panel = load_fundamental_csv(args.fund_csv)
    sample_code = next(iter(prices_panel))
    if sample_code in fund_panel:
        print(f"\n  [示例] {sample_code} 最近 4 期:")
        print(fund_panel[sample_code].tail(4).round(2).to_string())

    # 4. 沪深 300 基准
    print(f"\n[5/5] 拉沪深 300 指数作为基准 ...")
    bench_returns = pd.Series(dtype=float)
    try:
        xtdata.download_history_data("000300.SH", period="1d", start_time="20230101")
        bench_data = xtdata.get_market_data_ex(
            field_list=["close"], stock_list=["000300.SH"], period="1d",
            count=args.lookback, dividend_type="none",
        )
        bench_df = bench_data.get("000300.SH")
        if bench_df is not None and len(bench_df) > 200:
            bench_df = bench_df.copy()
            bench_df.index = pd.to_datetime(bench_df.index)
            bench_returns = calc_benchmark_returns(bench_df, rebal_period=args.rebal,
                                                   min_warmup=130)
            print(f"  沪深 300 同期累计: {(1+bench_returns).prod() - 1:+.2%} "
                  f"({len(bench_returns)} 期)")
    except Exception as e:
        print(f"  [WARN] 拉基准失败: {e}")

    # 5. 三方案对比
    print(f"\n{'='*72}\n  跑三方案分层回测 (IC 加权 + 月度调仓)\n{'='*72}")

    bench_arg = bench_returns if not bench_returns.empty else None

    print(f"\n  方案 A: 纯技术因子 (10 个)")
    res_tech = run_combined_backtest(prices_panel, industry_map, fundamental_panel=None,
                                     mode="tech_only", rebal_period=args.rebal,
                                     ic_lookback=args.ic_lookback,
                                     top_n_list=[5, 10],
                                     benchmark_returns=bench_arg,
                                     fund_lag_days=args.fund_lag)

    print(f"\n  方案 B: 纯财务因子 (4 个) [财报滞后 {args.fund_lag} 天]")
    res_fund = run_combined_backtest(prices_panel, industry_map, fundamental_panel=fund_panel,
                                     mode="fund_only", rebal_period=args.rebal,
                                     ic_lookback=args.ic_lookback,
                                     top_n_list=[5, 10],
                                     benchmark_returns=bench_arg,
                                     fund_lag_days=args.fund_lag)

    print(f"\n  方案 C: 技术 + 财务 联合 (10 + 4 = 14 个) [财报滞后 {args.fund_lag} 天]")
    res_both = run_combined_backtest(prices_panel, industry_map, fundamental_panel=fund_panel,
                                     mode="tech_plus_fund", rebal_period=args.rebal,
                                     ic_lookback=args.ic_lookback,
                                     top_n_list=[5, 10],
                                     benchmark_returns=bench_arg,
                                     fund_lag_days=args.fund_lag)

    # ---- 汇总输出 ----
    print(f"\n{'='*72}")
    print(f"  三方案对比 -- 沪深 300 前 {args.max_stocks} 只 x 月度调仓 x IC 加权")
    print(f"{'='*72}\n")

    bench_total = res_tech["metrics"].get("bench_total_ret")

    rows = [
        ("IC 均值",         "ic_mean",        "{:+.4f}"),
        ("IC IR",           "ic_ir",          "{:+.4f}"),
        ("L5 累计收益",     "L5_total_ret",   "{:+.2%}"),
        ("L1 累计收益",     "L1_total_ret",   "{:+.2%}"),
        ("L5-L1 多空",      "L5_minus_L1",    "{:+.2%}"),
        ("Top5 (顺向, 买高)", "Top5_total_ret",  "{:+.2%}"),
        ("Bot5 (反转, 买低)", "Bot5_total_ret",  "{:+.2%}"),
        ("Top10 (顺向)",      "Top10_total_ret", "{:+.2%}"),
        ("Bot10 (反转)",      "Bot10_total_ret", "{:+.2%}"),
    ]
    if bench_total is not None:
        rows.append(("Top5 vs 沪深300",   "Top5_excess",  "{:+.2%}"))
        rows.append(("Bot5 vs 沪深300",   "Bot5_excess",  "{:+.2%}"))
        rows.append(("Top10 vs 沪深300",  "Top10_excess", "{:+.2%}"))
        rows.append(("Bot10 vs 沪深300",  "Bot10_excess", "{:+.2%}"))

    print(f"  {'指标':<22} {'A 纯技术':>14} {'B 纯财务':>14} {'C 技术+财务':>14}")
    print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*14}")
    for label, key, fmt in rows:
        v_a = res_tech["metrics"].get(key, 0)
        v_b = res_fund["metrics"].get(key, 0)
        v_c = res_both["metrics"].get(key, 0)
        print(f"  {label:<22} {fmt.format(v_a):>14} {fmt.format(v_b):>14} {fmt.format(v_c):>14}")

    if bench_total is not None:
        print(f"\n  [基准] 沪深 300 同期累计: {bench_total:+.2%}")

    # ---- 找出 6 种组合里跑赢基准的赢家 ----
    print(f"\n[结论]")
    candidates = []
    for plan_name, res in [("A 纯技术", res_tech),
                           ("B 纯财务", res_fund),
                           ("C 技术+财务", res_both)]:
        for k in [5, 10]:
            for direction in ["Top", "Bot"]:
                key = f"{direction}{k}_total_ret"
                v = res["metrics"].get(key, 0)
                candidates.append((f"{plan_name}-{direction}{k}", v))

    # 排序找最强
    candidates.sort(key=lambda x: -x[1])
    if bench_total is not None:
        winners = [c for c in candidates if c[1] > bench_total]
        if winners:
            print(f"  跑赢沪深 300 ({bench_total:+.2%}) 的方案 ({len(winners)} 个):")
            for name, v in winners:
                print(f"    {name:<22} 累计收益 {v:+.2%}  (超额 {v - bench_total:+.2%})")
            print(f"\n  最强: {winners[0][0]} = {winners[0][1]:+.2%}")
        else:
            print(f"  6 个组合都没跑赢沪深 300 ({bench_total:+.2%}).")
            print(f"  最接近的: {candidates[0][0]} = {candidates[0][1]:+.2%} "
                  f"(差距 {candidates[0][1] - bench_total:+.2%})")
            print(f"  -- 这个池子 (沪深 300 前 30 只) + 1 年回测里, 现成的因子/方向都失效.")
            print(f"     可能原因: 样本期太短(12 期); 30 只大盘股 alpha 早被市场定价;")
            print(f"     需要更长历史/更换池子(中证 1000)/加估值因子才能继续挖.")
    else:
        print(f"  最强组合: {candidates[0][0]} = {candidates[0][1]:+.2%}")


if __name__ == "__main__":
    main()
