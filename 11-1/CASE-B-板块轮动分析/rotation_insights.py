# -*- coding: utf-8 -*-
# 21-CASE-B: 板块轮动深度洞察分析
"""
RotationInsights -- 从 backtest.py 落盘的 rotation_events.csv 里, 再往下挖 4 类规律

输入:  outputs/rotation_events.csv  (由 backtest.py 产出)
输出:  outputs/insights/             (本脚本落盘)
       - switch_matrix.csv           5x5 切换矩阵详细统计
       - decay_curves.csv            5d/10d/20d 衰减
       - sector_reliability.csv      板块信号可靠度排行
       - monthly_density.csv         月度切换密度
       - INSIGHTS.txt                人类可读的总结报告
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# ============================================================
# 工具
# ============================================================

PHASE_ORDER = ["accel_up", "decel_up", "neutral", "decel_down", "accel_down"]
PHASE_DESC = {
    "accel_up":   "主升加速",
    "decel_up":   "高位钝化",
    "accel_down": "主跌",
    "decel_down": "左侧抄底",
    "neutral":    "中性",
}


def _winrate(s: pd.Series) -> float:
    """胜率: 收益 > 0 的占比"""
    s = s.dropna()
    if len(s) == 0:
        return np.nan
    return float((s > 0).mean() * 100)


# ============================================================
# 1. 切换矩阵 (5x5)
# ============================================================

def build_switch_matrix(events: pd.DataFrame) -> pd.DataFrame:
    """
    按 from_phase x to_phase 分组, 统计:
        count          事件数
        mean_excess_20d  平均后 20 日超额收益 (%)
        median_excess_20d 中位数 (避免极端值)
        winrate_20d    胜率 (%)
    """
    g = events.groupby(["from_phase", "to_phase"])
    matrix = pd.DataFrame({
        "count":             g.size(),
        "mean_excess_20d":   g["excess_20d"].mean().round(2),
        "median_excess_20d": g["excess_20d"].median().round(2),
        "winrate_20d":       g["excess_20d"].apply(_winrate).round(1),
    }).reset_index()

    # 排序: 按平均超额收益从高到低
    matrix = matrix.sort_values("mean_excess_20d", ascending=False)
    return matrix


# ============================================================
# 2. 衰减曲线 (5d / 10d / 20d)
# ============================================================

def build_decay_curves(events: pd.DataFrame, min_count: int = 30) -> pd.DataFrame:
    """
    各 (from -> to) 路径在 5d / 10d / 20d 三个窗口的平均超额收益
    只保留 count >= min_count 的路径 (避免小样本噪音)
    """
    rows = []
    for (frm, to), sub in events.groupby(["from_phase", "to_phase"]):
        if len(sub) < min_count:
            continue
        rows.append({
            "from_phase":      frm,
            "to_phase":        to,
            "count":           len(sub),
            "mean_excess_5d":  round(float(sub["excess_5d"].mean()),  2),
            "mean_excess_10d": round(float(sub["excess_10d"].mean()), 2),
            "mean_excess_20d": round(float(sub["excess_20d"].mean()), 2),
            "winrate_5d":      round(_winrate(sub["excess_5d"]),  1),
            "winrate_10d":     round(_winrate(sub["excess_10d"]), 1),
            "winrate_20d":     round(_winrate(sub["excess_20d"]), 1),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["decay_signal"] = df["mean_excess_20d"] - df["mean_excess_5d"]
    return df.sort_values("mean_excess_20d", ascending=False)


# ============================================================
# 3. 板块信号可靠度
# ============================================================

def build_sector_reliability(events: pd.DataFrame, min_signals: int = 3) -> pd.DataFrame:
    """
    各板块 accel_up 切换的胜率 + 平均超额收益
    至少 min_signals 个信号才纳入排行 (避免单次极端值)
    """
    bull = events[events["to_phase"].isin(["accel_up", "decel_down"])]
    rows = []
    for sector, sub in bull.groupby("sector_name"):
        sub = sub.dropna(subset=["excess_20d"])
        if len(sub) < min_signals:
            continue
        rows.append({
            "sector_name":     sector,
            "n_bull_signals":  len(sub),
            "mean_excess_20d": round(float(sub["excess_20d"].mean()),   2),
            "median_20d":      round(float(sub["excess_20d"].median()), 2),
            "winrate_20d":     round(_winrate(sub["excess_20d"]), 1),
            "best_event":      round(float(sub["excess_20d"].max()),  2),
            "worst_event":     round(float(sub["excess_20d"].min()),  2),
        })
    df = pd.DataFrame(rows)
    return df.sort_values("mean_excess_20d", ascending=False) if not df.empty else df


# ============================================================
# 4. 月度切换密度 (市场温度计)
# ============================================================

def build_monthly_density(events: pd.DataFrame) -> pd.DataFrame:
    """
    每个月发生多少次 accel_up / accel_down 切换
    -- 牛市 accel_up 多, 熊市 accel_down 多
    """
    df = events.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["month"] = df["event_date"].dt.to_period("M").astype(str)

    pivot = (df.groupby(["month", "to_phase"]).size()
             .unstack(fill_value=0)
             .reset_index())

    for col in ["accel_up", "accel_down", "decel_up", "decel_down", "neutral"]:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["bull_minus_bear"] = pivot["accel_up"] - pivot["accel_down"]
    return pivot[["month", "accel_up", "accel_down", "decel_up", "decel_down",
                   "neutral", "bull_minus_bear"]]


# ============================================================
# 总结报告 (人类可读)
# ============================================================

def write_insights_report(matrix: pd.DataFrame,
                           decay: pd.DataFrame,
                           reliability: pd.DataFrame,
                           density: pd.DataFrame,
                           output_path: Path) -> None:
    lines = []
    sep = "=" * 78
    lines.append(sep)
    lines.append("  板块轮动深度洞察 -- 4 类规律")
    lines.append(sep)

    lines.append("\n[1] 切换矩阵 Top 5 (按平均后 20 日超额收益)")
    lines.append("-" * 78)
    lines.append(f"{'from_phase':<14s}{'to_phase':<14s}{'count':>8s}"
                 f"{'mean_ex20d':>14s}{'win_rate':>12s}")
    for _, r in matrix.head(5).iterrows():
        lines.append(f"{r['from_phase']:<14s}{r['to_phase']:<14s}"
                     f"{int(r['count']):>8d}{r['mean_excess_20d']:>13.2f}%"
                     f"{r['winrate_20d']:>11.1f}%")

    lines.append("\n[1] 切换矩阵 Bottom 3 (最值得回避)")
    lines.append("-" * 78)
    for _, r in matrix.tail(3).iterrows():
        lines.append(f"{r['from_phase']:<14s}{r['to_phase']:<14s}"
                     f"{int(r['count']):>8d}{r['mean_excess_20d']:>13.2f}%"
                     f"{r['winrate_20d']:>11.1f}%")

    if not decay.empty:
        lines.append("\n[2] 信号衰减曲线 (各路径在 5d / 10d / 20d 的均值变化)")
        lines.append("-" * 78)
        lines.append(f"{'path':<28s}{'5d':>10s}{'10d':>10s}{'20d':>10s}{'decay':>10s}")
        for _, r in decay.head(8).iterrows():
            path = f"{r['from_phase']}->{r['to_phase']}"
            lines.append(f"{path:<28s}{r['mean_excess_5d']:>9.2f}%"
                         f"{r['mean_excess_10d']:>9.2f}%"
                         f"{r['mean_excess_20d']:>9.2f}%"
                         f"{r['decay_signal']:>+9.2f}%")

    if not reliability.empty:
        lines.append("\n[3] 板块信号可靠度排行 Top 10 (accel_up + decel_down 切换的平均超额)")
        lines.append("-" * 78)
        lines.append(f"{'sector':<14s}{'#signals':>10s}{'mean_ex20d':>14s}"
                     f"{'win_rate':>12s}{'best':>10s}{'worst':>10s}")
        for _, r in reliability.head(10).iterrows():
            lines.append(f"{r['sector_name']:<14s}{int(r['n_bull_signals']):>10d}"
                         f"{r['mean_excess_20d']:>13.2f}%{r['winrate_20d']:>11.1f}%"
                         f"{r['best_event']:>+9.2f}%{r['worst_event']:>+9.2f}%")

        lines.append("\n[3] 板块信号可靠度排行 Bottom 5 (信号最不靠谱)")
        lines.append("-" * 78)
        for _, r in reliability.tail(5).iterrows():
            lines.append(f"{r['sector_name']:<14s}{int(r['n_bull_signals']):>10d}"
                         f"{r['mean_excess_20d']:>13.2f}%{r['winrate_20d']:>11.1f}%"
                         f"{r['best_event']:>+9.2f}%{r['worst_event']:>+9.2f}%")

    if not density.empty:
        lines.append("\n[4] 月度切换密度 (近 12 个月)")
        lines.append("-" * 78)
        lines.append(f"{'month':<10s}{'accel_up':>10s}{'accel_down':>12s}"
                     f"{'bull-bear':>12s}")
        for _, r in density.tail(12).iterrows():
            lines.append(f"{r['month']:<10s}{int(r['accel_up']):>10d}"
                         f"{int(r['accel_down']):>12d}"
                         f"{int(r['bull_minus_bear']):>+12d}")

    lines.append("\n" + sep)
    lines.append("  详细数据见同目录:")
    lines.append("    switch_matrix.csv      switch matrix (full 5x5)")
    lines.append("    decay_curves.csv       5d/10d/20d decay")
    lines.append("    sector_reliability.csv sector signal reliability rank")
    lines.append("    monthly_density.csv    monthly switch density")
    lines.append(sep)

    output_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 主流程
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="板块轮动深度洞察")
    parser.add_argument("--events",
                        default="outputs/rotation_events.csv",
                        help="backtest.py 落盘的事件清单 (默认 outputs/rotation_events.csv)")
    parser.add_argument("--out",
                        default="outputs/insights",
                        help="洞察落盘目录 (默认 outputs/insights)")
    args = parser.parse_args()

    events_path = Path(__file__).parent / args.events
    out_dir = Path(__file__).parent / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Insights] 读入事件清单: {events_path}")
    events = pd.read_csv(events_path)
    print(f"[Insights] 共 {len(events)} 条事件")

    # 1. 切换矩阵
    matrix = build_switch_matrix(events)
    matrix.to_csv(out_dir / "switch_matrix.csv", index=False, encoding="utf-8-sig")
    print(f"[Insights] [1/4] 切换矩阵: {len(matrix)} 行 -> switch_matrix.csv")

    # 2. 衰减曲线
    decay = build_decay_curves(events)
    decay.to_csv(out_dir / "decay_curves.csv", index=False, encoding="utf-8-sig")
    print(f"[Insights] [2/4] 衰减曲线: {len(decay)} 条路径 -> decay_curves.csv")

    # 3. 板块信号可靠度
    reliability = build_sector_reliability(events)
    reliability.to_csv(out_dir / "sector_reliability.csv",
                        index=False, encoding="utf-8-sig")
    print(f"[Insights] [3/4] 板块可靠度: {len(reliability)} 个板块 -> sector_reliability.csv")

    # 4. 月度切换密度
    density = build_monthly_density(events)
    density.to_csv(out_dir / "monthly_density.csv",
                    index=False, encoding="utf-8-sig")
    print(f"[Insights] [4/4] 月度密度: {len(density)} 个月 -> monthly_density.csv")

    # 总结报告
    report_path = out_dir / "INSIGHTS.txt"
    write_insights_report(matrix, decay, reliability, density, report_path)
    print(f"\n[OK] 总结报告: {report_path}")
    print(f"\n--- 在线预览 ---\n")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
