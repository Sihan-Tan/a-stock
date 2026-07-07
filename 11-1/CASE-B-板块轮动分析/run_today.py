# -*- coding: utf-8 -*-
# 21-CASE-B: 当日板块轮动综合分析
"""
RunToday -- 综合调用 industry_strength + inflection_detector
        给出当日的"板块轮动综合视图"

输出 (供 CASE-D 晨会工作流消费):
    - sector_strength_today.csv      静态强度排名 (MOM_21 + RS_60 + VOL_RATIO)
    - sector_phase_today.csv         一/二阶导拐点扫描
    - sector_combined_today.csv      合并视图: 强度 + phase, 按推荐度排序
    - sector_today_topN.txt          人类可读的 Top N 摘要 (晨会用)
"""
from __future__ import annotations
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from industry_strength import rank_industries
from inflection_detector import scan_all_sectors_phase, PHASE_DESC


# ============================================================
# 综合视图
# ============================================================

# 给 phase 一个"推荐分", 用于综合排序时加权
PHASE_BONUS = {
    "accel_up":   3.0,   # 主升: 强烈推荐
    "decel_down": 2.0,   # 见底: 推荐 (左侧抄底)
    "decel_up":   0.5,   # 高位钝化: 中性偏空
    "accel_down": -2.0,  # 主跌: 规避
    "neutral":    0.0,
}


def build_combined_view(level: int = 2,
                         end_date: Optional[str] = None,
                         lookback_days: int = 90) -> pd.DataFrame:
    """
    构建当日板块综合视图: 强度排名 + phase 拐点

    返回: DataFrame, index=sector_name, columns=[
        score, rank, MOM_21, RS_60, VOL_RATIO,
        phase, phase_desc, ROC_20, MA20_SLOPE, MACD_HIST,
        composite_score, composite_rank
    ]

    composite_score = score (强度 z-score 之和) + PHASE_BONUS[phase]
    """
    print(f"\n[TODAY] 构建综合视图 (level={level}) ...\n")

    df_strength = rank_industries(level=level, end_date=end_date,
                                   lookback_days=lookback_days)
    df_phase    = scan_all_sectors_phase(level=level, end_date=end_date)

    if df_strength.empty:
        print("[TODAY] 强度排名为空, 退出")
        return pd.DataFrame()

    # 以 strength 为基底, 左 join phase
    cols_phase = ["phase", "phase_desc", "ROC_20", "MA20_SLOPE",
                  "MACD_HIST", "MA20_ACCEL", "vote_velocity", "vote_accel"]
    df_phase_sub = df_phase[cols_phase] if not df_phase.empty else pd.DataFrame(columns=cols_phase)

    df = df_strength.join(df_phase_sub, how="left")
    df["phase"]      = df["phase"].fillna("neutral")
    df["phase_desc"] = df["phase_desc"].fillna(PHASE_DESC["neutral"])

    df["phase_bonus"]     = df["phase"].map(PHASE_BONUS).fillna(0.0)
    df["composite_score"] = df["score"] + df["phase_bonus"]
    df = df.sort_values("composite_score", ascending=False)
    df["composite_rank"]  = range(1, len(df) + 1)

    return df


# ============================================================
# 落盘
# ============================================================

def save_today_view(df_combined: pd.DataFrame,
                     output_dir: Path,
                     top_n: int = 10) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    today_str = date.today().strftime("%Y-%m-%d")

    # 1. 完整 CSV
    df_combined.to_csv(output_dir / "sector_combined_today.csv",
                        encoding="utf-8-sig")

    # 2. 强度单表
    cols_strength = ["score", "rank", "MOM_21", "RS_60", "VOL_RATIO",
                     "MOM_21_z", "RS_60_z", "VOL_RATIO_z", "member_count"]
    df_combined[cols_strength].to_csv(output_dir / "sector_strength_today.csv",
                                        encoding="utf-8-sig")

    # 3. phase 单表
    cols_phase = ["phase", "phase_desc", "ROC_20", "MA20_SLOPE",
                  "MACD_HIST", "MA20_ACCEL", "vote_velocity", "vote_accel"]
    df_combined[cols_phase].to_csv(output_dir / "sector_phase_today.csv",
                                     encoding="utf-8-sig")

    # 4. 人类可读 Top N
    top = df_combined.head(top_n)
    bottom = df_combined.tail(5).iloc[::-1]
    bullish = df_combined[df_combined["phase"].isin(["accel_up", "decel_down"])]

    lines = []
    lines.append(f"{'='*78}")
    lines.append(f"  板块轮动晨会 -- {today_str} (申万{'一' if False else '二'}级)")
    lines.append(f"{'='*78}\n")

    lines.append(f"[1] Top {top_n} 综合推荐板块  (composite_score = strength_z + phase_bonus)\n")
    for i, (name, row) in enumerate(top.iterrows(), 1):
        lines.append(
            f"  #{i:>2} {name:<14s}  score={row['composite_score']:>+5.2f}  "
            f"phase={row['phase']:<11s}  ROC20={row.get('ROC_20', 0):>+5.1f}%  "
            f"MOM21={row['MOM_21']:>+5.1f}%  RS60={row['RS_60']:>+5.1f}%")

    lines.append(f"\n[2] 拐点信号: 主升 (accel_up) + 左侧抄底 (decel_down) 共 {len(bullish)} 个\n")
    for i, (name, row) in enumerate(bullish.head(10).iterrows(), 1):
        lines.append(
            f"  #{i:>2} {name:<14s}  {row['phase_desc']:<14s}  "
            f"ROC20={row.get('ROC_20', 0):>+5.1f}%  "
            f"MACD_HIST={row.get('MACD_HIST', 0):>+6.2f}")

    lines.append(f"\n[3] 警示板块 (排名末 5)\n")
    for i, (name, row) in enumerate(bottom.iterrows(), 1):
        lines.append(
            f"  #{i} {name:<14s}  score={row['composite_score']:>+5.2f}  "
            f"phase={row['phase']:<11s}  ROC20={row.get('ROC_20', 0):>+5.1f}%")

    lines.append(f"\n{'-'*78}")
    lines.append(f"完整数据见 sector_combined_today.csv")

    with open(output_dir / "sector_today_topN.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[OK] 当日视图已落盘: {output_dir}")
    print(f"    - sector_combined_today.csv")
    print(f"    - sector_strength_today.csv")
    print(f"    - sector_phase_today.csv")
    print(f"    - sector_today_topN.txt\n")


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="当日板块轮动综合分析")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2,
                        help="申万级别 (默认 2)")
    parser.add_argument("--end-date", default=None,
                        help="截止日 YYYY-MM-DD, 默认最新")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()

    df = build_combined_view(level=args.level,
                              end_date=args.end_date)
    if df.empty:
        return

    save_today_view(df, Path(__file__).parent / args.output, top_n=args.top)


if __name__ == "__main__":
    main()
