# -*- coding: utf-8 -*-
# 21-CASE-B: 板块拐点检测 (基于一阶/二阶导的四象限分类)
"""
InflectionDetector -- 用一阶导 + 二阶导联合判断板块所处的"轮动象限"

四象限模型:

    一阶导 v / 二阶导 a       描述                  操作建议
    ─────────────────────────────────────────────────────────────
    v > 0,  a > 0             加速上涨, 强势确认       追涨 / 持有
    v > 0,  a < 0             减速上涨, 资金撤出预警   减仓 / 观望
    v < 0,  a < 0             加速下跌, 杀跌确认       清仓 / 不接刀
    v < 0,  a > 0             减速下跌, 见底信号       小仓试探 / 等右侧

业内别名 (技术分析里的常见叫法):
    减速上涨 = MACD 顶背离前兆
    减速下跌 = MACD 底背离前兆
    加速上涨 = 主升浪
    加速下跌 = 杀跌段 / 戴维斯双杀

==================================================
象限分类的输入信号
==================================================

为了减少单一指标的噪音, 本模块同时看 3 组导数信号:
    速度组 (一阶导): ROC_20, MA20_SLOPE
    加速度组 (二阶导): MA20_ACCEL, MACD_HIST 的方向 (与上一日比)

判定规则 (3 信号"投票多数"):
    - 速度方向 = sign(ROC_20)         (+1 / -1 / 0)
    - 加速度方向: 对 MA20_ACCEL 和 MACD_HIST 变化方向投票

输出: phase 字段 ∈ {accel_up, decel_up, accel_down, decel_down, neutral}
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from derivatives import calc_all_derivatives
from sector_loader import (load_all_sectors, list_sectors)


# ============================================================
# 单板块象限判定
# ============================================================

PHASE_DESC = {
    "accel_up":   "加速上涨 / 强势确认",
    "decel_up":   "减速上涨 / 撤出预警",
    "accel_down": "加速下跌 / 杀跌确认",
    "decel_down": "减速下跌 / 见底信号",
    "neutral":    "震荡 / 信号不明",
}


def _sign(x: float, threshold: float = 0.0) -> int:
    if pd.isna(x):
        return 0
    if x > threshold:
        return 1
    if x < -threshold:
        return -1
    return 0


def detect_one_sector_phase(close: pd.Series,
                              roc_threshold: float = 0.005,   # 0.5%
                              accel_threshold: float = 0.0
                              ) -> Dict[str, object]:
    """
    给单板块判一个 phase

    阈值说明:
        roc_threshold=0.5% 是为了过滤"几乎平盘"的板块 (噪音)
        accel_threshold=0  是因为加速度本身已经是"差值", 0 附近就当中性

    返回字典:
        phase, phase_desc, vote_velocity, vote_accel,
        roc_20, ma20_slope, macd_hist, ma20_accel, hist_delta
    """
    deriv = calc_all_derivatives(close)
    if len(deriv) < 30:
        return {"phase": "neutral", "phase_desc": "样本不足", "_skip": True}

    last = deriv.iloc[-1]
    prev = deriv.iloc[-2]

    roc_20      = last["ROC_20"]
    ma20_slope  = last["MA20_SLOPE"]
    macd_hist   = last["MACD_HIST"]
    ma20_accel  = last["MA20_ACCEL"]
    hist_delta  = last["MACD_HIST"] - prev["MACD_HIST"]

    # ---- 速度组投票 (ROC_20 占 2 票, MA20_SLOPE 占 1 票) ----
    v_score = (
        _sign(roc_20, roc_threshold) * 2
        + _sign(ma20_slope, accel_threshold) * 1
    )
    velocity_dir = 1 if v_score > 0 else (-1 if v_score < 0 else 0)

    # ---- 加速度组投票 (MA20_ACCEL 占 1 票, MACD_HIST 变化方向占 1 票) ----
    a_score = (
        _sign(ma20_accel, accel_threshold) * 1
        + _sign(hist_delta, accel_threshold) * 1
    )
    accel_dir = 1 if a_score > 0 else (-1 if a_score < 0 else 0)

    # ---- 组合判定 ----
    if velocity_dir > 0 and accel_dir > 0:
        phase = "accel_up"
    elif velocity_dir > 0 and accel_dir < 0:
        phase = "decel_up"
    elif velocity_dir < 0 and accel_dir < 0:
        phase = "accel_down"
    elif velocity_dir < 0 and accel_dir > 0:
        phase = "decel_down"
    else:
        phase = "neutral"

    return {
        "phase":         phase,
        "phase_desc":    PHASE_DESC[phase],
        "vote_velocity": velocity_dir,
        "vote_accel":    accel_dir,
        "ROC_20":        float(roc_20)     if pd.notna(roc_20) else np.nan,
        "MA20_SLOPE":    float(ma20_slope) if pd.notna(ma20_slope) else np.nan,
        "MACD_HIST":     float(macd_hist)  if pd.notna(macd_hist) else np.nan,
        "MA20_ACCEL":    float(ma20_accel) if pd.notna(ma20_accel) else np.nan,
        "HIST_DELTA":    float(hist_delta) if pd.notna(hist_delta) else np.nan,
    }


# ============================================================
# 横截面: 给所有板块标 phase
# ============================================================

def scan_all_sectors_phase(level: int = 2,
                            end_date: Optional[str] = None) -> pd.DataFrame:
    """
    扫描申万 N 级所有板块, 给每个标一个轮动象限

    返回: DataFrame, index=sector_name, columns=[
        phase, phase_desc, vote_velocity, vote_accel,
        ROC_20, MA20_SLOPE, MACD_HIST, MA20_ACCEL, HIST_DELTA, member_count
    ]
    """
    panel = load_all_sectors(level=level, end_date=end_date)
    sectors_meta = {s["sector_name"]: s for s in list_sectors(level=level)}

    rows = {}
    for sector_name, df in panel.items():
        result = detect_one_sector_phase(df["close"])
        if result.get("_skip"):
            continue
        result["member_count"] = sectors_meta.get(sector_name, {}).get("member_count", 0)
        rows[sector_name] = result

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame.from_dict(rows, orient="index")
    # 把 phase 排成展示顺序: accel_up -> decel_up -> decel_down -> accel_down -> neutral
    phase_order = {"accel_up": 0, "decel_up": 1, "decel_down": 2, "accel_down": 3, "neutral": 4}
    df["_phase_order"] = df["phase"].map(phase_order)
    df = df.sort_values(["_phase_order", "ROC_20"], ascending=[True, False])
    df = df.drop(columns=["_phase_order"])
    return df


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="板块轮动象限扫描")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2)
    parser.add_argument("--end", default=None, help="截止日 YYYY-MM-DD")
    args = parser.parse_args()

    df = scan_all_sectors_phase(level=args.level, end_date=args.end)
    if df.empty:
        print("\n[ERROR] 没有可分析的板块, 请确认 CASE-A 已经把数据落库\n")
        return

    print(f"\n{'='*78}")
    print(f"  申万 {'一' if args.level == 1 else '二'} 级板块 -- 轮动象限扫描")
    print(f"  截止日: {args.end or '最新'}")
    print(f"{'='*78}\n")

    show = df[["phase", "phase_desc", "ROC_20", "MA20_SLOPE", "MACD_HIST", "MA20_ACCEL"]].copy()
    show["ROC_20"]     = (show["ROC_20"] * 100).round(2)
    show["MA20_SLOPE"] = show["MA20_SLOPE"].round(1)
    show["MACD_HIST"]  = show["MACD_HIST"].round(2)
    show["MA20_ACCEL"] = show["MA20_ACCEL"].round(1)
    show.columns = ["phase", "描述", "ROC20(%)", "MA20斜率(%)", "MACD柱", "MA20加速度"]

    for phase, group in show.groupby("phase", sort=False):
        print(f"\n--- {phase} ({len(group)} 个) ---")
        print(group.to_string())


if __name__ == "__main__":
    main()
