# -*- coding: utf-8 -*-
# 21-CASE-B: 板块强度排名 (基础三指标版本)
"""
IndustryStrength -- 申万板块强度排名 (静态横截面打分)

理论背景:
    板块轮动是 A 股最稳定的"风格收益"之一. 在不同的宏观环境下, 资金会在不同
    行业间循环 (例: 牛市初期金融 -> 中期成长 -> 末期资源 -> 熊市消费防御).

    经典理论:
        - 美林时钟 (Merrill Lynch Clock): 复苏 -> 过热 -> 滞胀 -> 衰退 4 个阶段
        - 行业景气度: 看下游需求 + 上游成本 + 政策催化
        - 动量效应: 强者恒强, 强势行业容易延续 1-3 个月

本模块用 3 个简单可计算的"水平指标"做横截面排名:

    1. 21 日动量 (MOM_21)
        过去 21 日 (1 个月) 板块指数收益率
        含义: 短期资金流向

    2. 60 日相对强度 (RS_60)
        板块 60 日收益 - 市场基准 60 日收益
        含义: 中期跑赢市场的程度

    3. 量价配合 (VOL_RATIO)
        近 5 日成交额 / 近 60 日成交额
        含义: 是否伴随放量 (有量才是真涨)

合成: 三者各 Z-score 后等权 -> 综合行业强度分

注意: 本模块只看 "当前是不是强", 不能告诉你 "现在是加速还是减速"
      想看趋势的速度和加速度, 用 derivatives.py
      想检测拐点, 用 inflection_detector.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sector_loader import (load_all_sectors, list_sectors,
                            build_market_benchmark)


# ============================================================
# 三个强度指标
# ============================================================

def calc_strength_indicators(industry_index: pd.DataFrame,
                              benchmark_index: pd.DataFrame) -> Dict[str, float]:
    """
    给定单板块指数 + 市场基准, 算 3 个静态强度指标
    """
    if len(industry_index) < 70 or len(benchmark_index) < 70:
        return {}

    # 对齐时间索引
    common = industry_index.index.intersection(benchmark_index.index)
    if len(common) < 70:
        return {}
    ind = industry_index.loc[common]
    bench = benchmark_index.loc[common]

    # MOM_21: 21 日累计收益
    mom_21 = ind["close"].iloc[-1] / ind["close"].iloc[-22] - 1

    # RS_60: 60 日相对强度 (行业 60 日收益 - 基准 60 日收益)
    ind_ret = ind["close"].iloc[-1] / ind["close"].iloc[-61] - 1
    bench_ret = bench["close"].iloc[-1] / bench["close"].iloc[-61] - 1
    rs_60 = ind_ret - bench_ret

    # VOL_RATIO: 近 5 日成交额 / 近 60 日成交额
    recent_vol = ind["amount"].iloc[-5:].mean()
    long_vol = ind["amount"].iloc[-60:].mean()
    vol_ratio = recent_vol / long_vol if long_vol > 0 else np.nan

    return {
        "MOM_21":    float(mom_21),
        "RS_60":     float(rs_60),
        "VOL_RATIO": float(vol_ratio),
    }


# ============================================================
# 完整 pipeline
# ============================================================

def rank_industries(level: int = 2,
                     end_date: Optional[str] = None,
                     lookback_days: int = 90) -> pd.DataFrame:
    """
    给申万 N 级所有板块排名

    参数:
        level:        1=一级板块, 2=二级板块 (默认 2)
        end_date:     回溯截止日 'YYYY-MM-DD', None=最新
        lookback_days: 至少需要多少日数据才参与排名 (默认 90)

    返回: DataFrame, index=sector_name, columns=[
        MOM_21, RS_60, VOL_RATIO,
        MOM_21_z, RS_60_z, VOL_RATIO_z, score, rank, member_count
    ]
    """
    print(f"[STRENGTH] 加载申万 {'一' if level == 1 else '二'} 级所有板块指数 (level={level}) ...")
    sector_panel = load_all_sectors(level=level, end_date=end_date)
    print(f"[STRENGTH] 有效板块: {len(sector_panel)}")

    if not sector_panel:
        return pd.DataFrame()

    # 合成市场基准 = 全部板块等权
    bench = build_market_benchmark(sector_panel)
    if bench.empty:
        return pd.DataFrame()

    # 板块成员数 (从 trade_stock_status 反查)
    sectors_meta = {s["sector_name"]: s for s in list_sectors(level=level)}

    rows: Dict[str, Dict] = {}
    for sector_name, df in sector_panel.items():
        df_window = df.tail(max(lookback_days, 70))
        if len(df_window) < 70:
            continue

        bench_window = bench.loc[bench.index.intersection(df_window.index)]
        ind_dict = calc_strength_indicators(df_window, bench_window)
        if not ind_dict:
            continue

        meta = sectors_meta.get(sector_name, {})
        ind_dict["member_count"] = meta.get("member_count", 0)
        rows[sector_name] = ind_dict

    df_ind = pd.DataFrame.from_dict(rows, orient="index").dropna(
        subset=["MOM_21", "RS_60", "VOL_RATIO"])

    # 综合评分: 三个指标 Z-score 后等权
    for col in ["MOM_21", "RS_60", "VOL_RATIO"]:
        mu = df_ind[col].mean()
        sd = df_ind[col].std(ddof=1)
        df_ind[f"{col}_z"] = (df_ind[col] - mu) / sd if sd > 0 else 0.0
    df_ind["score"] = df_ind[["MOM_21_z", "RS_60_z", "VOL_RATIO_z"]].mean(axis=1)
    df_ind["rank"]  = df_ind["score"].rank(ascending=False, method="min").astype(int)

    return df_ind.sort_values("score", ascending=False)


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="申万板块强度排名")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2,
                        help="1=申万一级, 2=申万二级 (默认)")
    parser.add_argument("--lookback", type=int, default=90, help="回看天数 (默认 90)")
    parser.add_argument("--end", default=None, help="截止日 YYYY-MM-DD")
    args = parser.parse_args()

    df = rank_industries(level=args.level, end_date=args.end, lookback_days=args.lookback)
    if df.empty:
        print("\n[ERROR] 没有可排名的板块, 请确认 CASE-A 已经把数据落库\n")
        return

    print(f"\n{'='*78}")
    print(f"  申万 {'一' if args.level == 1 else '二'} 级板块强度排名 "
          f"(截止 {args.end or '最新'}, 回看 {args.lookback} 日)")
    print(f"{'='*78}\n")

    show = df[["MOM_21", "RS_60", "VOL_RATIO", "score", "rank", "member_count"]].copy()
    show["MOM_21"]    = (show["MOM_21"]    * 100).round(2)
    show["RS_60"]     = (show["RS_60"]     * 100).round(2)
    show["VOL_RATIO"] = show["VOL_RATIO"].round(2)
    show["score"]     = show["score"].round(3)
    show.columns = ["MOM_21(%)", "RS_60(%)", "VOL_R", "Score", "Rank", "成员数"]

    print("[Top 5 强势板块]")
    print(show.head(5).to_string())
    print(f"\n[Bottom 5 弱势板块]")
    print(show.tail(5).to_string())


if __name__ == "__main__":
    main()
