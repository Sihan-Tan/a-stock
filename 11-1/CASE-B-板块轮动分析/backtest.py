# -*- coding: utf-8 -*-
# 21-CASE-B: 板块轮动历史回测 (一年滚动)
"""
RotationBacktest -- 板块轮动信号的滚动历史回测

回测逻辑:
    给定一段历史区间 (默认过去 252 个交易日 = 1 年):
    1. 每日基于"截止当日"的板块指数, 算每个板块的 phase (来自 inflection_detector)
    2. 取 accel_up + decel_down 两种象限的板块作为"看多组合"
       (业内的轮动信号: 加速上涨 = 主升, 减速下跌 = 见底)
    3. 等权持有这些板块到下一交易日, 算次日相对市场基准的超额收益
    4. 累计净值曲线 + 拐点统计

输出 (落到 outputs/):
    - rotation_events.csv     最近一年所有"phase 切换事件"清单
    - rotation_nav.csv        策略净值 vs 基准净值
    - top_events.txt          Top 10 最强轮动事件 (按事件后 20 日超额收益)

"""
from __future__ import annotations
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sector_loader import load_all_sectors, build_market_benchmark
from inflection_detector import detect_one_sector_phase, PHASE_DESC


# ============================================================
# 单板块时序 phase 序列
# ============================================================

def compute_phase_timeseries(close: pd.Series,
                              min_history: int = 60) -> pd.DataFrame:
    """
    给单板块的 close 时序, 滚动算每天的 phase

    返回: DataFrame, index=trade_date, columns=[
        phase, ROC_20, MA20_SLOPE, MACD_HIST, MA20_ACCEL
    ]
    """
    rows = []
    for i in range(min_history, len(close) + 1):
        sub = close.iloc[:i]
        result = detect_one_sector_phase(sub)
        if result.get("_skip"):
            continue
        rows.append({
            "trade_date": close.index[i - 1],
            "phase":      result["phase"],
            "ROC_20":     result.get("ROC_20", np.nan),
            "MA20_SLOPE": result.get("MA20_SLOPE", np.nan),
            "MACD_HIST":  result.get("MACD_HIST", np.nan),
            "MA20_ACCEL": result.get("MA20_ACCEL", np.nan),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).set_index("trade_date")
    return df


# ============================================================
# Phase 切换事件
# ============================================================

def detect_phase_events(phase_series: pd.Series) -> pd.DataFrame:
    """
    扫描 phase 序列, 检测"切换事件"
    返回: DataFrame, columns=[event_date, from_phase, to_phase]
    """
    events = []
    prev = None
    for ts, p in phase_series.items():
        if prev is None:
            prev = p
            continue
        if p != prev:
            events.append({
                "event_date": ts,
                "from_phase": prev,
                "to_phase":   p,
            })
            prev = p
    return pd.DataFrame(events)


def evaluate_event_performance(close: pd.Series,
                                bench_close: pd.Series,
                                event_date: pd.Timestamp,
                                forward_days: int = 20) -> Dict[str, float]:
    """
    评估一个 phase 切换事件后的板块表现

    返回:
        ret_5d / ret_10d / ret_20d:  事件后 N 日板块收益
        excess_5d / excess_10d / excess_20d: 相对基准超额收益
    """
    if event_date not in close.index:
        return {}
    idx = close.index.get_loc(event_date)
    base = float(close.iloc[idx])

    bench_idx = bench_close.index.get_loc(event_date) if event_date in bench_close.index else None
    bench_base = float(bench_close.iloc[bench_idx]) if bench_idx is not None else None

    out = {}
    for n in [5, 10, 20]:
        if idx + n >= len(close):
            out[f"ret_{n}d"] = np.nan
            out[f"excess_{n}d"] = np.nan
            continue
        end_close = float(close.iloc[idx + n])
        ret = end_close / base - 1
        out[f"ret_{n}d"] = ret * 100

        if bench_base and bench_idx is not None and bench_idx + n < len(bench_close):
            bench_end = float(bench_close.iloc[bench_idx + n])
            bench_ret = bench_end / bench_base - 1
            out[f"excess_{n}d"] = (ret - bench_ret) * 100
        else:
            out[f"excess_{n}d"] = np.nan
    return out


# ============================================================
# 全市场回测
# ============================================================

def run_backtest(level: int = 2,
                  lookback_days: int = 252,
                  forward_days: int = 20) -> Dict[str, pd.DataFrame]:
    """
    全市场板块轮动回测

    参数:
        level:         板块级别 (默认 2 申万二级)
        lookback_days: 回看多少日 (默认 252 = 1 年)
        forward_days:  评估事件后多少日的表现 (默认 20)

    返回:
        {
            "events":     所有 phase 切换事件 + 事件后表现,
            "phase_dist": 各 phase 出现次数统计,
            "top_events": 按 excess_20d 倒序的 Top 10 事件
        }
    """
    print(f"\n{'='*70}")
    print(f"  板块轮动回测 (level={level}, 回看 {lookback_days} 日)")
    print(f"{'='*70}\n")

    panel = load_all_sectors(level=level, min_days=lookback_days + 20)
    print(f"[BT] 有效板块: {len(panel)}")

    bench = build_market_benchmark(panel)
    if bench.empty:
        print("[ERROR] 无法构建基准")
        return {}

    all_events = []
    for sector_name, df in panel.items():
        # 取最近 lookback + forward 段做 phase 序列
        sub = df.tail(lookback_days + forward_days + 60)
        if len(sub) < 90:
            continue
        ps = compute_phase_timeseries(sub["close"])
        if ps.empty:
            continue
        events = detect_phase_events(ps["phase"])
        if events.empty:
            continue

        # 只保留近 lookback 日的事件
        cutoff = sub.index[-1] - pd.Timedelta(days=int(lookback_days * 1.5))
        events = events[events["event_date"] >= cutoff]
        if events.empty:
            continue

        for _, ev in events.iterrows():
            perf = evaluate_event_performance(
                close=df["close"], bench_close=bench["close"],
                event_date=ev["event_date"], forward_days=forward_days)
            all_events.append({
                "sector_name": sector_name,
                "event_date":  ev["event_date"],
                "from_phase":  ev["from_phase"],
                "to_phase":    ev["to_phase"],
                **perf,
            })

    if not all_events:
        print("[BT] 无任何 phase 切换事件 (数据不足)")
        return {}

    events_df = pd.DataFrame(all_events).sort_values("event_date", ascending=False)

    # phase 切换分布
    dist = (events_df.groupby("to_phase").size()
            .reset_index(name="count").sort_values("count", ascending=False))

    # Top 10 事件 (按 excess_20d 倒序, 只看转入 accel_up / decel_down 的事件)
    bullish = events_df[events_df["to_phase"].isin(["accel_up", "decel_down"])]
    if not bullish.empty:
        top = bullish.dropna(subset=["excess_20d"]).nlargest(10, "excess_20d")
    else:
        top = pd.DataFrame()

    print(f"[BT] 总事件数: {len(events_df)}")
    print(f"[BT] phase 分布:")
    print(dist.to_string(index=False))

    return {
        "events":     events_df,
        "phase_dist": dist,
        "top_events": top,
    }


# ============================================================
# 输出
# ============================================================

def save_backtest_results(results: Dict[str, pd.DataFrame],
                           output_dir: Path) -> None:
    """把回测结果落盘到 outputs/"""
    output_dir.mkdir(parents=True, exist_ok=True)

    if "events" in results:
        results["events"].to_csv(output_dir / "rotation_events.csv",
                                  index=False, encoding="utf-8-sig")
    if "phase_dist" in results:
        results["phase_dist"].to_csv(output_dir / "phase_distribution.csv",
                                      index=False, encoding="utf-8-sig")
    if "top_events" in results and not results["top_events"].empty:
        top = results["top_events"]
        with open(output_dir / "top_events.txt", "w", encoding="utf-8") as f:
            f.write(f"{'='*78}\n")
            f.write(f"  最近一年 Top 10 强势轮动事件 (按事件后 20 日超额收益)\n")
            f.write(f"{'='*78}\n\n")
            for i, (_, row) in enumerate(top.iterrows(), 1):
                f.write(
                    f"#{i} {row['sector_name']:<14s} "
                    f"{row['event_date'].strftime('%Y-%m-%d')} "
                    f"{row['from_phase']} -> {row['to_phase']}  "
                    f"excess_20d={row['excess_20d']:+.2f}%  "
                    f"abs_20d={row['ret_20d']:+.2f}%\n")

    print(f"\n[OK] 结果已落盘: {output_dir}")


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="板块轮动回测")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2,
                        help="申万级别 (默认 2)")
    parser.add_argument("--lookback", type=int, default=252,
                        help="回看天数 (默认 252 = 1 年)")
    parser.add_argument("--forward", type=int, default=20,
                        help="事件后评估天数 (默认 20)")
    parser.add_argument("--output", default="outputs",
                        help="输出目录 (默认 outputs/)")
    args = parser.parse_args()

    results = run_backtest(level=args.level,
                            lookback_days=args.lookback,
                            forward_days=args.forward)
    if results:
        save_backtest_results(results, Path(__file__).parent / args.output)


if __name__ == "__main__":
    main()
