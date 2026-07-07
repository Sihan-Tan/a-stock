# -*- coding: utf-8 -*-
# 21-CASE-B: 板块指数加载器 (从 CASE-A 落库的 trade_sector_daily 读)
"""
SectorLoader -- 板块指数 K 线加载

依赖 CASE-A 落库的 2 张表 (与 WucaiTrade 同名表对齐):
    trade_stock_status   股票状态 (含 sector_2 申万二级映射)
    trade_sector_daily   板块每日聚合 + 等权合成 OHLC

主用申万二级 (sector_level=2), 想跑一级在所有 API 里传 level=1
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_config import execute_query


# ============================================================
# 板块清单
# ============================================================

def list_sectors(level: int = 2) -> List[Dict]:
    """
    列出某级别全部板块 (从 trade_stock_status 反查)

    返回: [{sector_name, member_count}, ...]
    """
    field = "sector_1" if level == 1 else "sector_2"
    sql = f"""
        SELECT {field} AS sector_name, COUNT(*) AS member_count
        FROM trade_stock_status
        WHERE {field} IS NOT NULL
        GROUP BY {field}
        ORDER BY {field}
    """
    return execute_query(sql)


# ============================================================
# 单板块指数加载
# ============================================================

def load_sector_index(sector_name: str,
                       level: int = 2,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
    """
    加载单个板块的指数 K 线 (来自 trade_sector_daily 的合成 OHLC)

    返回: DataFrame, index=trade_date, columns=[
        open, high, low, close, volume, amount, change_pct,
        stock_count, kline_stock_count
    ]
    """
    conditions = ["sector_name = %s", "sector_level = %s"]
    params: list = [sector_name, level]
    if start_date:
        conditions.append("trade_date >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= %s")
        params.append(end_date)

    sql = f"""
        SELECT trade_date,
               open_idx, high_idx, low_idx, close_idx,
               total_volume, total_amount, change_pct,
               stock_count, kline_stock_count
        FROM trade_sector_daily
        WHERE {' AND '.join(conditions)}
        ORDER BY trade_date ASC
    """
    rows = execute_query(sql, params)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.set_index("trade_date", inplace=True)
    df.rename(columns={
        "open_idx":     "open",
        "high_idx":     "high",
        "low_idx":      "low",
        "close_idx":    "close",
        "total_volume": "volume",
        "total_amount": "amount",
    }, inplace=True)
    for col in ["open", "high", "low", "close", "amount", "change_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_all_sectors(level: int = 2,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      min_days: int = 60) -> Dict[str, pd.DataFrame]:
    """
    一次性加载某级别全部板块的指数 K 线

    参数:
        min_days: 历史数据少于这个天数的板块过滤掉 (避免新上市/数据不足)
    返回: {sector_name: DataFrame}
    """
    sectors = list_sectors(level=level)
    result: Dict[str, pd.DataFrame] = {}
    for s in sectors:
        df = load_sector_index(s["sector_name"], level=level,
                                start_date=start_date, end_date=end_date)
        if not df.empty and len(df) >= min_days:
            result[s["sector_name"]] = df
    return result


# ============================================================
# 市场基准 (用所有板块等权代理"全市场")
# ============================================================

def build_market_benchmark(sector_panel: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    用某级别全部板块等权合成"市场基准"
    申万二级 134 个板块的等权 ≈ 全市场风格中性的近似基准

    返回: DataFrame, columns=[close, amount], 与板块同周期
    """
    if not sector_panel:
        return pd.DataFrame()

    closes  = pd.DataFrame()
    amounts = pd.DataFrame()
    for name, df in sector_panel.items():
        if df["close"].iloc[0] > 0:
            closes[name] = df["close"] / df["close"].iloc[0]
        amounts[name] = df["amount"]

    return pd.DataFrame({
        "close":  closes.mean(axis=1) * 1000,
        "amount": amounts.sum(axis=1),
    })
