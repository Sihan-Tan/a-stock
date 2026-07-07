# -*- coding: utf-8 -*-
# 21-CASE-A: 板块每日聚合 + 等权合成指数 (写入 trade_sector_daily)
"""
SectorIndexBuilder -- 板块每日聚合 + 等权合成指数

为什么用累乘而非简单 close 等权均值:
    1. 对停牌/退市鲁棒: 前一日 close_idx 已经吸收了, 不会因成员减少出现跳空
    2. 业内多数行业指数 (中证 / 申万官方) 都用类似累乘思路
    3. 增量更新友好: 只要算出当日的"成份股均收益"就够了, 不需要回看历史
"""
from __future__ import annotations
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_config import execute_query, execute_many, execute_update
from industry_meta import list_sectors_from_db, get_sector_member_codes


BASE_INDEX = 1000.0


# ============================================================
# 工具
# ============================================================

def get_all_trade_dates() -> List[date]:
    """从 trade_stock_daily 查所有出现过的交易日 (升序)"""
    rows = execute_query(
        "SELECT DISTINCT trade_date FROM trade_stock_daily ORDER BY trade_date ASC")
    return [r["trade_date"] for r in rows]


def get_prev_close_idx(sector_name: str, sector_level: int,
                        prev_trade_date: date) -> float:
    """从 trade_sector_daily 取某板块前一交易日的 close_idx"""
    rows = execute_query(
        "SELECT close_idx FROM trade_sector_daily "
        "WHERE sector_name = %s AND sector_level = %s AND trade_date = %s",
        (sector_name, sector_level, prev_trade_date))
    if rows and rows[0]["close_idx"] is not None:
        return float(rows[0]["close_idx"])
    return BASE_INDEX


def fetch_member_kline_panel(member_codes: List[str],
                              trade_dates: List[date]) -> Dict[Tuple[str, date], Dict]:
    """
    一次性批量拉成分股在指定交易日的 K 线
    返回: {(stock_code, trade_date): {open, high, low, close, volume, amount}}
    """
    if not member_codes or not trade_dates:
        return {}
    ph_codes = ",".join(["%s"] * len(member_codes))
    ph_dates = ",".join(["%s"] * len(trade_dates))
    sql = f"""
        SELECT stock_code, trade_date, open_price, high_price, low_price,
               close_price, volume, amount
        FROM trade_stock_daily
        WHERE stock_code IN ({ph_codes}) AND trade_date IN ({ph_dates})
    """
    params = list(member_codes) + list(trade_dates)
    rows = execute_query(sql, params)
    result: Dict[Tuple[str, date], Dict] = {}
    for r in rows:
        result[(r["stock_code"], r["trade_date"])] = {
            "open":   float(r["open_price"])  if r["open_price"]  is not None else None,
            "high":   float(r["high_price"])  if r["high_price"]  is not None else None,
            "low":    float(r["low_price"])   if r["low_price"]   is not None else None,
            "close":  float(r["close_price"]) if r["close_price"] is not None else None,
            "volume": int(r["volume"])        if r["volume"]      is not None else 0,
            "amount": float(r["amount"])      if r["amount"]      is not None else 0.0,
        }
    return result


# ============================================================
# 单板块单日聚合
# ============================================================

def aggregate_sector_one_day(sector_name: str, sector_level: int,
                              member_codes: List[str],
                              trade_date: date,
                              prev_trade_date: Optional[date],
                              prev_close_idx: float,
                              kline_panel: Dict[Tuple[str, date], Dict]) -> Optional[Dict]:
    """
    给定一个交易日, 计算单板块的聚合统计 + 合成 OHLC

    返回: dict 含所有 trade_sector_daily 字段; 若有效成分股不足 1 只则返回 None
    """
    pcts: List[Tuple[str, Dict, Dict]] = []   # (code, today_row, prev_row)
    for code in member_codes:
        today = kline_panel.get((code, trade_date))
        if not today or today["close"] is None:
            continue
        prev = kline_panel.get((code, prev_trade_date)) if prev_trade_date else None
        pcts.append((code, today, prev))

    if not pcts:
        return None

    # 1. 涨跌幅 (要前日 close)
    rise = fall = flat = lu = ld = 0
    pct_list: List[Tuple[str, float]] = []
    for code, today, prev in pcts:
        if not prev or not prev["close"]:
            continue
        pct = today["close"] / prev["close"] - 1
        pct_list.append((code, pct))
        if pct > 0.0001:
            rise += 1
        elif pct < -0.0001:
            fall += 1
        else:
            flat += 1
        if pct >= 0.097:
            lu += 1
        elif pct <= -0.097:
            ld += 1

    change_pct = float(np.mean([p for _, p in pct_list])) * 100 if pct_list else 0.0

    # 2. 领涨股
    if pct_list:
        top_code, top_pct = max(pct_list, key=lambda x: x[1])
    else:
        top_code, top_pct = "", 0.0

    # 3. 总成交量/额
    total_volume = sum(t["volume"] for _, t, _ in pcts)
    total_amount = sum(t["amount"] for _, t, _ in pcts)

    # 4. OHLC 指数 (累乘)
    open_rets, high_rets, low_rets, close_rets = [], [], [], []
    for code, today, prev in pcts:
        if not prev or not prev["close"]:
            continue
        if today["open"] is not None:
            open_rets.append(today["open"]  / prev["close"] - 1)
        if today["high"] is not None:
            high_rets.append(today["high"]  / prev["close"] - 1)
        if today["low"] is not None:
            low_rets.append(today["low"]   / prev["close"] - 1)
        close_rets.append(today["close"] / prev["close"] - 1)

    if close_rets:
        open_idx  = prev_close_idx * (1 + np.mean(open_rets))  if open_rets  else prev_close_idx
        high_idx  = prev_close_idx * (1 + np.mean(high_rets))  if high_rets  else prev_close_idx
        low_idx   = prev_close_idx * (1 + np.mean(low_rets))   if low_rets   else prev_close_idx
        close_idx = prev_close_idx * (1 + np.mean(close_rets))
    else:
        # 首日 (没有前日 close): 用 BASE_INDEX
        open_idx = high_idx = low_idx = close_idx = prev_close_idx

    return {
        "sector_name":       sector_name,
        "sector_level":      sector_level,
        "trade_date":        trade_date,
        "change_pct":        round(change_pct, 4),
        "stock_count":       len(member_codes),
        "rise_count":        rise,
        "fall_count":        fall,
        "flat_count":        flat,
        "limit_up":          lu,
        "limit_down":        ld,
        "top_stock":         top_code,
        "top_stock_name":    top_code,
        "top_stock_pct":     round(top_pct * 100, 2),
        "open_idx":          round(open_idx, 4),
        "high_idx":          round(high_idx, 4),
        "low_idx":           round(low_idx, 4),
        "close_idx":         round(close_idx, 4),
        "total_volume":      total_volume,
        "total_amount":      round(total_amount, 2),
        "avg_turnover":      0.0,
        "kline_stock_count": len(pcts),
    }


def save_sector_rows(rows: List[Dict]) -> int:
    """批量写入 trade_sector_daily, ON DUPLICATE KEY UPDATE"""
    if not rows:
        return 0
    sql = """
        INSERT INTO trade_sector_daily
            (sector_name, sector_level, trade_date, change_pct, stock_count,
             rise_count, fall_count, flat_count, limit_up, limit_down,
             top_stock, top_stock_name, top_stock_pct,
             open_idx, high_idx, low_idx, close_idx,
             total_volume, total_amount, avg_turnover, kline_stock_count)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON DUPLICATE KEY UPDATE
            change_pct = VALUES(change_pct),
            stock_count = VALUES(stock_count),
            rise_count = VALUES(rise_count),
            fall_count = VALUES(fall_count),
            flat_count = VALUES(flat_count),
            limit_up = VALUES(limit_up),
            limit_down = VALUES(limit_down),
            top_stock = VALUES(top_stock),
            top_stock_name = VALUES(top_stock_name),
            top_stock_pct = VALUES(top_stock_pct),
            open_idx = VALUES(open_idx),
            high_idx = VALUES(high_idx),
            low_idx = VALUES(low_idx),
            close_idx = VALUES(close_idx),
            total_volume = VALUES(total_volume),
            total_amount = VALUES(total_amount),
            avg_turnover = VALUES(avg_turnover),
            kline_stock_count = VALUES(kline_stock_count)
    """
    rows_tuples = [(
        r["sector_name"], r["sector_level"], r["trade_date"], r["change_pct"], r["stock_count"],
        r["rise_count"], r["fall_count"], r["flat_count"], r["limit_up"], r["limit_down"],
        r["top_stock"], r["top_stock_name"], r["top_stock_pct"],
        r["open_idx"], r["high_idx"], r["low_idx"], r["close_idx"],
        r["total_volume"], r["total_amount"], r["avg_turnover"], r["kline_stock_count"],
    ) for r in rows]
    return execute_many(sql, rows_tuples)


# ============================================================
# 全量 / 增量
# ============================================================

def rebuild_one_sector(sector_name: str, sector_level: int,
                        target_dates: List[date],
                        all_dates: List[date]) -> int:
    """
    重算单个板块在指定日期范围内的所有 trade_sector_daily 行
    返回: 写入行数
    """
    members = get_sector_member_codes(sector_name, level=sector_level)
    if not members:
        return 0

    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    # 一次性把成分股在 [target_dates 的前一日 ~ target_dates] 上的 K 线全拉
    fetch_dates = set()
    for d in target_dates:
        fetch_dates.add(d)
        idx = date_to_idx.get(d)
        if idx is not None and idx > 0:
            fetch_dates.add(all_dates[idx - 1])
    panel = fetch_member_kline_panel(members, sorted(fetch_dates))

    # 顺序遍历, 维持 prev_close_idx 链
    rows = []
    prev_close_idx = None
    for d in target_dates:
        idx = date_to_idx.get(d)
        prev_d = all_dates[idx - 1] if idx and idx > 0 else None

        if prev_close_idx is None:
            prev_close_idx = get_prev_close_idx(sector_name, sector_level, prev_d) if prev_d else BASE_INDEX

        row = aggregate_sector_one_day(
            sector_name=sector_name, sector_level=sector_level,
            member_codes=members,
            trade_date=d, prev_trade_date=prev_d,
            prev_close_idx=prev_close_idx,
            kline_panel=panel)
        if row is None:
            continue
        rows.append(row)
        prev_close_idx = row["close_idx"]

    return save_sector_rows(rows)


def rebuild_all_sectors(level: int = 2,
                         days: Optional[int] = None,
                         full: bool = False) -> int:
    """
    重算所有板块的 trade_sector_daily

    参数:
        level: 1=申万一级, 2=申万二级 (默认 2)
        days:  增量重算最近 N 个交易日 (含 prev close 校正)
        full:  True=全量回填所有历史交易日
    返回: 累计写入行数
    """
    print(f"\n{'='*70}")
    print(f"  合成 trade_sector_daily (level={level}, "
          f"{'全量回填' if full else f'最近 {days or 30} 日'})")
    print(f"{'='*70}\n")

    all_dates = get_all_trade_dates()
    if not all_dates:
        print("[ERROR] trade_stock_daily 没有任何数据, 请先跑 stock_kline_loader.py")
        return 0

    if full:
        target_dates = all_dates
    else:
        n = days or 30
        target_dates = all_dates[-n:]

    sectors = list_sectors_from_db(level=level)
    print(f"[INDEX] 待重算板块: {len(sectors)}, 日期范围: "
          f"{target_dates[0]} ~ {target_dates[-1]} ({len(target_dates)} 个交易日)\n")

    total = 0
    for i, sector in enumerate(sectors, 1):
        n = rebuild_one_sector(sector, level, target_dates, all_dates)
        total += n
        if i % 20 == 0 or i == len(sectors):
            print(f"  [{i:>3}/{len(sectors)}] {sector:<20} 写入 {n:>4} 行  累计 {total}")

    print(f"\n[OK] trade_sector_daily 合成完成, 累计写入 {total} 行\n")
    return total


# ============================================================
# 查询接口 (CASE-B 用)
# ============================================================

def load_sector_index(sector_name: str, sector_level: int = 2,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
    """
    从 trade_sector_daily 加载单板块的指数 OHLC + 量价指标
    返回: DataFrame, index=trade_date, 含 [open, high, low, close, volume, amount,
                                            change_pct, stock_count, kline_stock_count]
    """
    conditions = ["sector_name = %s", "sector_level = %s"]
    params: list = [sector_name, sector_level]
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


def load_all_sector_index(level: int = 2,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """
    一次性加载某级别所有板块的指数 K 线
    返回: {sector_name: DataFrame}
    """
    sectors = list_sectors_from_db(level=level)
    result = {}
    for s in sectors:
        df = load_sector_index(s, sector_level=level, start_date=start_date, end_date=end_date)
        if not df.empty and len(df) > 60:
            result[s] = df
    return result


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="合成 trade_sector_daily")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2)
    parser.add_argument("--days", type=int, default=30,
                        help="增量重算最近 N 个交易日 (默认 30)")
    parser.add_argument("--full", action="store_true",
                        help="全量回填所有历史交易日")
    args = parser.parse_args()

    rebuild_all_sectors(level=args.level, days=args.days, full=args.full)


if __name__ == "__main__":
    main()
