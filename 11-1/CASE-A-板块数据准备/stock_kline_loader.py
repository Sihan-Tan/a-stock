# -*- coding: utf-8 -*-
# 21-CASE-A: 个股 K 线下载到 trade_stock_daily
"""
StockKlineLoader -- 个股日 K 增量下载

策略:
    1. 从 trade_stock_status 取所有需要下载的股票 (即所有有申万分类的股票)
    2. 调 xtdata.download_history_data 触发 QMT 增量下载
    3. xtdata.get_market_data_ex 取出 K 线
    4. 写入 trade_stock_daily, INSERT ... ON DUPLICATE KEY UPDATE 实现增量
"""
from __future__ import annotations
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_config import execute_query, execute_many
from dotenv import dotenv_values

_env = dotenv_values(Path(__file__).parent / '.env')
DEFAULT_START = _env.get("SW_INIT_START_DATE", "20240101")
DOWNLOAD_BATCH = int(_env.get("SW_DOWNLOAD_BATCH", "200"))


# ============================================================
# 收集要下载的股票
# ============================================================

def list_target_stocks(level: int = 2) -> List[str]:
    """
    从 trade_stock_status 取所有有申万 N 级分类的股票
    """
    field = "sector_1" if level == 1 else "sector_2"
    rows = execute_query(
        f"SELECT DISTINCT stock_code FROM trade_stock_status "
        f"WHERE {field} IS NOT NULL ORDER BY stock_code")
    return [r["stock_code"] for r in rows]


def get_last_trade_date(stock_code: str) -> Optional[date]:
    """查某股在 trade_stock_daily 中的最大 trade_date, 决定增量起点"""
    rows = execute_query(
        "SELECT MAX(trade_date) AS d FROM trade_stock_daily WHERE stock_code = %s",
        (stock_code,))
    if rows and rows[0]["d"]:
        return rows[0]["d"]
    return None


# ============================================================
# 下载 + 写入
# ============================================================

def download_and_save_one(stock_code: str, start_yyyymmdd: str) -> int:
    """
    下载并写入一只股票的 K 线
        start_yyyymmdd: 起始日, 'YYYYMMDD'
    返回: 写入行数
    """
    from xtquant import xtdata
    try:
        xtdata.download_history_data(stock_code, period="1d", start_time=start_yyyymmdd)
    except Exception as e:
        print(f"  [WARN] {stock_code} 下载失败: {e}")
        return 0

    data = xtdata.get_market_data_ex(
        field_list=["open", "high", "low", "close", "volume", "amount"],
        stock_list=[stock_code], period="1d",
        start_time=start_yyyymmdd,
        dividend_type="back",
    )
    df = data.get(stock_code)
    if df is None or df.empty:
        return 0

    rows = []
    for idx, row in df.iterrows():
        trade_date = pd.to_datetime(str(idx)).date()
        # 换手率 xtdata 不直接给, 教学版填 NULL (实际由 trade_stock_status.float_shares 估算, 这里略)
        rows.append((
            stock_code, trade_date,
            float(row["open"])  if pd.notna(row["open"])  else None,
            float(row["high"])  if pd.notna(row["high"])  else None,
            float(row["low"])   if pd.notna(row["low"])   else None,
            float(row["close"]) if pd.notna(row["close"]) else None,
            int(row["volume"])  if pd.notna(row["volume"]) else 0,
            float(row["amount"]) if pd.notna(row["amount"]) else 0.0,
        ))

    sql = """
        INSERT INTO trade_stock_daily
            (stock_code, trade_date, open_price, high_price, low_price,
             close_price, volume, amount)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open_price  = VALUES(open_price),
            high_price  = VALUES(high_price),
            low_price   = VALUES(low_price),
            close_price = VALUES(close_price),
            volume      = VALUES(volume),
            amount      = VALUES(amount)
    """
    return execute_many(sql, rows)


def sync_all_kline(start_date: Optional[str] = None,
                   incremental: bool = True,
                   level: int = 2,
                   stock_filter: Optional[List[str]] = None) -> int:
    """
    全量或增量同步个股 K 线

    参数:
        start_date:    全量时的起始日 'YYYYMMDD', 默认读 .env
        incremental:   True=按每股最大 trade_date+1 增量; False=全部从 start_date 拉
        level:         决定从哪一级取股票池 (默认 2 申万二级, 覆盖全市场)
        stock_filter:  指定股票, None=全部
    返回: 累计写入行数
    """
    print(f"\n{'='*70}")
    print(f"  同步个股 K 线到 trade_stock_daily")
    print(f"  模式: {'增量' if incremental else '全量'}, 起始日: {start_date or DEFAULT_START}")
    print(f"{'='*70}\n")

    codes = stock_filter or list_target_stocks(level=level)
    print(f"[KLINE] 待同步股票: {len(codes)} 只\n")

    fallback_start = start_date or DEFAULT_START
    today_str = date.today().strftime("%Y%m%d")
    total_rows = 0
    t0 = time.time()

    for i, code in enumerate(codes, 1):
        if incremental:
            last = get_last_trade_date(code)
            start = (last + timedelta(days=1)).strftime("%Y%m%d") if last else fallback_start
        else:
            start = fallback_start

        if start > today_str:
            continue

        n = download_and_save_one(code, start)
        total_rows += n

        if i % DOWNLOAD_BATCH == 0:
            elapsed = time.time() - t0
            print(f"  ... 进度 {i:>4}/{len(codes)}  累计写入 {total_rows} 行  耗时 {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\n[OK] 完成, 累计写入 {total_rows} 行, 总耗时 {elapsed:.1f}s")
    return total_rows


# ============================================================
# 查询接口 (CASE-C 多因子选股复用)
# ============================================================

def load_stock_kline(stock_code: str,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> pd.DataFrame:
    """从 trade_stock_daily 加载单股 K 线, 返回 DataFrame (index=trade_date)"""
    conditions = ["stock_code = %s"]
    params: list = [stock_code]
    if start_date:
        conditions.append("trade_date >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= %s")
        params.append(end_date)

    sql = f"""
        SELECT trade_date, open_price, high_price, low_price, close_price, volume, amount
        FROM trade_stock_daily
        WHERE {' AND '.join(conditions)}
        ORDER BY trade_date ASC
    """
    rows = execute_query(sql, params)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.set_index("trade_date", inplace=True)
    df.columns = ["open", "high", "low", "close", "volume", "amount"]
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_multi_stock_kline(stock_codes: List[str],
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None) -> dict:
    """批量加载, 返回 {stock_code: DataFrame}"""
    result = {}
    for code in stock_codes:
        df = load_stock_kline(code, start_date, end_date)
        if not df.empty:
            result[code] = df
    return result


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="个股 K 线同步")
    parser.add_argument("--mode", choices=["init", "daily"], default="daily",
                        help="init=全量初始化, daily=每日增量")
    parser.add_argument("--start", default=None,
                        help="全量起始日 YYYYMMDD, 默认读 .env")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2,
                        help="按哪一级板块取股票池 (默认 2)")
    args = parser.parse_args()

    sync_all_kline(start_date=args.start,
                   incremental=(args.mode == "daily"),
                   level=args.level)


if __name__ == "__main__":
    main()
