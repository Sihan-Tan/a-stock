# -*- coding: utf-8 -*-
# 21-CASE-A: 个股 K 线下载到 trade_stock_daily
"""
StockKlineLoader -- 个股日 K 增量下载 (双复权: 前复权 + 后复权)

策略:
    1. 从 trade_stock_status 取所有需要下载的股票 (即所有有申万分类的股票)
    2. 调 xtdata.download_history_data 触发 QMT 增量下载
    3. xtdata.get_market_data_ex 分别读取前复权 / 后复权 K 线
    4. 多线程并行写入 trade_stock_daily, INSERT ... ON DUPLICATE KEY UPDATE 实现增量

列语义:
    open_price  / high_price  / low_price  / close_price   → 前复权 (dividend_type='front')
    open_price_b/ high_price_b/ low_price_b/ close_price_b → 后复权 (dividend_type='back')
    volume / amount / turnover_rate → 共享字段 (与复权无关)
"""
from __future__ import annotations
import sys
import time
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_config import execute_query, execute_many
from dotenv import dotenv_values

_env = dotenv_values(Path(__file__).parent / '.env')
DEFAULT_START = _env.get("SW_INIT_START_DATE", "20240101")
DOWNLOAD_BATCH = int(_env.get("SW_DOWNLOAD_BATCH", "200"))  # 进度打印频率

# 并行下载配置
NUM_WORKERS = 8          # 并行下载线程数


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


# ============================================================
# 批量查询已有数据 (替换逐只 N+1 查询)
# ============================================================

def get_existing_latest_dates() -> Dict[str, str]:
    """
    一次性查出所有股票在 trade_stock_daily 中的最新交易日

    返回:
        {stock_code: 'YYYYMMDD'} 格式的映射字典
        如果某股票没有数据，则不会出现在返回结果中
    """
    rows = execute_query(
        "SELECT stock_code, MAX(trade_date) AS max_date "
        "FROM trade_stock_daily GROUP BY stock_code"
    )
    result = {}
    for r in rows:
        if r['max_date']:
            result[r['stock_code']] = r['max_date'].strftime('%Y%m%d')
    return result


def get_last_trade_date(stock_code: str) -> Optional[date]:
    """查某股在 trade_stock_daily 中的最大 trade_date (单只调试用)"""
    rows = execute_query(
        "SELECT MAX(trade_date) AS d FROM trade_stock_daily WHERE stock_code = %s",
        (stock_code,))
    if rows and rows[0]["d"]:
        return rows[0]["d"]
    return None


# ============================================================
# 换手率计算
# ============================================================

def _get_float_shares(stock_code: str) -> int:
    """
    从 xtdata.get_instrument_detail 获取流通股本 (股), 用于计算换手率

    返回:
        int: 流通股本数, 获取失败返回 0
    """
    from xtquant import xtdata
    try:
        detail = xtdata.get_instrument_detail(stock_code)
        if detail:
            neg = detail.get('NegotiableVolume') or detail.get('TotalVolume') or 0
            if neg > 0:
                return int(neg)
    except Exception:
        pass
    return 0


# ============================================================
# 双复权下载 + 写入 (单只股票)
# ============================================================

INSERT_SQL = """
    INSERT INTO trade_stock_daily
        (stock_code, trade_date,
         open_price, high_price, low_price, close_price,
         open_price_b, high_price_b, low_price_b, close_price_b,
         volume, amount, turnover_rate)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
        open_price   = VALUES(open_price),
        high_price   = VALUES(high_price),
        low_price    = VALUES(low_price),
        close_price  = VALUES(close_price),
        open_price_b = VALUES(open_price_b),
        high_price_b = VALUES(high_price_b),
        low_price_b  = VALUES(low_price_b),
        close_price_b= VALUES(close_price_b),
        volume       = VALUES(volume),
        amount       = VALUES(amount),
        turnover_rate= VALUES(turnover_rate)
"""


def download_and_save_one(stock_code: str, start_date: str) -> Tuple[str, int]:
    """
    下载一只股票的双复权 K 线并写入数据库

    参数:
        stock_code: 股票代码, 如 '600519.SH'
        start_date: 起始日, 格式 'YYYYMMDD'

    返回:
        (stock_code, row_count) 元组, row_count >= 0 为写入行数
    """
    from xtquant import xtdata

    # 1) 下载历史数据到本地缓存 (只需下载一次, 前后复权共享缓存)
    try:
        xtdata.download_history_data(stock_code, period="1d", start_time=start_date)
    except Exception as e:
        print(f"  [WARN] {stock_code} 下载失败: {e}")
        return stock_code, -1

    # 2) 读取前复权数据
    data_front = xtdata.get_market_data_ex(
        field_list=["open", "high", "low", "close", "volume", "amount"],
        stock_list=[stock_code], period="1d",
        start_time=start_date, dividend_type="front",
    )

    # 3) 读取后复权数据 (仅价格, volume/amount 用前复权的即可)
    data_back = xtdata.get_market_data_ex(
        field_list=["open", "high", "low", "close"],
        stock_list=[stock_code], period="1d",
        start_time=start_date, dividend_type="back",
    )

    df_front = data_front.get(stock_code)
    df_back = data_back.get(stock_code)
    if df_front is None or df_front.empty:
        return stock_code, 0

    # 4) 获取流通股本用于计算换手率
    float_shares = _get_float_shares(stock_code)

    # 5) 组装行数据: 按日期对齐前复权 + 后复权
    rows = []
    for idx, row in df_front.iterrows():
        idx_str = str(idx)
        if len(idx_str) < 8:
            continue
        trade_date = f"{idx_str[:4]}-{idx_str[4:6]}-{idx_str[6:8]}"

        # 前复权价格
        of = _nf(row["open"])
        hf = _nf(row["high"])
        lf = _nf(row["low"])
        cf = _nf(row["close"])

        # 成交量 / 成交额 (从前复权 DataFrame 取, 与复权无关)
        vol = int(row["volume"]) if pd.notna(row["volume"]) else 0
        amt = float(row["amount"]) if pd.notna(row["amount"]) else 0.0

        # 换手率: xtdata volume 单位是"手" (1手=100股)
        vol_shares = vol * 100
        turnover = round(vol_shares / float_shares * 100, 4) if float_shares > 0 and vol > 0 else None

        # 后复权价格 (从后复权 DataFrame 按相同日期索引查找)
        if df_back is not None and not df_back.empty and idx in df_back.index:
            back_row = df_back.loc[idx]
            ob = _nf(back_row["open"])
            hb = _nf(back_row["high"])
            lb = _nf(back_row["low"])
            cb = _nf(back_row["close"])
        else:
            ob = hb = lb = cb = None

        rows.append((
            stock_code, trade_date,
            of, hf, lf, cf,
            ob, hb, lb, cb,
            vol, amt, turnover,
        ))

    # 6) 批量写入
    if rows:
        n = execute_many(INSERT_SQL, rows)
        return stock_code, n
    return stock_code, 0


def _nf(v):
    """将 numpy nan 转为 None, MySQL 不接受 nan 值"""
    if v is None:
        return None
    try:
        if math.isnan(v):
            return None
    except TypeError:
        pass
    return float(v)


# ============================================================
# 全量 / 增量同步 (多线程并行)
# ============================================================

def sync_all_kline(start_date: Optional[str] = None,
                   incremental: bool = True,
                   level: int = 2,
                   stock_filter: Optional[List[str]] = None,
                   workers: int = NUM_WORKERS) -> int:
    """
    全量或增量同步个股 K 线 (双复权)

    参数:
        start_date:    全量时的起始日 'YYYYMMDD', 默认读 .env
        incremental:   True=按每股最大 trade_date+1 增量; False=全部从 start_date 拉
        level:         决定从哪一级取股票池 (默认 2 申万二级)
        stock_filter:  指定股票, None=全部
        workers:       并行线程数 (默认 NUM_WORKERS)

    返回:
        累计写入行数
    """
    print(f"\n{'='*70}")
    print(f"  同步个股 K 线到 trade_stock_daily (双复权)")
    print(f"  模式: {'增量' if incremental else '全量'}, 起始日: {start_date or DEFAULT_START}")
    print(f"{'='*70}\n")

    codes = stock_filter or list_target_stocks(level=level)
    print(f"[KLINE] 待同步股票: {len(codes)} 只\n")

    # 批量查询已有数据的最新日期
    existing = get_existing_latest_dates() if incremental else {}
    today_str = date.today().strftime("%Y%m%d")
    fallback_start = start_date or DEFAULT_START

    # 构建任务列表
    tasks: List[Tuple[str, str]] = []
    skip_count = 0
    for code in codes:
        if incremental:
            latest = existing.get(code)
            if latest and latest >= today_str:
                skip_count += 1
                continue
            start = (datetime.strptime(latest, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d") if latest else fallback_start
        else:
            start = fallback_start

        if start > today_str:
            continue
        tasks.append((code, start))

    if skip_count > 0:
        print(f"  跳过(已是最新交易日): {skip_count} 只")

    total = len(tasks)
    if total == 0:
        print("[KLINE] 全部已是最新, 无需更新\n")
        return 0

    total_rows = 0
    success_count = 0
    fail_list: List[str] = []
    t0 = time.time()

    # 下载执行
    if total <= 5 or workers <= 1:
        # 少量串行, 避免线程池开销
        for i, (code, start) in enumerate(tasks, 1):
            print(f"[{i}/{total}] {code} (从 {start} 开始)")
            _, count = download_and_save_one(code, start)
            if count >= 0:
                success_count += 1
                total_rows += count
            else:
                fail_list.append(code)
    else:
        # 多线程并行下载
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(download_and_save_one, c, s): c for c, s in tasks}
            done = 0
            for future in as_completed(futures):
                code, count = future.result()
                done += 1
                if count >= 0:
                    success_count += 1
                    total_rows += count
                else:
                    fail_list.append(code)

                # 进度显示
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / speed if speed > 0 else 0
                sys.stdout.write(
                    f"\r  进度 {done}/{total} ({done*100/total:.1f}%) | "
                    f"{speed:.1f} 只/秒 | 剩余约 {eta:.0f}秒 | "
                    f"成功 {success_count} 失败 {len(fail_list)}    "
                )
                sys.stdout.flush()
        print()  # 换行

    elapsed = time.time() - t0
    print(f"\n[OK] 完成, 成功 {success_count}/{total} 只, 累计写入 {total_rows} 行, 总耗时 {elapsed:.1f}s")
    if fail_list:
        print(f"  失败 {len(fail_list)} 只: {fail_list[:10]}{'...' if len(fail_list) > 10 else ''}")
    return total_rows


# ============================================================
# 查询接口 (CASE-C 多因子选股 + 板块索引构建复用)
# ============================================================

def load_stock_kline(stock_code: str,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     dividend_type: str = "front") -> pd.DataFrame:
    """
    从 trade_stock_daily 加载单股 K 线

    参数:
        stock_code:    股票代码
        start_date:    起始日 'YYYY-MM-DD'
        end_date:      结束日 'YYYY-MM-DD'
        dividend_type: 'front' 读前复权列, 'back' 读后复权列

    返回:
        DataFrame (index=trade_date), 列名: open/high/low/close/volume/amount
    """
    if dividend_type == "back":
        price_cols = "open_price_b, high_price_b, low_price_b, close_price_b"
    else:
        price_cols = "open_price, high_price, low_price, close_price"

    conditions = ["stock_code = %s"]
    params: list = [stock_code]
    if start_date:
        conditions.append("trade_date >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= %s")
        params.append(end_date)

    sql = f"""
        SELECT trade_date, {price_cols}, volume, amount
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
                            end_date: Optional[str] = None,
                            dividend_type: str = "front") -> dict:
    """
    批量加载, 返回 {stock_code: DataFrame}

    参数:
        dividend_type: 'front' 读前复权列, 'back' 读后复权列
    """
    result = {}
    for code in stock_codes:
        df = load_stock_kline(code, start_date, end_date, dividend_type=dividend_type)
        if not df.empty:
            result[code] = df
    return result


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="个股 K 线同步 (双复权)")
    parser.add_argument("--mode", choices=["init", "daily"], default="daily",
                        help="init=全量初始化, daily=每日增量")
    parser.add_argument("--start", default=None,
                        help="全量起始日 YYYYMMDD, 默认读 .env")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2,
                        help="按哪一级板块取股票池 (默认 2)")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help=f"并行下载线程数 (默认 {NUM_WORKERS})")
    args = parser.parse_args()

    sync_all_kline(start_date=args.start,
                   incremental=(args.mode == "daily"),
                   level=args.level,
                   workers=args.workers)


if __name__ == "__main__":
    main()
