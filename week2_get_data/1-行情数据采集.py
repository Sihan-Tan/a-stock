# -*- coding: utf-8 -*-
"""
行情数据采集 - 使用MiniQMT(xtquant)下载全量A股日线数据存入MySQL

功能：
  1. 连接MiniQMT数据服务
  2. 获取沪深A股全量股票列表（约5000只）
  3. 一次性批量查询DB中已有的最新日期，仅下载增量数据
  4. 多线程写入MySQL的trade_stock_daily表（ON DUPLICATE KEY UPDATE）

优化：
  - 不逐只查名称（太慢），直接用股票代码
  - 批量查询DB最新日期，跳过已是最新的股票
  - 移除不必要的sleep，提升吞吐量

模式：
  - TEST_MODE = True  -> 只采集1只股票(贵州茅台)，用于验证流程
  - TEST_MODE = False -> 采集沪深A股全量股票

运行：python 1-行情数据采集.py
环境：需安装QMT并配置好xtquant, pip install pymysql python-dotenv
"""
import sys
import os
import time
import math
from datetime import date, timedelta, datetime
from concurrent.futures import ThreadPoolExecutor

from xtquant import xtdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import get_connection, execute_query, execute_update

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ============================================================
# 配置
# ============================================================
# 测试模式开关: True=只采贵州茅台验证流程, False=全量采集
TEST_MODE = False
# 测试用股票代码（600519.SH = 贵州茅台，SH表示上海交易所）
TEST_STOCK = '600519.SH'

# 板块名称常量，对应xtquant内置板块分类
SECTOR = '沪深A股'
# 并行工作线程数，根据CPU核心数和网络IO情况调整
NUM_WORKERS = 8
# 默认起始日期（YYYYMMDD格式），当数据库中没有该股票数据时从此日期开始下载
DATA_START = '20240101'
# 数据库写入最大重试次数，应对并发写入时的死锁场景
DB_MAX_RETRIES = 5
# 重试基础等待时间（秒），采用指数退避策略：base * 2^(attempt-1)
DB_RETRY_BASE_SLEEP = 0.2


# ============================================================
# 股票筛选
# ============================================================

# A股代码前缀范围（排除 ETF 51xxxx/159xxx、债券 11xxxx/12xxxx、基金 50xxxx 等）
# SH: 主板600-605 + 科创板688-689 | SZ: 主板000-004 + 创业板300-301 | BJ: 北交所8xxxxx
A_STOCK_SH_PREFIXES = set(list(range(600, 607)) + list(range(688, 691)))  # 主板600-605 + 科创板688-689
A_STOCK_SZ_PREFIXES = set(list(range(0, 5)) + list(range(300, 303)))       # 主板000-004 + 创业板300-302
A_STOCK_BJ_PREFIXES = set(range(800, 930))                                  # 北交所 8xxxxx ，9xxxxx


def _is_a_stock(code):
    """判断是否为A股股票，排除ETF/债券/基金等"""
    if not code or '.' not in code:
        return False
    num_part = code.split('.')[0]
    if len(num_part) != 6:
        return False
    try:
        prefix = int(num_part[:3])
    except ValueError:
        return False
    if code.endswith('.SH'):
        return prefix in A_STOCK_SH_PREFIXES
    if code.endswith('.SZ'):
        return prefix in A_STOCK_SZ_PREFIXES
    if code.endswith('.BJ'):
        return prefix in A_STOCK_BJ_PREFIXES
    return False


def _clean_delisted():
    """
    删除DB中已退市股票的数据。

    通过 xtdata 查询退市板块，与 DB 中现有股票做交集，
    找到已退市但仍有残余数据的股票并删除。

    Returns:
        int: 清理的股票数量
    """
    # 尝试多个可能的退市板块名称
    delisted = set()
    for sector_name in ['退市股票', '退市', '已退市', '退市板块']:
        try:
            codes = xtdata.get_stock_list_in_sector(sector_name)
            if codes:
                delisted.update(codes)
        except Exception:
            pass

    if not delisted:
        print("  未获取到退市股票列表（可能QMT版本不支持该板块名）")
        return 0

    # 与DB有数据的股票取交集
    db_codes = {r['stock_code'] for r in execute_query(
        "SELECT DISTINCT stock_code FROM trade_stock_daily"
    )}
    to_delete = delisted & db_codes

    if not to_delete:
        print(f"  DB中无退市股票残留")
        return 0

    print(f"  发现 {len(to_delete)} 只已退市股票: {sorted(to_delete)[:10]}{'...' if len(to_delete) > 10 else ''}")

    for code in to_delete:
        execute_update("DELETE FROM trade_stock_daily WHERE stock_code = %s", (code,))

    print(f"  已清理 {len(to_delete)} 只退市股票的全部数据")
    return len(to_delete)


# ============================================================
# 数据库辅助
# ============================================================

def _add_one_day(date_str):
    """给YYYYMMDD日期字符串加一天，返回YYYYMMDD字符串"""
    d = datetime.strptime(date_str, '%Y%m%d')
    return (d + timedelta(days=1)).strftime('%Y%m%d')


def get_existing_latest_dates():
    """
    一次性查询所有股票在DB中的最新交易日。

    核心优化：
      如果不做此查询，每只股票都需要单独查一次数据库(O(n))，
      批量查询只需一次(O(1))，大幅减少数据库交互次数。

    返回:
        dict: {stock_code: 'YYYYMMDD'} 格式的映射字典
              如果某股票没有数据，则不会出现在返回结果中
    """
    # GROUP BY 聚合查询，对每只股票取 MAX(trade_date)
    rows = execute_query(
        "SELECT stock_code, MAX(trade_date) AS max_date FROM trade_stock_daily GROUP BY stock_code"
    )
    result = {}
    for r in rows:
        if r['max_date']:
            # 将 datetime 对象转为 YYYYMMDD 字符串，方便与 DATA_START 比较
            date_str = r['max_date'].strftime('%Y%m%d')
            result[r['stock_code']] = date_str
    return result


# ============================================================
# 核心逻辑
# ============================================================

# INSERT ... ON DUPLICATE KEY UPDATE 是 MySQL 的"UPSERT"语法：
#   1) 如果主键或唯一索引不冲突，则插入新记录
#   2) 如果冲突，则更新指定字段
# 这确保了重复运行脚本不会产生重复数据，而是会覆盖更新已有数据。
INSERT_SQL = """
    INSERT INTO trade_stock_daily
    (stock_code, trade_date, open_price, high_price, low_price, close_price, volume, amount, turnover_rate)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    open_price=VALUES(open_price), high_price=VALUES(high_price),
    low_price=VALUES(low_price), close_price=VALUES(close_price),
    volume=VALUES(volume), amount=VALUES(amount),
    turnover_rate=VALUES(turnover_rate)
"""


def _batch_get_float_shares(stock_codes):
    """
    批量获取流通股本（股），用于计算换手率。

    一次性拉取所有需要更新的股票的流通股本，避免5000次单独API调用。
    流通股本是静态数据（仅在增发/回购时变动），不需要每次更新。

    Returns:
        dict: {stock_code: float_shares}
    """
    result = {}
    for code in stock_codes:
        try:
            detail = xtdata.get_instrument_detail(code)
            if detail:
                neg = detail.get('NegotiableVolume') or detail.get('TotalVolume') or 0
                if neg > 0:
                    result[code] = neg
                    continue
        except Exception:
            pass
        result[code] = 0
    return result


def _batch_write_to_db(rows):
    """批量写入DB，带死锁重试"""
    for attempt in range(1, DB_MAX_RETRIES + 1):
        conn = None
        cursor = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.executemany(INSERT_SQL, rows)
            conn.commit()
            return len(rows)
        except Exception as e:
            err_msg = str(e)
            is_deadlock = ("1213" in err_msg) or ("Deadlock found" in err_msg)
            if conn:
                conn.rollback()
            if is_deadlock and attempt < DB_MAX_RETRIES:
                sleep_s = DB_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
                print(f"  写入死锁，重试 {attempt}/{DB_MAX_RETRIES - 1}，等待 {sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue
            print(f"  写入数据库失败: {err_msg}")
            return -1
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    return -1


def _nf(v):
    """将 numpy nan 转为 None，MySQL 不接受 nan 值"""
    if v is None:
        return None
    try:
        if math.isnan(v):
            return None
    except TypeError:
        pass
    return float(v)


def _process_batch(stock_codes, start_date, float_shares_map):
    """
    按 chunk 处理一组股票（相同 start_date 的分为一组）。

    每 50 只为一个 chunk：下载 → 读取 → 入库 闭环完成后再处理下一批。
    中途崩溃最多丢失当前 chunk 的 50 只数据，已入库的不受影响。

    Returns:
        (success_count, failed_codes)
    """
    DL_CHUNK_SIZE = 50
    dl_workers = NUM_WORKERS * 2
    total = len(stock_codes)
    success = 0
    failed = []
    done = 0
    t0 = time.time()

    for i in range(0, total, DL_CHUNK_SIZE):
        chunk = stock_codes[i:i + DL_CHUNK_SIZE]

        # 1) 多线程下载当前 chunk
        with ThreadPoolExecutor(max_workers=min(len(chunk), dl_workers)) as dl_executor:
            dl_executor.map(lambda c: xtdata.download_history_data(c, '1d', start_time=start_date), chunk)

        # 2) 批量读取当前 chunk
        data = xtdata.get_market_data_ex(
            field_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
            stock_list=chunk,
            period='1d',
            start_time=start_date,
            dividend_type='front',
        )

        if not data:
            done += len(chunk)
            continue

        # 3) 转换 + 入库
        all_rows = []
        pending = []

        for code in chunk:
            df = data.get(code)
            if df is None or len(df) == 0:
                continue

            float_shares = float_shares_map.get(code, 0)
            try:
                idx_arr = df.index.values
                open_arr = df['open'].values
                high_arr = df['high'].values
                low_arr = df['low'].values
                close_arr = df['close'].values
                vol_arr = df['volume'].values
                amt_arr = df['amount'].values
            except Exception:
                failed.append(code)
                continue

            for j in range(len(df)):
                idx_str = str(idx_arr[j])
                if len(idx_str) < 8:
                    continue
                trade_date = f"{idx_str[:4]}-{idx_str[4:6]}-{idx_str[6:8]}"
                vol = int(vol_arr[j])
                vol_shares = vol * 100
                turnover = round(vol_shares / float_shares * 100, 4) if float_shares > 0 and vol > 0 else None
                all_rows.append((
                    code, trade_date,
                    _nf(open_arr[j]), _nf(high_arr[j]),
                    _nf(low_arr[j]), _nf(close_arr[j]),
                    vol, _nf(amt_arr[j]),
                    turnover,
                ))

            pending.append(code)

        if all_rows:
            written = _batch_write_to_db(all_rows)
            if written > 0:
                success += len(pending)
            else:
                failed.extend(pending)

        done += len(chunk)
        elapsed = time.time() - t0
        speed = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / speed if speed > 0 else 0
        sys.stdout.write(
            f"\r  进度 {min(done, total)}/{total} ({done*100/total:.0f}%) | "
            f"入库 {success} 只 | {speed:.0f} 只/秒 | 剩余约 {eta:.0f}秒    "
        )
        sys.stdout.flush()

    print()
    return success, failed


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("行情数据采集 (MiniQMT -> MySQL)")
    if TEST_MODE:
        print("[测试模式] 只采集贵州茅台")
    else:
        print(f"[全量模式] 采集{SECTOR}, 下载{NUM_WORKERS * 2}线程 + 批量读取")
    print("=" * 60)

    print("\n连接QMT数据服务...")
    # xtdata.connect() 建立与MiniQMT的数据连接
    # 如果QMT客户端未运行或未登录，此调用会失败
    xtdata.connect()
    print("  连接成功")

    # 获取股票列表
    if TEST_MODE:
        all_codes = [TEST_STOCK]
        print(f"\n[测试模式] 只采集 {TEST_STOCK}")
    else:
        print(f"\n获取 {SECTOR} 股票列表...")
        # get_stock_list_in_sector 获取指定板块的所有股票代码
        # 沪深A股板块包含上海主板、深圳主板、创业板、科创板等
        all_codes = xtdata.get_stock_list_in_sector(SECTOR)
        all_codes = [c for c in all_codes if '.' in str(c)]
        # 过滤ETF/债券/基金等非股票代码
        filtered_out = [c for c in all_codes if not _is_a_stock(c)]
        all_codes = [c for c in all_codes if _is_a_stock(c)]
        if filtered_out:
            print(f"  过滤非股票代码 {len(filtered_out)} 只: {filtered_out[:5]}{'...' if len(filtered_out) > 5 else ''}")
        print(f"  共 {len(all_codes)} 只股票")

    # 批量查询DB中已有的最新日期
    # 这是核心优化点：一次性查出所有股票的最新日期，而不是逐只查询
    print("查询数据库已有数据...")
    existing = get_existing_latest_dates()
    # 今天的数据已有则跳过，否则尝试增量更新
    print(f"数据库中已有 {len(existing)} 只股票的数据")
    if len(existing) > 0:
        sample_code = list(existing.keys())[0]
        print(f"示例: {sample_code} 最新日期为 {existing[sample_code]}")

    # 用今天的日期推算最近交易日，跳过周末（周六-1天，周日-2天）
    # 不用 max(existing)：中途失败会导致部分股票落后，不应以最超前的那只为基准
    today = date.today()
    w = today.weekday()  # 0=周一 ... 5=周六 6=周日
    offset = w - 4 if w > 4 else 0  # 周六→2天前(周五), 周日→2天前(周五)
    recent_cutoff = (today - timedelta(days=offset)).strftime('%Y%m%d')

    # 按 start_date 分组，同一组内可批量下载
    groups = {}  # {start_date: [codes]}
    skip_count = 0
    for code in all_codes:
        latest = existing.get(code)
        if latest and latest >= recent_cutoff:
            skip_count += 1
            continue
        start = _add_one_day(latest) if latest else DATA_START
        groups.setdefault(start, []).append(code)

    total = sum(len(v) for v in groups.values())
    print(f"  需更新: {total} 只, 跳过(已是最近交易日): {skip_count} 只")

    if not groups:
        print("\n全部已是最新，无需更新")
        _print_summary()
        return

    # 预取所有待更新股票的流通股本
    all_update_codes = [c for codes in groups.values() for c in codes]
    print(f"预取流通股本 ({len(all_update_codes)} 只)...")
    float_shares_map = _batch_get_float_shares(all_update_codes)
    print(f"  获取到 {len(float_shares_map)} 只股票的流通股本")

    success_count = 0
    fail_list = []
    start_time = time.time()

    # 按组处理（每组对应一个 start_date），_process_batch 内部多线程下载
    for start_date, codes in groups.items():
        print(f"\n批次 start={start_date}: {len(codes)} 只股票")

        s, f = _process_batch(codes, start_date, float_shares_map)
        success_count += s
        fail_list.extend(f)

        elapsed = time.time() - start_time
        speed = success_count / elapsed if elapsed > 0 else 0
        print(f"  完成 {success_count} 只, 耗时 {elapsed:.1f}秒, {speed:.1f} 只/秒")

    print()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"采集完成! 耗时 {elapsed:.1f} 秒")
    print(f"  成功: {success_count}/{total} 只股票")
    if fail_list:
        print(f"  失败 {len(fail_list)} 只: {fail_list[:20]}{'...' if len(fail_list) > 20 else ''}")

    # 清理已退市股票的数据
    print("\n清理退市股票...")
    _clean_delisted()

    _print_summary()


def _print_summary():
    """打印数据库汇总统计，用于验证采集结果的完整性"""
    summary = execute_query("""
        SELECT COUNT(DISTINCT stock_code) as stock_cnt,
               COUNT(*) as row_cnt,
               MIN(trade_date) as min_date, MAX(trade_date) as max_date
        FROM trade_stock_daily
    """)
    if summary:
        row = summary[0]
        print(f"\n数据库 trade_stock_daily 概况:")
        print(f"  {row['stock_cnt']} 只股票, {row['row_cnt']:,} 条记录")
        print(f"  日期范围: {row['min_date']} ~ {row['max_date']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
