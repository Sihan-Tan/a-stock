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
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from xtquant import xtdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import get_connection, execute_query

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
DATA_START = '20230101'
# 数据库写入最大重试次数，应对并发写入时的死锁场景
DB_MAX_RETRIES = 5
# 重试基础等待时间（秒），采用指数退避策略：base * 2^(attempt-1)
DB_RETRY_BASE_SLEEP = 0.2


# ============================================================
# 数据库辅助
# ============================================================

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


def _get_float_shares(stock_code):
    """
    获取流通股本（股），用于计算换手率。

    换手率 = 成交量(股) / 流通股本(股) * 100%
    成交量从 xtdata 获取时单位是"手"（1手=100股），需要先转换为股。

    Args:
        stock_code: 股票代码，如 "600519.SH"

    Returns:
        int: 流通股本数（股），获取失败返回 0
    """
    try:
        # get_instrument_detail 返回股票的详细信息字典
        detail = xtdata.get_instrument_detail(stock_code)
        if detail:
            # 优先取 NegotiableVolume（流通股本），失败时回退到 TotalVolume（总股本）
            neg = detail.get('NegotiableVolume') or detail.get('TotalVolume') or 0
            if neg > 0:
                return neg
    except Exception:
        pass
    return 0


def download_and_save(stock_code, start_date):
    """
    增量下载单只股票的日线数据并写入MySQL。

    工作流程：
      1. 调用 xtdata.download_history_data() 将数据下载到本地缓存
      2. 调用 xtdata.get_market_data_ex() 从本地缓存读取数据
      3. 计算换手率等衍生指标
      4. 写入 MySQL（带重试机制应对死锁）

    Args:
        stock_code: 股票代码，如 "600519.SH"
        start_date: 起始日期字符串，格式 "YYYYMMDD"

    Returns:
        tuple: (stock_code, count)
            count > 0 表示成功写入的行数
            count = 0 表示无新数据
            count = -1 表示写入失败
    """
    print(f"开始下载 {stock_code} 从 {start_date} 开始的数据")
    # download_history_data 是 xtquant 核心API：
    # 它会将历史行情数据下载到本地缓存目录，后续的 get_market_data_ex 从缓存读取
    # period='1d' 表示日线数据
    xtdata.download_history_data(stock_code, '1d', start_time=start_date)

    # get_market_data_ex 从本地缓存读取数据
    # dividend_type='front' 表示使用前复权价格（考虑分红除权，调整历史价格）
    # 前复权 vs 后复权：前复权保持当前价格不变调整历史价格，后复权保持历史价格不变调整当前价格
    data = xtdata.get_market_data_ex(
        field_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
        stock_list=[stock_code],
        period='1d',
        start_time=start_date,
        dividend_type='front',
    )

    if not data or stock_code not in data:
        print(f"未获取到 {stock_code} 的数据")
        return stock_code, 0

    df = data[stock_code]
    print(f"获取到 {stock_code} 的 {len(df)} 条数据")
    if df is None or len(df) == 0:
        return stock_code, 0

    # 获取流通股本用于计算换手率
    float_shares = _get_float_shares(stock_code)

    rows = []
    for idx, row in df.iterrows():
        # xtdata的DataFrame索引格式为 '20250115' 这样的YYYYMMDD字符串
        idx_str = str(idx)
        if len(idx_str) >= 8:
            # 将 YYYYMMDD 转为 MySQL 日期格式 YYYY-MM-DD
            trade_date = f"{idx_str[:4]}-{idx_str[4:6]}-{idx_str[6:8]}"
            vol = int(row['volume'])
            # xtdata volume 单位是"手"(1手=100股)，流通股本单位是"股"
            # 因此成交量(股) = 成交量(手) * 100
            vol_shares = vol * 100
            # 换手率 = 成交量(股) / 流通股本(股) * 100%
            # 保留4位小数，如果流通股本为0或成交量为0则设为None
            turnover = round(vol_shares / float_shares * 100, 4) if float_shares > 0 and vol > 0 else None
            rows.append((
                stock_code, trade_date,
                float(row['open']), float(row['high']),
                float(row['low']), float(row['close']),
                vol, float(row['amount']),
                turnover,
            ))

    if not rows:
        print("没有数据需要写入")
        return stock_code, 0

    # 写入数据库，带指数退避重试机制
    # 在高并发场景下，多个线程同时写入MySQL可能产生死锁(deadlock)，
    # MySQL检测到死锁后会回滚其中一个事务并返回 1213 错误码。
    for attempt in range(1, DB_MAX_RETRIES + 1):
        conn = None
        cursor = None
        try:
            conn = get_connection()
            cursor = conn.cursor()
            # executemany 批量写入，比逐条 execute 效率高得多
            cursor.executemany(INSERT_SQL, rows)
            conn.commit()
            print(f"成功写入 {len(rows)} 条数据到数据库")
            return stock_code, len(rows)
        except Exception as e:
            err_msg = str(e)
            # 判断是否为死锁错误（MySQL Error 1213 = ER_LOCK_DEADLOCK）
            is_deadlock = ("1213" in err_msg) or ("Deadlock found" in err_msg)
            if conn:
                conn.rollback()  # 回滚事务，释放锁

            if is_deadlock and attempt < DB_MAX_RETRIES:
                # 指数退避：第一次等待0.2s，第二次0.4s，第三次0.8s...
                # 这样设计是因为死锁通常是短暂的，稍后重试大概率成功
                sleep_s = DB_RETRY_BASE_SLEEP * (2 ** (attempt - 1))
                print(f"写入发生死锁，重试 {attempt}/{DB_MAX_RETRIES - 1}，等待 {sleep_s:.1f}s: {err_msg}")
                time.sleep(sleep_s)
                continue

            print(f"写入数据库失败: {err_msg}")
            return stock_code, -1
        finally:
            # 确保连接和游标被关闭，避免连接泄漏
            if cursor:
                cursor.close()
            if conn:
                conn.close()


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("行情数据采集 (MiniQMT -> MySQL)")
    if TEST_MODE:
        print("[测试模式] 只采集贵州茅台")
    else:
        print(f"[全量模式] 采集{SECTOR}, {NUM_WORKERS}线程并行")
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
        # 过滤掉不含'.'的代码（如指数代码000001.SH保留，无效数据剔除）
        all_codes = [c for c in all_codes if '.' in str(c)]
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

    # 今天的日期作为截止点：如果数据库中最新的日期 >= 今天，说明已是最新不需要更新
    recent_cutoff = date.today().strftime('%Y%m%d')

    tasks = []
    skip_count = 0
    for code in all_codes:
        latest = existing.get(code)
        if latest and latest >= recent_cutoff:
            # 数据库中已有今天的日线数据，跳过
            skip_count += 1
            continue
        # 如果有历史数据则从最新日期的次日开始，否则从 DATA_START 开始
        start = latest if latest else DATA_START
        tasks.append((code, start))

    print(f"  需更新: {len(tasks)} 只, 跳过(今日已有数据): {skip_count} 只")

    if not tasks:
        print("\n全部已是最新，无需更新")
        _print_summary()
        return

    total = len(tasks)
    total_rows = 0
    success_count = 0
    fail_list = []
    start_time = time.time()

    # 数量较少时用串行（避免多线程开销），数量大时用并行
    if total <= 5:
        # 串行处理：逐只下载、写入
        for i, (code, start) in enumerate(tasks, 1):
            print(f"\n[{i}/{total}] {code} (从 {start} 开始)")
            _, count = download_and_save(code, start)
            if count >= 0:
                print(f"  写入 {count} 条")
                success_count += 1
                total_rows += max(count, 0)
            else:
                print(f"  失败")
                fail_list.append(code)
    else:
        # 并行处理：使用 ThreadPoolExecutor 实现多线程并发
        # 由于网络IO和数据库IO是瓶颈，多线程可以显著提升效率
        print(f"\n并行下载（{NUM_WORKERS} 线程）...")

        def _worker(args):
            """线程工作函数，捕获异常防止单个股票失败导致整个线程崩溃"""
            code, start = args
            try:
                return download_and_save(code, start)
            except Exception:
                return code, -1

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            # 提交所有任务到线程池
            futures = {executor.submit(_worker, t): t[0] for t in tasks}
            done = 0
            for future in as_completed(futures):
                code, count = future.result()
                done += 1

                if count >= 0:
                    success_count += 1
                    total_rows += max(count, 0)
                else:
                    fail_list.append(code)

                # 实时进度显示，使用 \r 在同一行刷新
                elapsed = time.time() - start_time
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / speed if speed > 0 else 0
                sys.stdout.write(
                    f"\r  进度 {done}/{total} ({done*100/total:.1f}%) | "
                    f"{speed:.1f} 只/秒 | 剩余约 {eta:.0f}秒 | "
                    f"成功 {success_count} 失败 {len(fail_list)}    "
                )
                sys.stdout.flush()

        print()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"采集完成! 耗时 {elapsed:.1f} 秒")
    print(f"  成功: {success_count}/{total} 只股票")
    print(f"  总写入: {total_rows:,} 条记录")

    if fail_list:
        print(f"  失败 {len(fail_list)} 只: {fail_list[:20]}{'...' if len(fail_list) > 20 else ''}")

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
