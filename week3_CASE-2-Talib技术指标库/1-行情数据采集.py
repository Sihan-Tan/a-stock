# -*- coding: utf-8 -*-
"""
行情数据采集 -- 使用 MiniQMT (xtquant) 下载全量 A 股日线数据存入 MySQL

本脚本是整个项目的数据源头。没有数据，一切策略和回测都无从谈起。

功能：
  1. 连接 MiniQMT 数据服务（需安装 QMT 客户端并配置 xtquant）
  2. 获取沪深 A 股全量股票和 ETF 列表（约 5000+ 只）
  3. 批量查询数据库中已有的最新日期，仅下载增量数据（避免重复下载）
  4. 多线程并行写入 MySQL 的 trade_stock_daily 表，使用 ON DUPLICATE KEY UPDATE
     实现幂等写入（可重复运行且不会产生重复记录）

优化思路：
  - 不逐只查询股票名称（太慢），直接用股票代码操作
  - 批量查询 DB 中所有股票的最新日期，跳过已是最新的股票
  - 使用线程池并行下载，充分利用网络和 CPU
  - 移除不必要的 time.sleep()，提升吞吐量

运行模式：
  - TEST_MODE = True  -> 只采集 1 只股票（贵州茅台 600519.SH），用于验证流程是否正常
  - TEST_MODE = False -> 采集沪深 A 股 + ETF 全量股票

运行：python 1-行情数据采集.py
环境：需安装 QMT 并配置好 xtquant，pip install pymysql python-dotenv
"""
import sys
import os
import time
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from xtquant import xtdata  # MiniQMT 数据接口库

# 将上级目录加入模块搜索路径，确保能导入 db_config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import get_connection, execute_query

# 解决 Windows 下控制台输出 UTF-8 中文乱码的问题
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ============================================================
# 运行配置
# ============================================================
TEST_MODE = False           # True=仅测试茅台，False=全量采集
TEST_STOCK = '600519.SH'    # 测试用股票：贵州茅台

SECTOR = ['沪深A股', '沪深ETF']   # 需要采集的板块
NUM_WORKERS = 8                    # 并行下载线程数
DATA_START = '20230101'            # 对于新股票（库中无数据），从此日期开始下载


# ============================================================
# 数据库辅助函数
# ============================================================

def get_existing_latest_dates():
    """
    一次性查询所有股票在数据库中的最新交易日

    为什么需要批量查询？
      如果逐只查询，5000 只股票需要 5000 次 SQL 查询，耗时数十秒。
      一条 GROUP BY 查询即可完成，耗时不到 0.1 秒。

    返回:
        dict {stock_code: 'YYYYMMDD'}
        例如 {'600519.SH': '20250520', '000001.SZ': '20250520'}
    """
    rows = execute_query(
        "SELECT stock_code, MAX(trade_date) AS max_date FROM trade_stock_daily GROUP BY stock_code"
    )
    result = {}
    for r in rows:
        if r['max_date']:
            # 将 datetime.date 转为字符串 'YYYYMMDD'，与 xtdata 的日期格式一致
            result[r['stock_code']] = r['max_date'].strftime('%Y%m%d')
    return result


# ============================================================
# 核心逻辑：下载并保存单只股票数据
# ============================================================

# SQL 插入语句，使用 ON DUPLICATE KEY UPDATE 实现"有则更新，无则插入"
# 这样脚本可以每天重复运行，不会产生重复数据
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
    获取流通股本（单位：股），用于计算换手率

    换手率 = 当日成交量(股) / 流通股本(股) * 100
    换手率是衡量股票活跃度的重要指标：
      - 换手率高（>5%）：交易活跃，可能有大资金进出
      - 换手率低（<1%）：交投冷清，流动性差

    参数:
        stock_code: 股票代码

    返回:
        int 流通股本数，获取失败返回 0
    """
    try:
        detail = xtdata.get_instrument_detail(stock_code)
        if detail:
            # NegotiableVolume = 流通股本，TotalVolume = 总股本
            # 优先用流通股本，计算出的换手率更有意义
            neg = detail.get('NegotiableVolume') or detail.get('TotalVolume') or 0
            if neg > 0:
                return neg
    except Exception:
        pass
    return 0


def download_and_save(stock_code, start_date):
    """
    增量下载单只股票的日线数据并写入 MySQL

    流程：
      1. 调用 xtdata.download_history_data() 从 QMT 服务器下载数据
      2. 调用 xtdata.get_market_data_ex() 获取本地缓存中的行情数据
      3. 解析数据，计算换手率
      4. 批量写入 MySQL

    为什么用 xtdata.get_market_data_ex() 而非 download 的直接结果？
      download_history_data() 只是将数据下载到本地缓存，
      get_market_data_ex() 才是真正从缓存中读取结构化数据的接口。

    参数:
        stock_code: 股票代码，如 '600519.SH'
        start_date: 起始日期，如 '20230101'

    返回:
        (stock_code, record_count) 元组
        record_count >= 0 表示写入的记录数，-1 表示失败
    """
    # 第1步：下载历史数据到本地缓存
    # start_time 指定从哪天开始下载，增量更新
    xtdata.download_history_data(stock_code, '1d', start_time=start_date)

    # 第2步：从本地缓存读取数据
    # period='1d' 日线，dividend_type='front' 前复权
    data = xtdata.get_market_data_ex(
        field_list=['open', 'high', 'low', 'close', 'volume', 'amount'],
        stock_list=[stock_code],
        period='1d',
        start_time=start_date,
        dividend_type='front',  # 前复权：调整历史价格，使价格连续可比
    )

    if not data or stock_code not in data:
        return stock_code, 0

    df = data[stock_code]
    if df is None or len(df) == 0:
        return stock_code, 0

    # 获取流通股本用于计算换手率
    float_shares = _get_float_shares(stock_code)

    # 第3步：解析 DataFrame 并组装 SQL 参数
    rows = []
    for idx, row in df.iterrows():
        idx_str = str(idx)
        if len(idx_str) >= 8:
            # xtdata 的日期格式为 'YYYYMMDD'，转为 MySQL 的 'YYYY-MM-DD' 格式
            trade_date = f"{idx_str[:4]}-{idx_str[4:6]}-{idx_str[6:8]}"
            vol = int(row['volume'])
            # xtdata 中 volume 单位是"手"（1手=100股）
            # 而流通股本（float_shares）单位是"股"，需要统一单位
            vol_shares = vol * 100
            # 换手率（%）= 成交股数 / 流通股本 * 100
            turnover = round(vol_shares / float_shares * 100, 4) if float_shares > 0 and vol > 0 else None
            rows.append((
                stock_code, trade_date,
                float(row['open']), float(row['high']),
                float(row['low']), float(row['close']),
                vol, float(row['amount']),
                turnover,
            ))

    # 第4步：批量写入 MySQL
    if rows:
        conn = get_connection()
        cursor = conn.cursor()
        # executemany 批量执行，一次网络往返写入所有数据，比逐条插入快百倍
        cursor.executemany(INSERT_SQL, rows)
        conn.commit()  # 提交事务
        cursor.close()
        conn.close()

    return stock_code, len(rows)


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

    # ---- 第1步：连接 QMT 数据服务 ----
    print("\n连接QMT数据服务...")
    xtdata.connect()
    print("  连接成功")

    # ---- 第2步：获取股票列表 ----
    if TEST_MODE:
        all_codes = [TEST_STOCK]
        print(f"\n[测试模式] 只采集 {TEST_STOCK}")
    else:
        print(f"\n获取 {SECTOR} 股票列表...")
        all_codes = []
        seen = set()
        for sector_name in SECTOR:
            codes = xtdata.get_stock_list_in_sector(sector_name)
            for c in (codes or []):
                # 过滤无效代码（xtdata 可能返回不含 '.' 的非标准代码）
                if '.' in str(c) and c not in seen:
                    seen.add(c)
                    all_codes.append(c)
        print(f"  共 {len(all_codes)} 只股票")

    # ---- 第3步：批量查询 DB 已有数据，确定增量范围 ----
    print("查询数据库已有数据...")
    existing = get_existing_latest_dates()
    # 如果数据库中的最新日期 >= 今天，说明今日数据已存在，跳过
    recent_cutoff = date.today().strftime('%Y%m%d')

    tasks = []
    skip_count = 0
    for code in all_codes:
        latest = existing.get(code)
        if latest and latest >= recent_cutoff:
            skip_count += 1
            continue  # 今日数据已存在，跳过
        # 有历史数据则从最新日期的次日开始，无历史数据则从 DATA_START 开始
        start = latest if latest else DATA_START
        tasks.append((code, start))

    print(f"  需更新: {len(tasks)} 只, 跳过(今日已有数据): {skip_count} 只")

    if not tasks:
        print("\n全部已是最新，无需更新")
        _print_summary()
        return

    # ---- 第4步：执行下载 ----
    total = len(tasks)
    total_rows = 0
    success_count = 0
    fail_list = []
    start_time = time.time()

    if total <= 5:
        # 数量少时串行处理，避免线程池开销
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
        # 数量多时使用线程池并行下载
        print(f"\n并行下载（{NUM_WORKERS} 线程）...")

        def _worker(args):
            """线程工作函数，捕获异常防止线程崩溃"""
            code, start = args
            try:
                return download_and_save(code, start)
            except Exception:
                return code, -1

        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
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

                # 进度显示：使用 \r 覆盖同一行，实现动态进度条效果
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

    # ---- 第5步：输出统计 ----
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"采集完成! 耗时 {elapsed:.1f} 秒")
    print(f"  成功: {success_count}/{total} 只股票")
    print(f"  总写入: {total_rows:,} 条记录")

    if fail_list:
        print(f"  失败 {len(fail_list)} 只: {fail_list[:20]}{'...' if len(fail_list) > 20 else ''}")

    _print_summary()


def _print_summary():
    """打印数据库中的整体数据概况"""
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
