# -*- coding: utf-8 -*-
"""
获取所有A股股票代码并存入 stock_list 表。

功能：
  1. 连接 MiniQMT 数据服务
  2. 获取沪深A股全量股票列表，过滤ETF/债券/基金等非股票代码
  3. 批量获取股票名称
  4. 写入 MySQL stock_list 表（ON DUPLICATE KEY UPDATE）

模式：
  - TEST_MODE = True  -> 只采集10只股票，用于验证流程
  - TEST_MODE = False -> 采集沪深A股全量股票

运行：python 0-获取所有股票代码.py
环境：需安装QMT并配置好xtquant, pip install pymysql python-dotenv
"""
import sys
import os

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
TEST_MODE = False
TEST_LIMIT = 10  # 测试模式下采集的股票数量

SECTOR = '沪深A股'

# A股代码前缀范围（排除 ETF 51xxxx/159xxx、债券 11xxxx/12xxxx、基金 50xxxx 等）
A_STOCK_SH_PREFIXES = set(list(range(600, 607)) + list(range(688, 691)))
A_STOCK_SZ_PREFIXES = set(list(range(0, 5)) + list(range(300, 303)))
A_STOCK_BJ_PREFIXES = set(range(800, 930))

INSERT_SQL = """
    INSERT INTO stock_list (stock_code, stock_name, exchange)
    VALUES (%s, %s, %s)
    ON DUPLICATE KEY UPDATE
        stock_name = VALUES(stock_name),
        exchange   = VALUES(exchange)
"""


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


def main():
    print("=" * 60)
    print("获取A股股票代码 (MiniQMT -> MySQL stock_list)")
    if TEST_MODE:
        print(f"[测试模式] 只采集前 {TEST_LIMIT} 只")
    else:
        print(f"[全量模式] 采集 {SECTOR} 全量股票")
    print("=" * 60)

    print("\n连接QMT数据服务...")
    xtdata.connect()
    print("  连接成功")

    print(f"\n获取 {SECTOR} 股票列表...")
    all_codes = xtdata.get_stock_list_in_sector(SECTOR)
    all_codes = [c for c in all_codes if '.' in str(c)]

    # 过滤非股票代码
    filtered_out = [c for c in all_codes if not _is_a_stock(c)]
    stocks = [c for c in all_codes if _is_a_stock(c)]
    if filtered_out:
        print(f"  过滤非股票代码 {len(filtered_out)} 只: "
              f"{filtered_out[:5]}{'...' if len(filtered_out) > 5 else ''}")
    print(f"  共 {len(stocks)} 只股票")

    if TEST_MODE:
        stocks = stocks[:TEST_LIMIT]
        print(f"  [测试模式] 仅处理前 {len(stocks)} 只")

    # 批量获取股票名称
    print(f"\n获取股票名称 ({len(stocks)} 只)...")
    rows = []
    fail_names = []
    for code in stocks:
        exchange = code.split('.')[-1]
        try:
            detail = xtdata.get_instrument_detail(code)
            name = detail.get('InstrumentName', '') if detail else ''
        except Exception:
            name = ''
        if not name:
            fail_names.append(code)
        rows.append((code, name, exchange))

    if fail_names:
        print(f"  未获取到名称: {len(fail_names)} 只 -> {fail_names[:5]}{'...' if len(fail_names) > 5 else ''}")

    # 写入数据库
    print(f"\n写入 stock_list 表...")
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.executemany(INSERT_SQL, rows)
        conn.commit()
        affected = cursor.rowcount
        cursor.close()
    finally:
        conn.close()

    print(f"  写入完成, 影响 {affected} 行")

    # 汇总
    summary = execute_query(
        "SELECT COUNT(*) AS cnt FROM stock_list"
    )
    print(f"\nstock_list 表共 {summary[0]['cnt']} 条记录")
    print("=" * 60)


if __name__ == "__main__":
    main()
