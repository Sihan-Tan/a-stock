# -*- coding: utf-8 -*-
# 21-CASE-A: 一次性全量初始化
"""
run_init.py -- 一键全量初始化

执行顺序:
    1. 建表 (执行 sql/schema.sql, IF NOT EXISTS, 已有不动)
    2. 同步申万一级 + 二级分类到 trade_stock_status
    3. 全量下载所有股票 K 线到 trade_stock_daily (默认 2024-01-01 起)
    4. 全量回填 trade_sector_daily (合成板块指数)

预期耗时:
    申万二级 134 个板块, 共 ~5000 只股票, 全量约 30-90 分钟

只需执行一次, 后续每日增量用 run_daily.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_config import execute_update
from industry_meta import sync_sw_classification
from stock_kline_loader import sync_all_kline
from sector_index_builder import rebuild_all_sectors


def init_schema():
    """执行 sql/schema.sql 建表 (IF NOT EXISTS, 安全幂等)"""
    sql_path = Path(__file__).parent / "sql" / "schema.sql"
    with open(sql_path, "r", encoding="utf-8") as f:
        ddl_text = f.read()

    statements = [s.strip() for s in ddl_text.split(";") if s.strip() and not s.strip().startswith("--")]
    print(f"\n{'='*70}")
    print(f"  建表: {sql_path.name}")
    print(f"{'='*70}\n")

    for stmt in statements:
        lines = [ln for ln in stmt.split("\n") if not ln.strip().startswith("--")]
        clean = "\n".join(lines).strip()
        if not clean:
            continue
        execute_update(clean)
    print("[OK] 表结构创建完成 (3 张表)\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CASE-A 全量初始化")
    parser.add_argument("--skip-schema", action="store_true", help="跳过建表")
    parser.add_argument("--skip-meta",   action="store_true", help="跳过申万分类同步")
    parser.add_argument("--skip-kline",  action="store_true", help="跳过 K 线下载")
    parser.add_argument("--skip-index",  action="store_true", help="跳过板块指数合成")
    parser.add_argument("--start", default=None,
                        help="K 线起始日 YYYYMMDD, 默认读 .env 的 SW_INIT_START_DATE")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2,
                        help="板块指数级别 (默认 2 申万二级)")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# CASE-A 板块数据基础设施 -- 全量初始化")
    print("#" * 70)

    if not args.skip_schema:
        init_schema()
    else:
        print("\n[SKIP] 跳过建表\n")

    if not args.skip_meta:
        sync_sw_classification()
    else:
        print("\n[SKIP] 跳过申万分类同步\n")

    if not args.skip_kline:
        sync_all_kline(start_date=args.start, incremental=False, level=args.level)
    else:
        print("\n[SKIP] 跳过 K 线下载\n")

    if not args.skip_index:
        rebuild_all_sectors(level=args.level, full=True)
    else:
        print("\n[SKIP] 跳过板块指数合成\n")

    print("\n" + "#" * 70)
    print("# 初始化完成, 后续用 run_daily.py 跑增量")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
