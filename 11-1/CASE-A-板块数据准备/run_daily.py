# -*- coding: utf-8 -*-
# 21-CASE-A: 每日增量
"""
run_daily.py -- 每日增量

执行顺序:
    1. 增量下载所有股票 K 线 (按 trade_stock_daily 中已有的 max(trade_date)+1)
    2. 重算最近 60 个交易日的 trade_sector_daily
       (60 天足够覆盖任何延迟到位的成分股 K 线 + 重新校验 close_idx 链)
    3. 默认不刷新 trade_stock_status 申万分类 (申万官方分类一年才调一次)
       学员手工觉得需要时, 用 --refresh-meta 触发

预期耗时:
    单日增量约 5-15 分钟, 建议 8:30 cron 触发, 给晨会 (9:00) 留充足准备时间
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from industry_meta import sync_sw_classification
from stock_kline_loader import sync_all_kline
from sector_index_builder import rebuild_all_sectors


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CASE-A 每日增量")
    parser.add_argument("--refresh-meta", action="store_true",
                        help="强制刷新申万分类 (默认不刷新, 一周或一月触发一次即可)")
    parser.add_argument("--days", type=int, default=60,
                        help="重算板块指数的最近交易日数 (默认 60)")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# CASE-A 板块数据 -- 每日增量")
    print("#" * 70)

    if args.refresh_meta:
        sync_sw_classification()

    sync_all_kline(incremental=True, level=args.level)

    rebuild_all_sectors(level=args.level, days=args.days)

    print("\n" + "#" * 70)
    print("# 每日增量完成")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
