# -*- coding: utf-8 -*-
# 21-CASE-C 多因子选股: 选股池构造 + 行业归类
"""
StockPool -- 选股池工具

为什么先做这个? 多因子选股的第一步是"先确定在哪个池子里选":
    - 不能在全市场 5000 只里选 -- 大量退市/ST/流动性差, 噪音太大
    - 标准做法: 在沪深 300 / 中证 500 / 中证 1000 等指数成分股里选

xtdata API:
    xtdata.get_stock_list_in_sector("沪深300")    -> 拉成分股列表
    xtdata.get_instrument_detail(code)            -> 拉个股详情 (含申万行业代码)

本模块提供:
    - get_csi300()            -- 拉沪深 300 成分股
    - get_csi500()            -- 拉中证 500 成分股
    - get_industry_map(codes) -- 把股票代码映射到申万一级行业
    - filter_tradable(codes)  -- 过滤掉 ST / 停牌 / 上市不足 N 天
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict, Optional


def get_index_components(index_name: str) -> List[str]:
    """
    拉指数成分股
    
    支持的 index_name:
        "沪深300"  -> 沪深 300 成分股
        "中证500"  -> 中证 500
        "中证1000" -> 中证 1000
    """
    from xtquant import xtdata
    xtdata.connect()
    codes = xtdata.get_stock_list_in_sector(index_name)
    if not codes:
        raise RuntimeError(f"未拉到 [{index_name}] 成分股, "
                           f"检查 miniQMT 是否启动 / 板块名是否正确")
    return sorted(codes)


def get_csi300() -> List[str]:
    return get_index_components("沪深300")


def get_csi500() -> List[str]:
    return get_index_components("中证500")


def get_industry_map(stock_codes: List[str]) -> Dict[str, str]:
    """
    把股票代码映射到申万一级行业
    
    实现思路：xtdata 没有直接的 stock -> industry 接口, 但有
        get_sector_list()  -> 拿到所有板块名
        get_stock_list_in_sector("SW1银行")  -> 拿这个板块的成分股
    我们反向构建索引: 遍历所有 SW1 板块, 把成分股都标注上行业名
    
    返回: {stock_code: industry_name}
    """
    from xtquant import xtdata
    xtdata.connect()

    target_set = set(stock_codes)
    result: Dict[str, str] = {code: "未分类" for code in stock_codes}

    # 拿所有 SW1 (申万一级) 板块, 排除"加权"版本
    all_sectors = xtdata.get_sector_list() or []
    sw1_sectors = [s for s in all_sectors
                   if s.startswith("SW1") and not s.endswith("加权")]

    for sector in sw1_sectors:
        # 行业名 = 去掉 SW1 前缀
        industry_name = sector.replace("SW1", "", 1) or sector
        try:
            members = xtdata.get_stock_list_in_sector(sector) or []
        except Exception:
            continue
        for code in members:
            if code in target_set:
                result[code] = industry_name

    return result


def filter_tradable(stock_codes: List[str],
                    min_listed_days: int = 250) -> List[str]:
    """
    过滤可交易股票:
        - 排除 ST / *ST
        - 排除已退市
        - 排除上市不足 N 天 (默认 250 日 = 1 年)
    """
    from datetime import date, datetime
    from xtquant import xtdata
    xtdata.connect()

    today = date.today()
    keep = []
    for code in stock_codes:
        try:
            detail = xtdata.get_instrument_detail(code)
            if not detail:
                continue
            name = detail.get("InstrumentName", "")
            if "ST" in name.upper() or "退" in name:
                continue
            # ExpireDate 已退市股票为真实退市日 (yyyymmdd), 在售股票为 99999999 / "0" / 空
            expire = str(detail.get("ExpireDate", "")).strip()
            if expire and expire not in ("0", "99999999"):
                continue
            listed = str(detail.get("OpenDate", "")).strip()
            if listed and listed not in ("0", ""):
                try:
                    listed_dt = datetime.strptime(listed, "%Y%m%d").date()
                    days = (today - listed_dt).days
                    if days < min_listed_days:
                        continue
                except Exception:
                    pass
            keep.append(code)
        except Exception:
            continue
    return keep


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="选股池工具")
    parser.add_argument("--index", default="沪深300", help="指数名: 沪深300/中证500/中证1000")
    parser.add_argument("--max", type=int, default=10, help="演示时只列前 N 只")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  选股池: {args.index}")
    print(f"{'='*60}\n")

    codes = get_index_components(args.index)
    print(f"原始成分股: {len(codes)} 只")
    print(f"前 {args.max} 只: {codes[:args.max]}")

    print(f"\n[FILTER] 过滤 ST / 已退市 / 上市不足 1 年 ...")
    tradable = filter_tradable(codes)
    print(f"可交易股票: {len(tradable)} 只 (剔除了 {len(codes) - len(tradable)} 只)")

    print(f"\n[INDUSTRY] 拉申万行业 (前 {args.max} 只) ...")
    sample = tradable[:args.max]
    ind_map = get_industry_map(sample)
    for code in sample:
        print(f"  {code} -> {ind_map.get(code, '未分类')}")


if __name__ == "__main__":
    main()
