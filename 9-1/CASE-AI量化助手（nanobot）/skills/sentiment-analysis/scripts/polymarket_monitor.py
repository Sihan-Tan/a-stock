#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Polymarket 预测市场监控

功能：通过 Polymarket Gamma API 获取地缘政治、宏观经济等预测市场数据，
      分析聪明钱押注方向，为投资决策提供前瞻性信号。

Polymarket 简介：
  - 全球最大的去中心化预测市场，基于 Polygon 区块链
  - 用户用真金白银（USDC）押注事件发生的概率
  - 所有交易链上透明，可追踪大额押注
  - 2026年被洲际交易所（纽交所母公司）投资20亿美元，估值90亿美元

API 架构：
  - Gamma API：元数据层（市场描述、标签、结算日期），免费无需认证
  - CLOB API：执行层（实时价格、订单簿、成交历史）
  - 结算层：链上 UMA 预言机（争议期 2 小时）

量化策略应用：
  - 开战概率 > 60% -> 做多黄金/原油，做空科技股
  - 停火概率快速上升 -> 平仓原油多头，买入被制裁国资产
  - 概率长期僵持(40%-60%) -> 做多波动率（买入VIX看涨期权）

用法：
    python polymarket_monitor.py
    python polymarket_monitor.py --keyword "Iran"
    python polymarket_monitor.py --keyword "tariff" --min_volume 1000000
    python polymarket_monitor.py --keyword "China" --output_dir output/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import List, Optional

try:
    import httpx
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "httpx", "-q"])
    import httpx


GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# 地缘政治和宏观经济相关的搜索关键词
DEFAULT_KEYWORDS = [
    "Iran",
    "China",
    "tariff",
    "war",
    "ceasefire",
    "sanctions",
    "Fed",
    "interest rate",
    "recession",
    "oil",
    "Bitcoin",
    "S&P 500",
]

# 资产配置映射：事件概率 -> 资产操作建议
EVENT_ASSET_MAP = {
    "war": {
        "high_prob": {"action": "做多黄金/原油，做空科技股", "reason": "避险资产上涨"},
        "low_prob": {"action": "平仓避险头寸，买入风险资产", "reason": "风险情绪修复"},
    },
    "ceasefire": {
        "high_prob": {"action": "平仓原油多头，买入被制裁国资产", "reason": "和平利好风险资产"},
        "low_prob": {"action": "维持避险配置", "reason": "冲突持续"},
    },
    "tariff": {
        "high_prob": {"action": "回避出口型企业，关注内需板块", "reason": "关税冲击出口"},
        "low_prob": {"action": "关注出口复苏机会", "reason": "贸易环境改善"},
    },
    "recession": {
        "high_prob": {"action": "增配国债和黄金，减仓周期股", "reason": "经济衰退避险"},
        "low_prob": {"action": "增配成长股和周期股", "reason": "经济前景向好"},
    },
    "interest rate": {
        "high_prob": {"action": "利率上行预期，关注银行股", "reason": "加息利好银行净息差"},
        "low_prob": {"action": "降息预期，关注成长股", "reason": "低利率利好高估值标的"},
    },
}


def fetch_events(keyword: str, limit: int = 20) -> List[dict]:
    """
    从 Polymarket Gamma API 获取相关预测市场事件。

    Args:
        keyword: 搜索关键词（如 Iran, tariff, China）
        limit: 返回事件数量上限

    Returns:
        事件列表
    """
    params = {
        "active": "true",
        "closed": "false",
        "search": keyword,
        "limit": limit,
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
    }

    try:
        resp = httpx.get(
            f"{GAMMA_API_BASE}/events",
            params=params,
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data:
            print(f"[Polymarket] 关键词 '{keyword}' 未找到活跃市场")
            return []

        print(f"[Polymarket] 关键词 '{keyword}' 找到 {len(data)} 个事件")
        return data

    except httpx.HTTPStatusError as e:
        print(f"[Polymarket] API 返回错误 {e.response.status_code}: {e}")
        return []
    except httpx.ConnectError:
        print(f"[Polymarket] 无法连接 API（可能需要代理）")
        return []
    except Exception as e:
        print(f"[Polymarket] 请求失败: {e}")
        return []


def parse_markets(events: List[dict], min_volume: float = 0) -> List[dict]:
    """
    解析事件中的市场数据，提取概率、交易量等关键信息。

    Args:
        events: Polymarket 事件列表
        min_volume: 最小交易量过滤（USDC）

    Returns:
        解析后的市场数据列表
    """
    markets = []

    for event in events:
        event_title = event.get("title", "")
        event_id = event.get("id", "")

        for market in event.get("markets", []):
            # 解析概率
            outcome_prices = market.get("outcomePrices", "[]")
            if isinstance(outcome_prices, str):
                try:
                    prices = json.loads(outcome_prices)
                except (json.JSONDecodeError, TypeError):
                    prices = [0, 0]
            else:
                prices = outcome_prices

            yes_price = float(prices[0]) if len(prices) > 0 else 0
            no_price = float(prices[1]) if len(prices) > 1 else 0

            # 交易量
            volume = float(market.get("volume", 0) or 0)

            # 按最小交易量过滤
            if volume < min_volume:
                continue

            # 流动性
            liquidity = float(market.get("liquidity", 0) or 0)

            # 概率百分比
            yes_pct = round(yes_price * 100, 1)
            no_pct = round(no_price * 100, 1)

            # 信号强度判断
            if yes_pct >= 80:
                signal_strength = "极强确定性"
            elif yes_pct >= 60:
                signal_strength = "较强倾向"
            elif yes_pct >= 40:
                signal_strength = "不确定（波动率机会）"
            elif yes_pct >= 20:
                signal_strength = "较强否定"
            else:
                signal_strength = "极强否定"

            markets.append({
                "event_id": event_id,
                "event_title": event_title,
                "market_id": market.get("id", ""),
                "question": market.get("question", ""),
                "yes_probability": yes_pct,
                "no_probability": no_pct,
                "volume_usd": round(volume, 2),
                "volume_display": _format_volume(volume),
                "liquidity_usd": round(liquidity, 2),
                "signal_strength": signal_strength,
                "end_date": market.get("endDate", ""),
                "description": market.get("description", "")[:200],
            })

    # 按交易量排序
    markets.sort(key=lambda x: x["volume_usd"], reverse=True)
    return markets


def _format_volume(volume: float) -> str:
    """将交易量格式化为可读字符串"""
    if volume >= 1_000_000_000:
        return f"${volume / 1_000_000_000:.1f}B"
    elif volume >= 1_000_000:
        return f"${volume / 1_000_000:.1f}M"
    elif volume >= 1_000:
        return f"${volume / 1_000:.1f}K"
    else:
        return f"${volume:.0f}"


def detect_smart_money_signals(markets: List[dict]) -> List[dict]:
    """
    检测聪明钱信号：高交易量 + 高概率倾向的组合。

    当一个市场的交易量远超平均值，且概率高度倾斜（>80% 或 <20%），
    通常意味着有大额资金押注，可能反映内幕信息或专业分析。

    Args:
        markets: 解析后的市场数据列表

    Returns:
        聪明钱信号列表
    """
    if not markets:
        return []

    # 计算平均交易量
    volumes = [m["volume_usd"] for m in markets if m["volume_usd"] > 0]
    if not volumes:
        return []

    avg_volume = sum(volumes) / len(volumes)
    signals = []

    for m in markets:
        signal_reasons = []

        # 交易量远超平均值（3倍以上）
        if m["volume_usd"] > avg_volume * 3:
            signal_reasons.append(f"交易量 {m['volume_display']} 远超平均值 {_format_volume(avg_volume)}")

        # 概率高度倾斜
        if m["yes_probability"] >= 85 or m["yes_probability"] <= 15:
            direction = "Yes" if m["yes_probability"] >= 85 else "No"
            prob = m["yes_probability"] if direction == "Yes" else m["no_probability"]
            signal_reasons.append(f"概率高度倾向 {direction}({prob}%)，市场有强烈共识")

        # 高流动性 + 高概率（大资金有信心的标志）
        if m["liquidity_usd"] > 1_000_000 and (m["yes_probability"] >= 70 or m["yes_probability"] <= 30):
            signal_reasons.append(f"高流动性 {_format_volume(m['liquidity_usd'])} 支撑价格稳定性")

        if signal_reasons:
            signals.append({
                "question": m["question"],
                "yes_probability": m["yes_probability"],
                "volume": m["volume_display"],
                "signal_reasons": signal_reasons,
                "alert_level": "高" if len(signal_reasons) >= 2 else "中",
            })

    return signals


def generate_asset_suggestions(markets: List[dict]) -> List[dict]:
    """
    根据预测市场概率生成资产配置建议。

    Args:
        markets: 解析后的市场数据列表

    Returns:
        资产配置建议列表
    """
    suggestions = []
    question_lower_list = [(m, m["question"].lower()) for m in markets]

    for event_type, actions in EVENT_ASSET_MAP.items():
        # 查找匹配的市场
        for m, q_lower in question_lower_list:
            if event_type in q_lower:
                prob = m["yes_probability"]
                if prob >= 60:
                    suggestion = actions["high_prob"].copy()
                    suggestion["trigger"] = f"{m['question']} (概率: {prob}%)"
                    suggestion["volume"] = m["volume_display"]
                    suggestions.append(suggestion)
                elif prob <= 30:
                    suggestion = actions["low_prob"].copy()
                    suggestion["trigger"] = f"{m['question']} (概率: {prob}%)"
                    suggestion["volume"] = m["volume_display"]
                    suggestions.append(suggestion)
                break

    return suggestions


def main():
    parser = argparse.ArgumentParser(description="Polymarket 预测市场监控")
    parser.add_argument("--keyword", default=None, help="搜索关键词（如 Iran, tariff, China）")
    parser.add_argument("--keywords", default=None, help="多个关键词，逗号分隔")
    parser.add_argument("--min_volume", type=float, default=10000, help="最小交易量过滤（USDC，默认 10000）")
    parser.add_argument("--output_dir", default="./output", help="输出目录")
    parser.add_argument("--top_n", type=int, default=20, help="每个关键词返回前N个市场（默认 20）")
    args = parser.parse_args()

    # 确定搜索关键词
    search_keywords = []
    if args.keyword:
        search_keywords = [args.keyword]
    elif args.keywords:
        search_keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    else:
        # 使用默认关键词中的前5个
        search_keywords = DEFAULT_KEYWORDS[:5]
        print(f"[提示] 未指定关键词，使用默认: {search_keywords}")

    print("=" * 60)
    print("  Polymarket 预测市场监控")
    print(f"  关键词: {search_keywords}")
    print(f"  最小交易量: ${args.min_volume:,.0f}")
    print("=" * 60)

    all_markets = []
    all_signals = []

    for i, keyword in enumerate(search_keywords):
        print(f"\n[{i + 1}/{len(search_keywords)}] 搜索: {keyword}")

        # 获取事件
        events = fetch_events(keyword, limit=args.top_n)
        if not events:
            continue

        # 解析市场数据
        markets = parse_markets(events, min_volume=args.min_volume)
        if not markets:
            print(f"  没有交易量超过 ${args.min_volume:,.0f} 的市场")
            continue

        print(f"  找到 {len(markets)} 个活跃市场")

        # 显示Top市场
        for j, m in enumerate(markets[:5]):
            print(f"  [{j + 1}] {m['question'][:60]}")
            print(f"      Yes: {m['yes_probability']}% | No: {m['no_probability']}% | 交易量: {m['volume_display']}")

        all_markets.extend(markets)

        # 检测聪明钱信号
        signals = detect_smart_money_signals(markets)
        if signals:
            all_signals.extend(signals)

    if not all_markets:
        print("\n[结果] 未找到任何活跃的预测市场")
        result = {
            "status": "no_data",
            "message": "未找到符合条件的预测市场",
            "keywords": search_keywords,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 去重（同一个market_id可能被多个关键词命中）
    seen_ids = set()
    unique_markets = []
    for m in all_markets:
        if m["market_id"] not in seen_ids:
            seen_ids.add(m["market_id"])
            unique_markets.append(m)
    all_markets = unique_markets

    # 生成资产配置建议
    asset_suggestions = generate_asset_suggestions(all_markets)

    # 保存结果
    os.makedirs(args.output_dir, exist_ok=True)
    date_tag = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(args.output_dir, f"polymarket_{date_tag}.json")

    output_data = {
        "search_keywords": search_keywords,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_markets": len(all_markets),
        "markets": all_markets,
        "smart_money_signals": all_signals,
        "asset_suggestions": asset_suggestions,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {output_file}")

    # 打印摘要
    print(f"\n{'=' * 60}")
    print(f"  [汇总] 共监控 {len(all_markets)} 个预测市场")

    # 按交易量排序，显示Top 10
    top_markets = sorted(all_markets, key=lambda x: x["volume_usd"], reverse=True)[:10]
    print(f"\n  [交易量 Top 10]")
    for i, m in enumerate(top_markets):
        print(f"  {i + 1}. {m['question'][:55]}")
        print(f"     Yes: {m['yes_probability']}% | 交易量: {m['volume_display']} | {m['signal_strength']}")

    # 聪明钱信号
    if all_signals:
        print(f"\n  [聪明钱信号] 检测到 {len(all_signals)} 个信号")
        for s in all_signals:
            print(f"  - [{s['alert_level']}] {s['question'][:50]}")
            print(f"    Yes概率: {s['yes_probability']}% | 交易量: {s['volume']}")
            for reason in s["signal_reasons"]:
                print(f"    -> {reason}")

    # 资产配置建议
    if asset_suggestions:
        print(f"\n  [资产配置建议]")
        for s in asset_suggestions:
            print(f"  - {s['action']}")
            print(f"    原因: {s['reason']}")
            print(f"    触发: {s['trigger']}")

    print(f"{'=' * 60}")

    # 输出结构化结果
    summary = {
        "status": "success",
        "total_markets": len(all_markets),
        "smart_money_signals": len(all_signals),
        "asset_suggestions": len(asset_suggestions),
        "output_file": output_file,
        "top_market": top_markets[0]["question"] if top_markets else None,
        "top_market_yes_pct": top_markets[0]["yes_probability"] if top_markets else None,
    }
    print(f"\n[结果] {json.dumps(summary, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
