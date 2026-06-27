#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场恐慌指数监控

功能：获取全球市场恐慌/贪婪相关指标，计算综合风险评分，
      为投资决策提供宏观维度的情绪参考。

监控指标体系：
  波动率维度：
    - VIX（美股恐慌指数）：衡量标普500的隐含波动率
    - OVX（原油ETF波动率）：衡量能源供应风险
    - GVZ（黄金ETF波动率）：衡量避险情绪
  利率维度：
    - 美国10年期国债收益率：全球资产定价之锚
  A股维度：
    - 上证指数近期涨跌幅
    - 北向资金流向

风险传导逻辑（来自开源证券研报）：
  - OVX飙升但VIX滞后 -> 风险集中在能源端，尚未传导至全球信用风险
  - OVX与VIX同步共振向上 -> 地缘风险已触发流动性危机，需立即风控

数据来源：
  - VIX / OVX / GVZ：FRED 公开 CSV（VIXCLS / OVXCLS / GVZCLS），无需 API Key
  - 美国10年期国债收益率：akshare bond_zh_us_rate

用法：
    python market_fear_index.py
    python market_fear_index.py --output_dir output/
    python market_fear_index.py --include_ashare
"""

import argparse
import json
import os
import sys
from datetime import datetime

import requests

# VIX 历史阈值参考
VIX_THRESHOLDS = {
    "极度恐慌": 35,   # VIX > 35：市场极度恐慌（如2020年3月疫情、2008年金融危机）
    "恐慌": 25,       # VIX 25-35：市场明显紧张
    "焦虑": 20,       # VIX 20-25：市场有一定担忧
    "正常": 15,       # VIX 15-20：正常波动范围
    "平静": 0,        # VIX < 15：市场极度平静（可能是暴风雨前的宁静）
}

# 美国10年期国债收益率阈值
TREASURY_THRESHOLDS = {
    "高利率压制": 4.8,     # > 4.8%：严重压制成长股估值
    "偏紧": 4.4,          # 4.4-4.8%：利率偏高，利好价值股
    "分水岭": 4.3,        # 4.3%附近：关键分水岭
    "宽松预期": 3.8,      # < 3.8%：市场预期降息，利好成长股
}

# FRED 序列 ID 映射
FRED_SERIES = {
    "VIX": "VIXCLS",
    "OVX": "OVXCLS",
    "GVZ": "GVZCLS",
}


def _fred_latest(series_id: str) -> tuple:
    """
    从 FRED 公开 CSV 获取最后一笔有效观测值。
    地址：https://fred.stlouisfed.org/graph/fredgraph.csv?id=<series_id>
    无需 API Key。

    Returns:
        (value, date_str) 或 (None, None)
    """
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    last_val, last_date = None, None
    for line in resp.text.strip().splitlines():
        if line.startswith("DATE"):
            continue
        parts = line.split(",", 1)
        if len(parts) < 2:
            continue
        raw = parts[1].strip()
        if raw == "" or raw == ".":
            continue
        try:
            last_val = float(raw)
            last_date = parts[0].strip()
        except ValueError:
            continue
    return last_val, last_date


def fetch_vix() -> dict:
    """获取 VIX 恐慌指数（FRED VIXCLS）。"""
    value, date = _fred_latest(FRED_SERIES["VIX"])
    if value is None:
        return {"indicator": "VIX", "value": None, "error": "FRED VIXCLS 数据获取失败"}

    risk_level = "平静"
    for level, threshold in VIX_THRESHOLDS.items():
        if value >= threshold:
            risk_level = level
            break

    print(f"[VIX] {value:.2f}, 风险等级: {risk_level}, 日期: {date}")
    return {
        "indicator": "VIX",
        "name": "标普500波动率指数",
        "value": round(value, 2),
        "date": date,
        "risk_level": risk_level,
        "description": "衡量美股市场的恐慌程度",
        "source": "FRED (VIXCLS)",
    }


def fetch_us_treasury_10y() -> dict:
    """
    获取美国10年期国债收益率（akshare bond_zh_us_rate）。

    > 4.4%: 更看好价值/防御品种
    < 4.3%: 资金可能回流成长股
    """
    import akshare as ak

    df = ak.bond_zh_us_rate(start_date="20240101")
    columns = df.columns.tolist()
    print(f"[10Y国债] 数据列: {columns}")

    # 定位 "美国国债收益率10年" 列
    target_col = None
    for col in columns:
        if "美国" in col and "10年" in col and "10年-" not in col:
            target_col = col
            break

    if target_col is None:
        return {"indicator": "US10Y", "value": None, "error": "未找到美国10年列"}

    # 取最后一个非 NaN 值（当日可能尚未更新）
    valid = df.dropna(subset=[target_col])
    if valid.empty:
        return {"indicator": "US10Y", "value": None, "error": "美国10年列全部为空"}

    latest = valid.iloc[-1]
    yield_value = float(latest[target_col])
    yield_date = str(latest["日期"]) if "日期" in columns else ""

    # 策略评估
    if yield_value >= TREASURY_THRESHOLDS["高利率压制"]:
        strategy, risk_level = "高利率压制成长股估值，关注价值股和防御板块", "高"
    elif yield_value >= TREASURY_THRESHOLDS["偏紧"]:
        strategy, risk_level = "利率偏高，更看好科技股的估值修复机会", "偏高"
    elif yield_value >= TREASURY_THRESHOLDS["分水岭"]:
        strategy, risk_level = "利率处于分水岭附近，密切关注方向选择", "中性"
    else:
        strategy, risk_level = "利率走低，资金回流成长股，利好科技和新能源", "偏低"

    print(f"[10Y国债] {yield_value:.3f}%, 策略: {strategy}")
    return {
        "indicator": "US10Y",
        "name": "美国10年期国债收益率",
        "value": round(yield_value, 3),
        "unit": "%",
        "date": yield_date,
        "risk_level": risk_level,
        "strategy_hint": strategy,
        "thresholds": {
            ">4.4%": "偏紧，看好价值股",
            "<4.3%": "宽松预期，看好成长股",
        },
        "description": "全球资产定价之锚",
        "source": "akshare",
    }


def fetch_ovx_gvz() -> list:
    """
    获取 OVX（原油波动率）和 GVZ（黄金波动率）。

    OVX飙升 + VIX滞后 -> 风险仅在能源端
    OVX + VIX同步飙升 -> 全球性流动性危机
    """
    results = []
    indicators = [
        ("OVX", "原油ETF波动率指数", "衡量能源供应风险和地缘政治风险传导"),
        ("GVZ", "黄金ETF波动率指数", "衡量避险情绪强度"),
    ]
    for code, name, desc in indicators:
        value, date = _fred_latest(FRED_SERIES[code])
        if value is None:
            results.append({"indicator": code, "value": None, "error": f"FRED {FRED_SERIES[code]} 数据获取失败"})
            continue
        print(f"[{code}] {value:.2f}, 日期: {date}")
        results.append({
            "indicator": code,
            "name": name,
            "value": round(value, 2),
            "date": date,
            "description": desc,
            "source": f"FRED ({FRED_SERIES[code]})",
        })
    return results


def fetch_ashare_sentiment() -> dict:
    """获取A股市场情绪指标：上证指数近期表现 + 北向资金流向。"""
    import akshare as ak

    result = {}

    # 上证指数
    try:
        df = ak.stock_zh_index_daily(symbol="sh000001")
        if df is not None and not df.empty:
            recent = df.tail(10)
            latest_close = float(recent.iloc[-1]["close"])
            prev_close = float(recent.iloc[0]["close"])
            change_pct = ((latest_close - prev_close) / prev_close) * 100
            result["shanghai_index"] = {
                "name": "上证指数",
                "latest_close": round(latest_close, 2),
                "10d_change_pct": round(change_pct, 2),
                "trend": "上涨" if change_pct > 1 else ("下跌" if change_pct < -1 else "震荡"),
            }
            print(f"[上证指数] {latest_close:.2f}, 近10日涨跌幅: {change_pct:.2f}%")
    except Exception as e:
        print(f"[上证指数] 获取失败: {e}")

    # 北向资金
    try:
        df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
        if df is not None and not df.empty:
            latest = df.iloc[-1]
            columns = df.columns.tolist()
            print(f"[北向资金] 数据列: {columns}")
            net_flow = None
            for col in columns:
                if "净流入" in col or "net" in col.lower():
                    val = latest[col]
                    if val is not None and str(val) != "nan":
                        net_flow = float(val)
                        break
            if net_flow is not None:
                result["north_flow"] = {
                    "name": "北向资金净流入",
                    "value": round(net_flow, 2),
                    "unit": "亿元",
                    "signal": "外资流入" if net_flow > 0 else "外资流出",
                }
                print(f"[北向资金] 净流入: {net_flow:.2f} 亿元")
    except Exception as e:
        print(f"[北向资金] 获取失败: {e}")

    return result


def compute_composite_score(indicators: list) -> dict:
    """
    根据多个指标计算综合恐慌/贪婪评分。

    评分规则：
      VIX:
        < 15 -> +30 (极度贪婪)
        15-20 -> +15 (偏贪婪)
        20-25 -> 0 (中性)
        25-35 -> -15 (恐慌)
        > 35 -> -30 (极度恐慌)

      US10Y:
        < 3.8% -> +10 (宽松利好)
        3.8-4.3% -> +5
        4.3-4.4% -> 0 (中性)
        4.4-4.8% -> -5
        > 4.8% -> -10 (紧缩利空)

    最终映射到 0-100 分。
    """
    score = 50  # 基准分50（中性）
    score_details = []

    for ind in indicators:
        if ind.get("value") is None:
            continue

        indicator = ind["indicator"]
        value = ind["value"]

        if indicator == "VIX":
            if value < 15:
                delta, note = 30, "VIX极低，市场极度平静（警惕自满）"
            elif value < 20:
                delta, note = 15, "VIX正常偏低，市场情绪偏乐观"
            elif value < 25:
                delta, note = 0, "VIX处于正常区间"
            elif value < 35:
                delta, note = -15, "VIX偏高，市场存在恐慌情绪"
            else:
                delta, note = -30, "VIX极高，市场极度恐慌（可能是抄底机会）"
            score += delta
            score_details.append({"indicator": "VIX", "value": value, "score_delta": delta, "note": note})

        elif indicator == "US10Y":
            if value < 3.8:
                delta, note = 10, "利率走低，宽松预期利好成长股"
            elif value < 4.3:
                delta, note = 5, "利率适中，市场环境较友好"
            elif value < 4.4:
                delta, note = 0, "利率处于分水岭，密切关注"
            elif value < 4.8:
                delta, note = -5, "利率偏高，成长股估值承压"
            else:
                delta, note = -10, "高利率环境，风险资产承压"
            score += delta
            score_details.append({"indicator": "US10Y", "value": value, "score_delta": delta, "note": note})

        elif indicator == "OVX":
            if value < 25:
                delta, note = 5, "原油波动率低，能源市场稳定"
            elif value < 40:
                delta, note = 0, "原油波动率正常"
            else:
                delta, note = -10, "原油波动率高，地缘政治风险可能升级"
            score += delta
            score_details.append({"indicator": "OVX", "value": value, "score_delta": delta, "note": note})

    # 限制在 0-100 范围内
    score = max(0, min(100, score))

    # 风险传导分析
    contagion_analysis = None
    vix_data = next((i for i in indicators if i["indicator"] == "VIX" and i.get("value")), None)
    ovx_data = next((i for i in indicators if i["indicator"] == "OVX" and i.get("value")), None)

    if vix_data and ovx_data:
        vix_val = vix_data["value"]
        ovx_val = ovx_data["value"]

        if ovx_val > 40 and vix_val < 25:
            contagion_analysis = {
                "pattern": "OVX飙升但VIX滞后",
                "interpretation": "风险仍集中在能源端，尚未传导至全球宏观信用风险",
                "action": "关注能源板块风险，但无需全面避险",
            }
        elif ovx_val > 40 and vix_val > 25:
            contagion_analysis = {
                "pattern": "OVX与VIX同步共振向上",
                "interpretation": "地缘风险已触发流动性危机或全球经济衰退预期",
                "action": "需立即风控，增配黄金和现金类资产",
            }

    # 整体情绪判定
    if score >= 80:
        overall = "极度贪婪"
        action = "市场可能过热，注意回调风险"
    elif score >= 65:
        overall = "贪婪"
        action = "市场情绪偏乐观，可顺势但控制仓位"
    elif score >= 45:
        overall = "中性"
        action = "市场情绪平衡，按策略正常操作"
    elif score >= 30:
        overall = "恐慌"
        action = "市场存在恐慌，可关注超跌反弹机会"
    else:
        overall = "极度恐慌"
        action = "市场极度恐慌，历史上往往是中长期买入良机"

    return {
        "composite_fear_greed_index": score,
        "overall_sentiment": overall,
        "action_suggestion": action,
        "score_details": score_details,
        "contagion_analysis": contagion_analysis,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    parser = argparse.ArgumentParser(description="市场恐慌指数监控")
    parser.add_argument("--output_dir", default="./output", help="输出目录")
    parser.add_argument("--include_ashare", action="store_true", help="包含A股情绪指标")
    args = parser.parse_args()

    print("=" * 60)
    print("  市场恐慌指数监控")
    print("=" * 60)

    all_indicators = []

    # 1. VIX
    print("\n[1/4] 获取 VIX 恐慌指数...")
    vix = fetch_vix()
    all_indicators.append(vix)

    # 2. 美国10年期国债收益率
    print("\n[2/4] 获取美国10年期国债收益率...")
    treasury = fetch_us_treasury_10y()
    all_indicators.append(treasury)

    # 3. OVX / GVZ
    print("\n[3/4] 获取 OVX/GVZ 波动率...")
    ovx_gvz = fetch_ovx_gvz()
    all_indicators.extend(ovx_gvz)

    # 4. A股情绪（可选）
    ashare_data = {}
    if args.include_ashare:
        print("\n[4/4] 获取A股市场情绪...")
        ashare_data = fetch_ashare_sentiment()
    else:
        print("\n[4/4] 跳过A股情绪（使用 --include_ashare 开启）")

    # 计算综合评分
    print("\n[计算] 综合恐慌/贪婪指数...")
    composite = compute_composite_score(all_indicators)

    # 汇总结果
    result = {
        "indicators": all_indicators,
        "ashare_sentiment": ashare_data if ashare_data else None,
        "composite": composite,
    }

    # 保存
    os.makedirs(args.output_dir, exist_ok=True)
    date_tag = datetime.now().strftime("%Y%m%d")
    output_file = os.path.join(args.output_dir, f"fear_index_{date_tag}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {output_file}")

    # 打印摘要
    print(f"\n{'=' * 60}")
    print(f"  综合恐慌/贪婪指数: {composite['composite_fear_greed_index']}/100")
    print(f"  整体情绪: {composite['overall_sentiment']}")
    print(f"  建议操作: {composite['action_suggestion']}")
    print()

    for detail in composite.get("score_details", []):
        print(f"  [{detail['indicator']}] {detail['value']} -> {detail['note']}")

    if composite.get("contagion_analysis"):
        ca = composite["contagion_analysis"]
        print(f"\n  [风险传导] {ca['pattern']}")
        print(f"  解读: {ca['interpretation']}")
        print(f"  操作: {ca['action']}")

    print(f"{'=' * 60}")

    # 输出结构化结果
    summary = {
        "status": "success",
        "fear_greed_index": composite["composite_fear_greed_index"],
        "overall_sentiment": composite["overall_sentiment"],
        "action_suggestion": composite["action_suggestion"],
        "output_file": output_file,
    }
    print(f"\n[结果] {json.dumps(summary, ensure_ascii=False, indent=2)}")


if __name__ == "__main__":
    main()
