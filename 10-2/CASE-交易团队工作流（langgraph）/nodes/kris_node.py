# -*- coding: utf-8 -*-
"""
kris_node -- 风控官节点

复用 19-打造你的风控体系/CASE-Kris的风控体系/1-风控引擎.py 的 RiskManager。
顺便从最近 60 根日 K 线计算 ATR(14)，喂给海龟仓位规则。

输入：state["trade_signal"], state["capital"], state["investment_view"]
输出：state["risk_verdict"]
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from utils.env import LIB_DIR, PROJECT_ROOT, SCRIPTS_DIR

# 把 lib 加到 sys.path，便于 import vendored 模块
if str(LIB_DIR) not in sys.path:
    sys.path.insert(0, str(LIB_DIR))

from risk_engine import Decision, Order, RiskManager  # noqa: E402  vendored from 19章

KLINE_SCRIPT = SCRIPTS_DIR / "get_kline.py"


def _extract_json_object(text: str) -> dict:
    """从混杂文本中提取最后一个合法 JSON 对象（与 zoe_node 同款）"""
    blocks = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append(text[start:i + 1])
                start = -1
    for blk in reversed(blocks):
        try:
            data = json.loads(blk)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    raise RuntimeError(f"未找到合法 JSON 对象，输出尾部: ...{text[-300:]}")


def _fetch_kline_atr(stock_code: str, period: int = 14) -> float:
    """通过 stock-price 脚本获取最近 K 线，计算 ATR(14)"""
    cmd = [sys.executable, str(KLINE_SCRIPT), stock_code, "1d", "60"]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        cmd,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=120,
    )
    # xtdata 连接日志里会混入 Python dict 表示（单引号），扫描全部块取最后一个合法 JSON
    text = proc.stdout or ""
    data = _extract_json_object(text)
    if "error" in data or not data.get("data"):
        raise RuntimeError(f"Kris 获取 K 线失败: {data}")

    bars = data["data"]
    trs = []
    for i in range(1, len(bars)):
        h = bars[i]["high"]
        l = bars[i]["low"]
        pc = bars[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return sum(trs) / max(len(trs), 1)
    return sum(trs[-period:]) / period


def _build_news_text(view: dict) -> str:
    """把 Charles 的风险点 + 催化剂拼成新闻文本，供事件关键词检查"""
    parts = [view.get("summary", "")]
    parts.extend(view.get("risks", []))
    parts.extend(view.get("catalysts", []))
    return " ".join(p for p in parts if p)


def kris_node(state: dict) -> dict:
    """风控节点：跑 8 条规则 -> RiskVerdict"""
    signal = state.get("trade_signal", {})
    view = state.get("investment_view", {})
    capital = float(state.get("capital", 100_000))

    print()
    print("=" * 70)
    print("[Kris] 风控官开始审批")
    print("=" * 70)

    # hold 或 0 股，无需走风控
    if signal.get("direction") == "hold" or signal.get("quantity", 0) <= 0:
        verdict = {
            "decision": "approve",
            "is_approved": True,
            "reason": "信号为 hold 或 0 股，无需风控",
            "rule_name": "空单跳过",
            "suggested_max_pct": 1.0,
        }
        print(f"[Kris] {verdict['reason']}")
        return {
            "risk_verdict": verdict,
            "messages": [{
                "role": "kris",
                "time": datetime.now().strftime("%H:%M:%S"),
                "content": "空单跳过",
            }],
        }

    stock = signal["stock_code"]
    price = float(signal["price"])
    qty = int(signal["quantity"])
    direction = signal["direction"]
    amount = price * qty

    atr_value = _fetch_kline_atr(stock)
    print(f"[Kris] {stock} ATR(14) = {atr_value:.3f}")

    kris = RiskManager(config={
        "pre_trade": {
            "max_order_amount": max(capital * 0.5, 100_000),
            "price_collar_pct": 0.05,
            "blacklist": [],
            "atr_risk_pct": 0.01,
            "atr_overshoot_ratio": 2.0,
        },
        "circuit_breaker": {"max_daily_loss_pct": 0.02},
    })
    kris.start_day(capital)
    kris.macro.update_vix(18.5)  # 演示用：真实可挂 VIX 数据源

    portfolio = {
        "total_asset": capital,
        "prices": {stock: price},
        "atr": {stock: atr_value},
    }
    order = Order(stock_code=stock, direction=direction, amount=amount, price=price)

    decision = kris.approve(order, portfolio, context={"news_text": _build_news_text(view)})
    is_approved = decision.is_approved
    suggested_pct = decision.max_position_pct if decision.decision == Decision.WARN else 1.0

    verdict = {
        "decision": decision.decision.value,
        "is_approved": is_approved,
        "reason": decision.reason,
        "rule_name": decision.rule_name,
        "suggested_max_pct": float(suggested_pct),
    }
    icon = "PASS" if is_approved else "REJECT"
    print(f"[Kris] [{icon}] {decision.rule_name}: {decision.reason[:120]}")
    if suggested_pct < 1.0:
        print(f"[Kris] 建议仓位上限 {suggested_pct:.0%}（Zoe 应缩量重发）")

    return {
        "risk_verdict": verdict,
        "messages": [{
            "role": "kris",
            "time": datetime.now().strftime("%H:%M:%S"),
            "content": f"{decision.decision.value} -- {decision.reason[:80]}",
        }],
    }
