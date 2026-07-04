# -*- coding: utf-8 -*-
"""
zoe_node -- 信号官节点

复用 20-团队架构设计/CASE-AI量化助手（nanobot）/skills/strategy-backtest/scripts/run_backtest.py
跑 MACD 回测获取最新信号 + 历史胜率，再结合 Charles 的 investment_view 决定是否出单。

输入：state["stock_code"], state["capital"], state["investment_view"], state.get("risk_verdict")
输出：state["trade_signal"]

关键逻辑：
  - 如果 Kris 上一轮给了 suggested_max_pct（仓位降级），按比例缩量重发
  - 如果 Charles 看空（bearish）且当前没有明确买入信号，输出 hold
  - 否则按 Charles 立场强度 * 信心 决定开仓比例
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from utils.env import PROJECT_ROOT, SCRIPTS_DIR

BACKTEST_SCRIPT = SCRIPTS_DIR / "run_backtest.py"


def _run_backtest(stock_code: str, strategy: str = "macd", count: int = 250) -> dict:
    """调用现成的回测脚本，返回 JSON dict"""
    cmd = [
        sys.executable,
        str(BACKTEST_SCRIPT),
        "--code", stock_code,
        "--strategy", strategy,
        "--count", str(count),
    ]
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
        timeout=180,
    )
    stdout = proc.stdout or ""
    if not stdout.strip():
        raise RuntimeError(f"Zoe 调用回测脚本无输出: {proc.stderr}")
    # xtdata 会向 stdout 打印连接日志，需要从输出中提取 JSON 块
    return _extract_json_object(stdout)


def _extract_json_object(text: str) -> dict:
    """
    从混杂文本中提取最后一个合法 JSON 对象。
    xtdata 连接日志里会出现 {'tag': 'sp3'} 这种 Python dict 表示（单引号），
    所以扫描所有 top-level {...} 块、跳过非法 JSON，返回最后一个能解析的。
    """
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


def _decide_position_pct(stance: str, confidence: float) -> float:
    """根据 Charles 立场和信心决定基础开仓比例（占可用资金）"""
    base = {"bullish": 0.30, "neutral": 0.10, "bearish": 0.0}.get(stance, 0.10)
    return base * confidence


def zoe_node(state: dict) -> dict:
    """信号节点：跑 MACD 回测 + 结合投研观点 -> TradeSignal"""
    stock = state["stock_code"]
    capital = float(state.get("capital", 100_000))
    view = state.get("investment_view", {})
    last_verdict = state.get("risk_verdict", {})

    print()
    print("=" * 70)
    print(f"[Zoe] 信号官开始工作 -- 标的: {stock}")
    print("=" * 70)

    bt = _run_backtest(stock, strategy="macd", count=250)
    if "error" in bt:
        raise RuntimeError(f"Zoe 回测失败: {bt['error']}")

    latest_signal = bt.get("latest_signal", "none")
    latest_close = float(bt.get("latest_close", 0))
    print(f"[Zoe] MACD 回测 {bt['data_range']} | 胜率 {bt['win_rate']}% | "
          f"总收益 {bt['total_return']}% | 最新信号 {latest_signal} @ {latest_close}")

    stance = view.get("stance", "neutral")
    confidence = float(view.get("confidence", 0.5))
    catalysts = view.get("catalysts", []) or []
    risks = view.get("risks", []) or []
    report_html = view.get("report_html_path", "")

    if report_html:
        print(f"[Zoe] 已读取 Charles 研报附件: {report_html}")
    if catalysts:
        print(f"[Zoe] Charles 催化剂参考: {catalysts[0][:60]}")
    if risks:
        print(f"[Zoe] Charles 风险提醒: {risks[0][:60]}")

    # 多因子决策矩阵（Charles 基本面 + MACD 技术面）
    # 完全共振 -> 标准仓位
    # 基本面强但技术面分歧 -> 小仓试探
    # 完全相反 -> hold
    if latest_signal in ("golden_cross", "bullish") and stance == "bullish":
        direction = "buy"
        pos_pct = _decide_position_pct(stance, confidence)
        reason = f"共振: MACD {latest_signal} + Charles bullish({confidence:.2f})"
    elif latest_signal in ("death_cross", "bearish") and stance == "bearish":
        direction = "sell"
        pos_pct = 0.0  # sell 量按持仓另算
        reason = f"共振: MACD {latest_signal} + Charles bearish"
    elif stance == "bullish" and confidence >= 0.7 and latest_signal in ("bearish", "death_cross"):
        # 基本面强势看多但技术面分歧 -> 小仓试探（半仓）
        direction = "buy"
        pos_pct = _decide_position_pct(stance, confidence) * 0.5
        reason = f"分歧: Charles 强势 bullish({confidence:.2f}), MACD {latest_signal} -> 小仓试探"
    elif stance == "neutral" and latest_signal in ("golden_cross", "bullish"):
        direction = "buy"
        pos_pct = _decide_position_pct(stance, confidence)
        reason = f"技术面驱动: MACD {latest_signal} + Charles 中性"
    else:
        direction = "hold"
        pos_pct = 0.0
        reason = f"观望: MACD {latest_signal} + Charles {stance}({confidence:.2f})"

    # 如果是从 Kris 重试回来的，应用上一轮建议的仓位上限
    suggested_max = float(last_verdict.get("suggested_max_pct", 1.0))
    if suggested_max < 1.0 and direction == "buy":
        old_pct = pos_pct
        pos_pct = min(pos_pct, suggested_max)
        print(f"[Zoe] 上一轮 Kris 建议仓位上限 {suggested_max:.0%}，"
              f"缩量 {old_pct:.0%} -> {pos_pct:.0%}")

    if direction == "buy" and latest_close > 0:
        amount = capital * pos_pct
        quantity = int(amount / latest_close / 100) * 100
        # 试探单至少 1 手（100 股）—— 由 Kris 的 ATR 仓位规则去判断是否过大
        if quantity == 0:
            quantity = 100
            print(f"[Zoe] 算出仓位不足 1 手，按最小试探单 100 股出单（交给 Kris 审批）")
    else:
        quantity = 0

    signal = {
        "stock_code": stock,
        "direction": direction,
        "quantity": quantity,
        "price": latest_close,
        "reason": reason,
        "strategy": bt.get("strategy", "MACD"),
        "latest_signal": latest_signal,
        "latest_close": latest_close,
        "backtest_winrate": float(bt.get("win_rate", 0)),
        "backtest_total_return": float(bt.get("total_return", 0)),
    }
    print(f"[Zoe] 决策: {direction.upper()} {quantity}股 @ {latest_close} | {reason}")

    return {
        "trade_signal": signal,
        "messages": [
            {
                "role": "zoe",
                "time": datetime.now().strftime("%H:%M:%S"),
                "content": f"{direction} {quantity}股 @ {latest_close} ({reason})",
            }
        ],
    }
