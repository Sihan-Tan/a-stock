# -*- coding: utf-8 -*-
"""
human_node -- 人在回路节点

LangGraph 的 interrupt 机制：
  - 节点内调用 interrupt(payload) 会把 payload 抛回给上层调用者
  - 调用者收集人类回复后，通过 graph.invoke(Command(resume=...)) 继续往下走
  - 状态全程被 Checkpointer 持久化，崩溃 / 重启都能恢复

这是 Workflow 模式相对 Agent 模式最大的优势之一：
  Agent 模式想做"必须人类确认才下单"得靠 prompt 约束（不可靠），
  Workflow 模式直接用 interrupt 在图里硬卡住。
"""

from datetime import datetime
from langgraph.types import interrupt


def human_review_node(state: dict) -> dict:
    """人在回路节点：把 Charles+Zoe+Kris 的全部信息抛给人类审批"""
    view = state.get("investment_view", {})
    signal = state.get("trade_signal", {})
    verdict = state.get("risk_verdict", {})

    payload = {
        "type": "trade_approval_request",
        "stock": signal.get("stock_code"),
        "direction": signal.get("direction"),
        "quantity": signal.get("quantity"),
        "price": signal.get("price"),
        "amount": signal.get("price", 0) * signal.get("quantity", 0),
        "charles_stance": view.get("stance"),
        "charles_confidence": view.get("confidence"),
        "charles_summary": view.get("summary"),
        "charles_report_html": view.get("report_html_path", ""),
        "zoe_reason": signal.get("reason"),
        "zoe_winrate": signal.get("backtest_winrate"),
        "kris_decision": verdict.get("decision"),
        "kris_reason": verdict.get("reason"),
        "prompt": "是否授权下单? 回复 yes / no",
    }

    user_decision = interrupt(payload)

    if isinstance(user_decision, dict):
        approved = bool(user_decision.get("approved", False))
        note = str(user_decision.get("note", ""))
    else:
        text = str(user_decision).strip().lower()
        approved = text in ("y", "yes", "ok", "approve", "1", "true")
        note = "" if approved else "用户拒绝"

    print()
    print("=" * 70)
    print(f"[Human] 用户决定: {'APPROVED' if approved else 'REJECTED'} {note}")
    print("=" * 70)

    return {
        "approved": approved,
        "messages": [{
            "role": "human",
            "time": datetime.now().strftime("%H:%M:%S"),
            "content": f"{'approved' if approved else 'rejected'} {note}",
        }],
    }
