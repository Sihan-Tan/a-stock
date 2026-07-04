# -*- coding: utf-8 -*-
"""
团队工作流入口 -- LangGraph 编排 Charles + Zoe + Kris + Trader

运行示例：
    python main.py                                       # 默认 600519.SH，10w 资金
    python main.py --stock 600519.SH --capital 200000
    python main.py --stock 002594.SZ --auto-approve      # 自动放行人审，方便 demo
    python main.py --export-graph                        # 仅导出 mermaid 图
"""

import argparse
import os
import sys
from pathlib import Path

# 让 nodes / utils / state 等同级模块可被 import
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.env import setup_utf8

setup_utf8()

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from langgraph.types import Command  # noqa: E402

from graph import build_graph, export_mermaid  # noqa: E402


def _print_summary(state: dict):
    print()
    print("=" * 70)
    print("                团队工作流执行汇总")
    print("=" * 70)
    view = state.get("investment_view", {}) or {}
    signal = state.get("trade_signal", {}) or {}
    verdict = state.get("risk_verdict", {}) or {}
    result = state.get("trade_result", {}) or {}

    print(f"标的:        {state.get('stock_code')}")
    print(f"可用资金:    {state.get('capital'):,.0f}")
    print()
    print(f"[Charles] {view.get('stance')} (信心 {view.get('confidence', 0):.2f}) "
          f"-- {view.get('summary', '')[:60]}")
    print(f"[Zoe    ] {signal.get('direction')} {signal.get('quantity', 0)} 股 "
          f"@ {signal.get('price', 0)} -- {signal.get('reason', '')[:60]}")
    print(f"[Kris   ] {verdict.get('decision')} -- {verdict.get('reason', '')[:80]}")
    print(f"[Human  ] approved = {state.get('approved')}")
    print(f"[Trader ] {result.get('note', '')}")
    print()
    print("--- 节点对话历史 ---")
    for m in state.get("messages", []):
        print(f"  [{m.get('time')}] {m.get('role'):8s} | {m.get('content')}")
    print("=" * 70)


def run_workflow(stock: str, capital: float, auto_approve: bool, question: str):
    """跑一次完整工作流"""
    graph = build_graph(with_checkpointer=True)
    thread_id = f"run-{stock}-{os.getpid()}"
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    initial_state = {
        "stock_code": stock,
        "capital": capital,
        "user_question": question,
        "retry_count": 0,
        "max_retry": 2,
        "messages": [],
    }

    print()
    print("#" * 70)
    print(f"# 团队工作流启动 -- 标的 {stock} | 资金 {capital:,.0f} "
          f"| auto-approve={auto_approve}")
    print("#" * 70)

    # 第一段：跑到 human interrupt
    state = graph.invoke(initial_state, config=config)

    # 检查是否被 human_review interrupt 卡住
    snapshot = graph.get_state(config)
    next_nodes = snapshot.next
    if next_nodes and "human" in next_nodes:
        # 拿出 interrupt 的 payload 给用户
        interrupts = snapshot.tasks[0].interrupts if snapshot.tasks else []
        if interrupts:
            payload = interrupts[0].value
            print()
            print("=" * 70)
            print("[人在回路] 待用户授权下单：")
            print("=" * 70)
            for k in ("stock", "direction", "quantity", "price", "amount",
                      "charles_stance", "charles_confidence", "charles_summary",
                      "zoe_reason", "zoe_winrate",
                      "kris_decision", "kris_reason"):
                print(f"  {k:22s}: {payload.get(k)}")

        if auto_approve:
            user_reply = "yes"
            print("\n[自动模式] 自动回复 yes")
        else:
            user_reply = input("\n是否授权下单 (yes/no): ").strip() or "no"

        # 第二段：恢复执行直到结束
        state = graph.invoke(Command(resume=user_reply), config=config)

    _print_summary(state)
    return state


def main():
    parser = argparse.ArgumentParser(description="AI 量化交易团队工作流（LangGraph 版）")
    parser.add_argument("--stock", default="600519.SH", help="股票代码，默认 600519.SH")
    parser.add_argument("--capital", type=float, default=100_000, help="可用资金")
    parser.add_argument("--question", default="", help="给 Charles 的问题，留空自动生成")
    parser.add_argument("--auto-approve", action="store_true", help="自动放行人审，方便演示")
    parser.add_argument("--export-graph", action="store_true",
                        help="仅导出 mermaid 工作流图，不执行")
    args = parser.parse_args()

    if args.export_graph:
        graph = build_graph(with_checkpointer=False)
        out = export_mermaid(graph, "outputs/workflow.mmd")
        print(f"已导出工作流图: {out}")
        return

    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("[错误] 请在 .env 或环境变量中设置 DASHSCOPE_API_KEY（Charles 节点需要）")
        sys.exit(1)

    question = args.question or f"请用国泰君安五步法分析 {args.stock} 当前的投资机会和催化剂。"
    run_workflow(args.stock, args.capital, args.auto_approve, question)


if __name__ == "__main__":
    main()
