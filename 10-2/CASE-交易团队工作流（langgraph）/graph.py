# -*- coding: utf-8 -*-
"""
团队工作流图 -- StateGraph 编排

工作流形状（mermaid 风格）：

    START
      v
    charles  (投研)
      v
    zoe      (信号)  <-----+
      v                    |
    kris     (风控) -------+ (reject 且未达重试上限 -> 回到 zoe 缩量重发)
      v
   (kris approve?) -- no --> END
      v yes
    human    (人在回路 interrupt)
      v
   (approved?) -- no --> END
      v yes
    trader   (下单)
      v
    END

Conditional edge 是这张图最值得讲的点：
  - kris -> zoe / human / END  根据 risk_verdict + retry_count
  - human -> trader / END       根据 approved
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes.charles_node import charles_node
from nodes.human_node import human_review_node
from nodes.kris_node import kris_node
from nodes.trader_node import trader_node
from nodes.zoe_node import zoe_node
from state import TradingState


# ---- 条件边：风控之后往哪走 ----
def route_after_kris(state: dict) -> str:
    """
    Kris 否决 -> 重试上限内回到 Zoe，否则结束
    Kris 通过 -> 走人在回路
    """
    verdict = state.get("risk_verdict", {})
    retry = int(state.get("retry_count", 0))
    max_retry = int(state.get("max_retry", 2))
    decision = verdict.get("decision", "approve")

    if decision in ("halt", "reject"):
        if retry >= max_retry:
            print(f"[Graph] Kris 连续否决达上限 ({retry}/{max_retry})，终止流程")
            return END
        print(f"[Graph] Kris 否决，重试 {retry + 1}/{max_retry} -> 回到 Zoe")
        return "zoe_retry"
    return "human"


# ---- 条件边：人在回路之后往哪走 ----
def route_after_human(state: dict) -> str:
    """人类授权 -> 下单；否则结束"""
    return "trader" if state.get("approved") else END


# ---- 重试中转节点：累加 retry_count（不能放在 route 里，路由函数不能改 state）----
def zoe_retry_bump(state: dict) -> dict:
    return {"retry_count": int(state.get("retry_count", 0)) + 1}


def build_graph(with_checkpointer: bool = True):
    """
    构建并编译团队工作流图

    参数：
        with_checkpointer: 是否启用内存检查点
            -- 启用后才能用 interrupt + Command(resume=...) 做人在回路
            -- 启用后才能时间旅行、状态回放
    """
    g = StateGraph(TradingState)

    g.add_node("charles", charles_node)
    g.add_node("zoe", zoe_node)
    g.add_node("zoe_retry", zoe_retry_bump)
    g.add_node("kris", kris_node)
    g.add_node("human", human_review_node)
    g.add_node("trader", trader_node)

    g.add_edge(START, "charles")
    g.add_edge("charles", "zoe")
    g.add_edge("zoe", "kris")

    g.add_conditional_edges(
        "kris",
        route_after_kris,
        {"zoe_retry": "zoe_retry", "human": "human", END: END},
    )
    g.add_edge("zoe_retry", "zoe")  # 重试节点回到 zoe

    g.add_conditional_edges(
        "human",
        route_after_human,
        {"trader": "trader", END: END},
    )
    g.add_edge("trader", END)

    checkpointer = MemorySaver() if with_checkpointer else None
    return g.compile(checkpointer=checkpointer)


def export_mermaid(graph, output_path: str = "outputs/workflow.mmd") -> str:
    """导出 mermaid 工作流图，方便贴到课件 / GitHub"""
    from pathlib import Path
    text = graph.get_graph().draw_mermaid()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return str(out)
