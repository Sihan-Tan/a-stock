# -*- coding: utf-8 -*-
"""
charles_node -- 投研情报官节点

复用本案例 vendor/charles_agent/agent.py 中的 create_charles_agent（与 14-16 章同源）。
Charles 内部仍然是 DeepAgents（封装好的 Agent 模式），但调度权由 LangGraph 接管，
这就是大纲里强调的 "Workflow 编排 Agent" 的混合范式。

输入：state["stock_code"], state["user_question"]
输出：state["investment_view"]
"""

import importlib.util
import json
import re
import sys
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from utils.env import CHARLES_AGENT_DIR, PROJECT_ROOT


def _load_charles_module():
    """从 vendor/charles_agent/agent.py 显式加载，避免与同名模块冲突"""
    agent_path = CHARLES_AGENT_DIR / "agent.py"
    # 把 Charles 根目录加到 sys.path 首位，让其内部 skills 等脚本以相对路径可被 subprocess 调用
    sp = str(CHARLES_AGENT_DIR)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    spec = importlib.util.spec_from_file_location("charles_agent_mod", agent_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_charles_mod = _load_charles_module()
create_charles_agent = _charles_mod.create_charles_agent


# 让 Charles 在产出研报后追加一个结构化 JSON 摘要，便于 LangGraph 解析
EXTRACT_SUFFIX = """

=== 强制要求：在你回复的最后必须追加一段结构化 JSON 摘要 ===

格式要求（严格遵守，否则下游团队成员无法读取）：
1. 必须使用 ```json 代码块包裹
2. 必须放在整个回复的最末尾，且 JSON 之后不能再有任何文字
3. stance 取值只能是 bullish / neutral / bearish 之一
4. confidence 是 0~1 之间的浮点数
5. catalysts 和 risks 是字符串数组

模板：
```json
{
  "stance": "bullish",
  "confidence": 0.75,
  "summary": "一句话核心观点（不超过 60 字）",
  "catalysts": ["短期催化剂1", "中期催化剂2"],
  "risks": ["主要风险1", "主要风险2"]
}
```
"""


def _find_json_blocks(text: str) -> list:
    """在文本中找出所有 top-level 平衡的 {...} 块"""
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
    return blocks


def _extract_view(report_text: str) -> dict:
    """从 Charles 输出中提取结构化摘要 JSON（容忍多种格式）"""
    candidates = []

    # 1) 优先尝试 ```json ... ``` 代码块（可能多个，取最后一个）
    fence_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", report_text, re.DOTALL)
    candidates.extend(fence_matches)

    # 2) 兜底：扫描所有 top-level {...} 块（取最后一个，通常摘要在末尾）
    candidates.extend(_find_json_blocks(report_text))

    # 反向尝试，因为摘要通常在文末
    last_err = None
    for payload in reversed(candidates):
        try:
            data = json.loads(payload)
            if not isinstance(data, dict) or "stance" not in data:
                continue
            return {
                "stance": str(data.get("stance", "neutral")).lower(),
                "confidence": float(data.get("confidence", 0.5)),
                "summary": str(data.get("summary", "")).strip(),
                "catalysts": list(data.get("catalysts", [])),
                "risks": list(data.get("risks", [])),
            }
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            last_err = e
            continue

    raise RuntimeError(
        f"无法从 Charles 输出中提取结构化摘要 (last_err={last_err}). "
        f"原文末尾 500 字: {report_text[-500:]}"
    )


def _strip_summary_block(text: str) -> str:
    """把末尾的 ```json ... ``` 摘要块剥掉，只留正文，避免在 HTML 中重复展示"""
    return re.sub(r"```(?:json)?\s*\{.*?\}\s*```\s*$", "", text, flags=re.DOTALL).rstrip()


def _md_to_html(md: str) -> str:
    """轻量 Markdown 渲染：标题、列表、引用、加粗、行内代码、段落"""
    lines = md.splitlines()
    out = []
    in_list = False
    in_pre = False
    pre_buf = []

    def flush_list():
        nonlocal in_list
        if in_list:
            out.append("</ul>")
            in_list = False

    def render_inline(s: str) -> str:
        s = escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"`([^`]+?)`", r"<code class='inline-code'>\1</code>", s)
        return s

    for line in lines:
        if line.strip().startswith("```"):
            if in_pre:
                out.append("<pre>" + escape("\n".join(pre_buf)) + "</pre>")
                pre_buf = []
                in_pre = False
            else:
                flush_list()
                in_pre = True
            continue
        if in_pre:
            pre_buf.append(line)
            continue

        stripped = line.rstrip()
        if not stripped:
            flush_list()
            continue

        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m:
            flush_list()
            level = min(len(m.group(1)) + 1, 6)  # h2~h6（h1 留给标题）
            out.append(f"<h{level}>{render_inline(m.group(2))}</h{level}>")
            continue

        if stripped.startswith(("- ", "* ", "+ ")):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{render_inline(stripped[2:])}</li>")
            continue

        m2 = re.match(r"^\d+\.\s+(.+)$", stripped)
        if m2:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{render_inline(m2.group(1))}</li>")
            continue

        if stripped.startswith("> "):
            flush_list()
            out.append(f"<blockquote>{render_inline(stripped[2:])}</blockquote>")
            continue

        flush_list()
        out.append(f"<p>{render_inline(stripped)}</p>")

    flush_list()
    if in_pre and pre_buf:
        out.append("<pre>" + escape("\n".join(pre_buf)) + "</pre>")
    return "\n".join(out)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, "Microsoft YaHei", "Segoe UI", sans-serif;
         max-width: 900px; margin: 30px auto; padding: 0 24px;
         color: #2c3e50; line-height: 1.8; background: #f8f9fa; }}
  .hero {{ background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 50%, #415a77 100%);
           color: #fff; padding: 36px 32px; border-radius: 12px; margin-bottom: 24px; }}
  .hero h1 {{ font-size: 1.6em; margin-bottom: 8px; }}
  .hero .meta {{ font-size: 0.9em; opacity: 0.85; }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 20px;
            font-size: 0.85em; font-weight: 600; margin-right: 8px; }}
  .badge-bullish {{ background: #d4edda; color: #155724; }}
  .badge-neutral {{ background: #fff3cd; color: #856404; }}
  .badge-bearish {{ background: #f8d7da; color: #721c24; }}
  .summary {{ background: #fff; padding: 20px 24px; border-radius: 10px;
              border-left: 4px solid #3498db; margin-bottom: 24px;
              box-shadow: 0 2px 8px rgba(0,0,0,0.04); }}
  .summary h3 {{ color: #3498db; margin-bottom: 10px; font-size: 1.1em; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 16px 0; }}
  .grid-card {{ background: #fff; padding: 16px 20px; border-radius: 8px; }}
  .grid-card.cat {{ border-left: 4px solid #27ae60; }}
  .grid-card.risk {{ border-left: 4px solid #e74c3c; }}
  .grid-card h4 {{ font-size: 1em; margin-bottom: 8px; }}
  .grid-card.cat h4 {{ color: #27ae60; }}
  .grid-card.risk h4 {{ color: #e74c3c; }}
  .body {{ background: #fff; padding: 24px 28px; border-radius: 10px;
           box-shadow: 0 2px 8px rgba(0,0,0,0.04); }}
  .body h2, .body h3, .body h4 {{ color: #2c3e50; margin: 18px 0 8px; }}
  .body p {{ margin-bottom: 10px; }}
  .body ul {{ padding-left: 22px; margin-bottom: 12px; }}
  .body blockquote {{ border-left: 3px solid #95a5a6; padding-left: 12px;
                      color: #555; background: #f1f3f5; margin: 10px 0;
                      padding-top: 8px; padding-bottom: 8px; }}
  .body pre {{ background: #1e1e2e; color: #cdd6f4; padding: 14px 18px;
               border-radius: 6px; overflow-x: auto; font-size: 0.88em; }}
  .inline-code {{ background: #e8ecef; padding: 2px 6px; border-radius: 4px;
                  font-size: 0.88em; font-family: "Cascadia Code","Consolas",monospace; }}
  .footer {{ text-align: center; color: #999; font-size: 0.85em;
             margin: 28px 0 12px; }}
</style>
</head>
<body>
<div class="hero">
  <h1>{title}</h1>
  <div class="meta">
    <span class="badge badge-{stance}">{stance_zh}</span>
    信心 {confidence:.2f} · 生成时间 {generated_at} · 来源 Charles 投研情报官
  </div>
</div>

<div class="summary">
  <h3>核心观点</h3>
  <p>{summary}</p>
</div>

<div class="grid">
  <div class="grid-card cat">
    <h4>催化剂</h4>
    <ul>{catalysts_html}</ul>
  </div>
  <div class="grid-card risk">
    <h4>主要风险</h4>
    <ul>{risks_html}</ul>
  </div>
</div>

<div class="body">
{body_html}
</div>

<div class="footer">
  本报告由 AI 投研情报官 Charles 生成，仅供学习和研究参考，不构成投资建议。
</div>
</body>
</html>"""


def _build_report_html(stock: str, view: dict, body_md: str) -> str:
    """组装研报 HTML"""
    stance = view.get("stance", "neutral")
    stance_zh_map = {"bullish": "看多", "neutral": "中性", "bearish": "看空"}
    catalysts = view.get("catalysts", []) or []
    risks = view.get("risks", []) or []
    return HTML_TEMPLATE.format(
        title=f"{stock} 投研简报",
        stance=stance,
        stance_zh=stance_zh_map.get(stance, stance),
        confidence=float(view.get("confidence", 0)),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        summary=escape(view.get("summary", "")),
        catalysts_html="\n".join(f"<li>{escape(str(c))}</li>" for c in catalysts) or "<li>(无)</li>",
        risks_html="\n".join(f"<li>{escape(str(r))}</li>" for r in risks) or "<li>(无)</li>",
        body_html=_md_to_html(body_md),
    )


def _save_report_attachments(stock: str, view: dict, raw_report: str) -> tuple:
    """把研报落盘成 .md 和 .html 两个附件，返回路径对"""
    out_dir = PROJECT_ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{stock.replace('.', '_')}_{ts}"

    body_md = _strip_summary_block(raw_report)
    md_path = out_dir / f"{stem}.md"
    md_path.write_text(body_md, encoding="utf-8")

    html_path = out_dir / f"{stem}.html"
    html_path.write_text(_build_report_html(stock, view, body_md), encoding="utf-8")
    return str(md_path), str(html_path)


def _extract_final_text(result: Any) -> str:
    """从 deepagents.invoke 返回中拿到最后一条 AI 文本"""
    messages = result.get("messages", []) if isinstance(result, dict) else []
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        msg_type = getattr(msg, "type", "")
        if msg_type == "ai" and content:
            return content
    return ""


def charles_node(state: dict) -> dict:
    """投研节点：调用 Charles，输出 investment_view"""
    stock = state["stock_code"]
    question = state.get("user_question") or f"请分析 {stock} 当前的投资机会。"
    full_question = question + EXTRACT_SUFFIX

    print()
    print("=" * 70)
    print(f"[Charles] 投研情报官开始工作 -- 标的: {stock}")
    print("=" * 70)

    agent = create_charles_agent()
    config = {
        "configurable": {"thread_id": f"trading-team-{stock}"},
        "recursion_limit": 80,
    }

    result = agent.invoke(
        {"messages": [{"role": "user", "content": full_question}]},
        config=config,
    )
    report_text = _extract_final_text(result)
    view = _extract_view(report_text)
    view["raw_report"] = report_text

    md_path, html_path = _save_report_attachments(stock, view, report_text)
    view["report_md_path"] = md_path
    view["report_html_path"] = html_path

    print(f"[Charles] 立场: {view['stance']} | 信心: {view['confidence']:.2f}")
    print(f"[Charles] 核心: {view['summary']}")
    print(f"[Charles] 催化剂 {len(view['catalysts'])} 条 | 风险 {len(view['risks'])} 条")
    print(f"[Charles] 研报附件: {Path(html_path).name} (HTML), {Path(md_path).name} (MD)")

    return {
        "investment_view": view,
        "messages": [
            {
                "role": "charles",
                "time": datetime.now().strftime("%H:%M:%S"),
                "content": f"{view['stance']} ({view['confidence']:.2f}) -- {view['summary']}",
            }
        ],
    }
