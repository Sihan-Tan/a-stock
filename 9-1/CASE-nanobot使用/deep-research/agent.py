#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deep Research Agent -- nanobot 版

对标 DeepAgents 官方 deep_research 示例。
使用 nanobot 的 web_search + 自定义 think_tool + 文件工具。

运行: python agent.py "List 2 trends in AI agents in one paragraph"
"""

import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = Path(__file__).resolve().parent
NANOBOT_ROOT = WORKSPACE.parent.parent / "nanobot-main"
if str(NANOBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(NANOBOT_ROOT))

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.loop import AgentLoop
from nanobot.agent.tools.base import Tool
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import load_config
from nanobot.nanobot import Nanobot, _make_provider


class ThinkTool(Tool):
    """反思工具 -- 让 Agent 在搜索后结构化思考"""

    @property
    def name(self) -> str:
        return "think_tool"

    @property
    def description(self) -> str:
        return (
            "Tool for strategic reflection on research progress. "
            "Use after each search to: "
            "1) Analyze current findings, "
            "2) Assess gaps in knowledge, "
            "3) Evaluate quality of sources, "
            "4) Decide whether to continue searching or synthesize an answer."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "reflection": {
                    "type": "string",
                    "description": "Your structured reflection on the research so far"
                }
            },
            "required": ["reflection"]
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        reflection = kwargs.get("reflection", "")
        return f"Reflection recorded: {reflection}"


class PrintHook(AgentHook):
    async def before_execute_tools(self, ctx: AgentHookContext) -> None:
        for tc in ctx.tool_calls:
            if tc.name == "think_tool":
                text = str(tc.arguments.get("reflection", ""))[:150]
                print(f"  [think] {text}...")
            elif tc.name == "write_file":
                path = tc.arguments.get("file_path", "")
                print(f"  [write] {path}")
            else:
                print(f"  >> {tc.name}: {str(tc.arguments)[:100]}")


def build_bot() -> Nanobot:
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not dashscope_key:
        print("[Error] DASHSCOPE_API_KEY not set")
        sys.exit(1)

    config = load_config(WORKSPACE / "config.json")
    config.providers.dashscope.api_key = dashscope_key
    config.tools.web.search.api_key = os.environ.get("TAVILY_API_KEY", "")
    config.agents.defaults.workspace = str(WORKSPACE)

    provider = _make_provider(config)
    defaults = config.agents.defaults

    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=WORKSPACE,
        model=defaults.model,
        max_iterations=defaults.max_tool_iterations,
        context_window_tokens=defaults.context_window_tokens,
        max_tool_result_chars=defaults.max_tool_result_chars,
        web_config=config.tools.web,
        exec_config=config.tools.exec,
        restrict_to_workspace=False,
        timezone=defaults.timezone,
    )

    # 注册反思工具
    loop.tools.register(ThinkTool())

    # 注入当前日期到 memory
    memory_file = WORKSPACE / "memory" / "MEMORY.md"
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    memory_file.write_text(f"# Date Context\n\nCurrent date: {today}\n", encoding="utf-8")

    return Nanobot(loop)


async def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "List 2 trends in AI agents in one paragraph"

    print(f"\nDeep Research Agent (nanobot)")
    print(f"Question: {question}\n")

    bot = build_bot()
    result = await bot.run(question, session_key="research:run", hooks=[PrintHook()])

    print(f"\n{'='*60}")
    print(result.content)
    print(f"{'='*60}")

    report = WORKSPACE / "final_report.md"
    if report.exists():
        print(f"\nReport saved to: {report}")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
