#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Content Builder Agent -- nanobot 版

对标 DeepAgents 官方 content-builder-agent 示例。
利用 nanobot 的 Skills 渐进式加载 + 内置 web_search + 文件工具。

运行: python agent.py "Write a short blog post about AI agents"
"""

import asyncio
import os
import sys
from pathlib import Path

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
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import load_config
from nanobot.nanobot import Nanobot, _make_provider


class PrintHook(AgentHook):
    async def before_execute_tools(self, ctx: AgentHookContext) -> None:
        for tc in ctx.tool_calls:
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
    return Nanobot(loop)


async def main():
    task = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Write a short blog post (under 200 words) about AI agents"

    print(f"\nContent Builder Agent (nanobot)")
    print(f"Task: {task}\n")

    bot = build_bot()
    result = await bot.run(task, session_key="content:run", hooks=[PrintHook()])

    print(f"\n{'='*60}")
    print(result.content)
    print(f"{'='*60}")
    print("\n[OK] Done!")


if __name__ == "__main__":
    asyncio.run(main())
