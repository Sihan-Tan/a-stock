#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skill Creator Agent -- 基于 nanobot 的业务技能创建工具

帮助业务人员通过交互式引导，将个人经验沉淀为 Agent Skill。

运行方式:
    python agent.py                          # 交互模式
    python agent.py -m "我想创建一个技能"    # 单次模式
"""

import asyncio
import os
import sys
from pathlib import Path

if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

WORKSPACE = Path(__file__).resolve().parent
NANOBOT_ROOT = WORKSPACE.parent / "nanobot-main"
if str(NANOBOT_ROOT) not in sys.path:
    sys.path.insert(0, str(NANOBOT_ROOT))

from nanobot.agent.hook import AgentHook, AgentHookContext
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.config.loader import load_config
from nanobot.nanobot import Nanobot, _make_provider


class SkillCreatorHook(AgentHook):
    """显示工具调用过程"""

    async def before_execute_tools(self, ctx: AgentHookContext) -> None:
        for tc in ctx.tool_calls:
            if tc.name == "write_file":
                path = tc.arguments.get("file_path", "")
                print(f"  [write_file] {path}")
            elif tc.name == "read_file":
                path = tc.arguments.get("file_path", "")
                print(f"  [read_file] {path}")
            else:
                args_str = str(tc.arguments)[:100]
                print(f"  [{tc.name}] {args_str}")


def build_bot() -> Nanobot:
    """构建 Skill Creator Agent"""
    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not dashscope_key:
        print("[错误] 请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)

    config = load_config(WORKSPACE / "config.json")
    config.providers.dashscope.api_key = dashscope_key
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


async def run_interactive():
    """交互式运行"""
    bot = build_bot()
    loop = bot._loop

    skills = loop.context.skills.list_skills()
    skill_names = " | ".join(s["name"] for s in skills)
    print("=" * 60)
    print("  Skill Creator (业务技能创建工具)")
    print(f"  模型: {loop.model}")
    print(f"  技能: {skill_names}")
    print("  输入 '创建技能' 开始，输入 'quit' 退出")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("再见!")
                break

            print("  [思考中...]")
            result = await bot.run(
                user_input,
                session_key="skill-creator:interactive",
                hooks=[SkillCreatorHook()],
            )
            print(f"\n助手: {result.content}")

        except KeyboardInterrupt:
            print("\n已退出")
            break
        except Exception as e:
            print(f"\n[错误] {type(e).__name__}: {e}")


async def run_single(message: str):
    """单次运行"""
    bot = build_bot()
    print(f"输入: {message}")
    print("  [思考中...]")
    result = await bot.run(
        message,
        session_key="skill-creator:oneshot",
        hooks=[SkillCreatorHook()],
    )
    print(f"\n助手: {result.content}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Skill Creator - 业务技能创建工具")
    parser.add_argument("-m", "--message", default=None, help="单次输入（不传则进入交互模式）")
    args = parser.parse_args()

    if args.message:
        asyncio.run(run_single(args.message))
    else:
        asyncio.run(run_interactive())


if __name__ == "__main__":
    main()
