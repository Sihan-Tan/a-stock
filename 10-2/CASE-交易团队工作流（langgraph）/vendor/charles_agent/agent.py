#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Charles 投研 Agent - 基于 DeepAgents + @tool 架构

参考 DeepAgents 官方 content-builder-agent 模式:
  - 核心能力定义为 @tool 函数（模型可直接调用）
  - 工作流指导写在 system_prompt 中
  - 不依赖 SkillsMiddleware（避免 Qwen 把技能名误当工具名）

运行方式：
    python agent.py
    python agent.py --model qwen-plus
"""

# ---- Windows UTF-8 兼容 ----
import locale
import os
import subprocess
import sys

if sys.platform == "win32":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(locale, "getencoding"):
        locale.getencoding = lambda: "utf-8"  # type: ignore[attr-defined]
    _orig_getpreferredencoding = locale.getpreferredencoding
    locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"

import argparse
from datetime import datetime
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import LocalShellBackend
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
#  脚本执行器（所有 @tool 共用）
# ---------------------------------------------------------------------------
def _run_script(cmd_args: list[str], timeout: int = 120) -> str:
    """执行 Python 脚本并返回 stdout，处理 Windows 编码"""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    try:
        result = subprocess.run(
            cmd_args,
            capture_output=True,
            cwd=str(PROJECT_ROOT),
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        output = result.stdout.strip()
        if result.returncode != 0 and result.stderr:
            output += "\n[stderr] " + result.stderr.strip()
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: script timed out"
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
#  @tool 定义 -- 参考官方 content-builder-agent 的 web_search 模式
# ---------------------------------------------------------------------------
@tool
def web_search(query: str, type: str = "general") -> str:
    """搜索网络获取最新市场新闻、政策、公司公告和分析报告。

    Args:
        query: 搜索关键词，如 "贵州茅台 2025年年报 业绩"
        type: 搜索类型 - general(通用)/news(新闻)/stock(个股)/policy(政策)
    """
    return _run_script([
        sys.executable, "skills/web-search/scripts/search_market.py",
        "--query", query, "--type", type,
    ])


@tool
def query_pdf(query: str, stock: str = "") -> str:
    """从本地 PDF 研报/财报知识库检索信息（RAG），支持页码溯源。

    Args:
        query: 查询问题，如 "2024年营收和净利润数据"
        stock: 按股票代码过滤（如 600519），留空搜索全部
    """
    cmd = [
        sys.executable, "skills/read-pdf/scripts/query_report.py",
        "--index_dir", "data/vector_store",
        "--query", query,
    ]
    if stock:
        cmd.extend(["--stock", stock])
    return _run_script(cmd)


@tool
def stock_price(code: str, period: str = "1d", count: int = 20) -> str:
    """获取 A 股实时 K 线数据（通过 MiniQMT）。

    Args:
        code: 股票代码，如 "600519.SH"、"000001.SZ"
        period: K线周期 - 1d(日线)/1w(周线)/1m(1分钟)/5m/15m/30m/1h
        count: 获取条数（默认20）
    """
    return _run_script([
        sys.executable, "skills/stock-price/scripts/get_kline.py",
        code, period, str(count),
    ])


@tool
def financial_analysis(stock: str, years: int = 5) -> str:
    """分析上市公司核心财务指标趋势（毛利率/ROE/负债率等），支持同行业横向对比。

    Args:
        stock: 股票代码，如 "600519"
        years: 分析年数（默认5年）
    """
    return _run_script([
        sys.executable, "skills/financial-analysis/scripts/ratio_analysis.py",
        "--stock", stock, "--years", str(years),
    ])


@tool
def compare_reports_period(stock: str, topics: str = "营收,净利润,毛利率,经营情况") -> str:
    """同一公司不同时期（季度/年度）的纵向对比分析。

    Args:
        stock: 股票代码，如 "688981"
        topics: 对比维度（逗号分隔），如 "营收,净利润,毛利率"
    """
    return _run_script([
        sys.executable, "skills/compare-reports/scripts/cross_period.py",
        "--stock", stock, "--topics", topics,
    ])


@tool
def compare_reports_company(stocks: str, topic: str = "经营状况和盈利能力") -> str:
    """不同公司之间的横向对比分析。

    Args:
        stocks: 股票代码列表（逗号分隔），如 "688981,600519"
        topic: 对比主题
    """
    return _run_script([
        sys.executable, "skills/compare-reports/scripts/cross_company.py",
        "--stocks", stocks, "--topic", topic,
    ])


# ---------------------------------------------------------------------------
#  工具列表
# ---------------------------------------------------------------------------
TOOLS = [web_search, query_pdf, stock_price, financial_analysis,
         compare_reports_period, compare_reports_company]


def _get_project_root() -> Path:
    return PROJECT_ROOT


def _discover_skills(root: Path) -> list[dict]:
    """扫描 skills/ 目录，解析 SKILL.md 前置元数据（仅用于启动时展示）"""
    skills_dir = root / "skills"
    found = []
    if not skills_dir.exists():
        return found
    for skill_dir in sorted(skills_dir.iterdir()):
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        try:
            text = skill_md.read_text(encoding="utf-8")
            name, desc = skill_dir.name, ""
            if text.startswith("---"):
                _, fm, _ = text.split("---", 2)
                for line in fm.strip().splitlines():
                    if line.startswith("name:"):
                        name = line.split(":", 1)[1].strip().strip('"')
                    elif line.startswith("description:"):
                        desc = line.split(":", 1)[1].strip().strip('"')
            found.append({"name": name, "description": desc})
        except Exception:
            found.append({"name": skill_dir.name, "description": ""})
    return found


def create_charles_agent(model: str = None, checkpointer=None):
    root = _get_project_root()

    backend = LocalShellBackend(
        root_dir=str(root),
        virtual_mode=True,
        inherit_env=True,
        timeout=300,
    )

    today_str = datetime.now().strftime("%Y年%m月%d日")
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    weekday_str = weekday_names[datetime.now().weekday()]

    system_prompt = f"""你是 Charles，一位专业的 AI 投研情报官。你为投资者提供深度研究分析服务。

=== 重要: 当前时间 ===
今天是 {today_str} {weekday_str}。
你必须以此日期为基准来理解时间:
- 2024年及之前的数据属于历史数据
- 2025年的数据属于近期已发生的数据（不是未来）
- 2026年截至今天之前的数据也是已发生的
- 在撰写研报时，请确保时间表述准确，不要把已经过去的时间当作未来

可用工具:
- web_search: 联网搜索最新市场信息、新闻、公告
- query_pdf: 从本地 PDF 研报/财报知识库检索（RAG）
- stock_price: 获取 A 股实时 K 线数据
- financial_analysis: 分析财务指标趋势（ROE/毛利率/负债率等）
- compare_reports_period: 同一公司跨期纵向对比
- compare_reports_company: 不同公司横向对比

数据说明:
- 本地已有索引: 中芯国际(688981) 11份研报/财报, 贵州茅台(600519) 1份三季报, 共 860 chunks
- 常用股票代码: 中芯国际 688981, 贵州茅台 600519, 五粮液 000858, 比亚迪 002594, 宁德时代 300750
- 投资建议需附带风险提示

=== 研报撰写策略（核心工作模式） ===

当用户要求写研报、深度分析、五步法分析时，你应该自己做研究和分析。
核心方法论: 国泰君安"五步法"（信息差 -> 逻辑差 -> 预期差 -> 催化剂 -> 结论+风险闭环）。

--- 第一阶段: 规划 ---
先思考再行动:
1. 识别分析对象（个股/行业/事件）
2. 判断属于哪种研报场景: 个股深度 / 季报速评 / 行业比较 / 事件驱动 / 财务异常
3. 列出五步法每一步需要搜集什么信息
4. 规划搜索序列（先搜什么、再搜什么）

--- 第二阶段: 迭代式信息收集 ---
以 web_search 为主要信息来源，通过多轮搜索逐步积累分析素材:
- 第一轮: 搜索公司/行业的基本面概况
- 分析结果: 从搜索结果中发现新线索、新问题
- 第二轮: 针对发现的线索追加搜索（这是关键 -- 不要一次搜完就停）
- 继续迭代: 直到五步法每一步都有足够的数据支撑

辅助信息来源（按需使用）:
- query_pdf: 本地 RAG，精确的财报附注数据（仅当有该股票本地数据时）
- financial_analysis: 结构化的 ROE/毛利率/负债率等趋势
- stock_price: 实时行情和 K 线走势
- compare_reports_period / compare_reports_company: 跨期或跨公司对比

--- 第三阶段: 五步法分析与输出 ---
收集够信息后，按五步法框架直接在对话中输出 Markdown 格式研报。

五步法思考链（每步必须回答核心问题）:

Step 1 信息差 -- 市场还不知道/忽视了什么？
  重点: 财报附注中的隐藏数据、非经常性损益、新业务增长信号、现金流与利润的背离
  输出: 3-5个被市场忽视的关键数据点，附具体数字

Step 2 逻辑差 -- 市场的推理错在哪里？
  重点: 识别市场的线性思维误区，构建正确的因果逻辑链
  输出: 市场误读 vs 正确逻辑的对比

Step 3 预期差 -- 一致预期 vs 实际偏离多大？
  重点: 量化偏离幅度，判断是一次性还是可持续的
  输出: 预期差对比表（指标/一致预期/我的预测/偏离幅度）

Step 4 催化剂 -- 什么事件会引爆重估？
  重点: 短期(1-3月)、中期(3-12月)催化剂时间轴 + 潜在风险催化
  输出: 按时间排序的催化剂清单

Step 5 结论+风险闭环 -- 最终判断 + 哪里可能出错？
  重点: 明确投资评级，关键假设
  风险闭环（国泰君安特别强调）: 必须指出"哪个假设出错会导致整个结论崩塌"
  输出: 核心观点 + 投资逻辑 + 失效条件

=== 五种研报场景 ===

场景1 - 个股深度: web_search 公司基本面 -> 行业格局 -> 竞品对比 -> 券商评级 -> 近期事件
场景2 - 季报速评: query_pdf 本地RAG优先(如有) + web_search 市场预期对比
场景3 - 行业比较: web_search 各公司最新业绩 -> 估值对比 -> compare_reports_company
场景4 - 事件驱动: web_search 政策/新闻 为主 -> 受益公司 -> 预期差分析
场景5 - 财务异常: financial_analysis 定量 + query_pdf 附注深挖"""

    llm = ChatTongyi(
        model=model or os.environ.get("CHARLES_MODEL", "qwen-plus"),
    )

    agent = create_deep_agent(
        model=llm,
        system_prompt=system_prompt,
        backend=backend,
        tools=TOOLS,
        checkpointer=checkpointer or InMemorySaver(),
    )

    return agent


def main():
    parser = argparse.ArgumentParser(description="Charles 投研情报官")
    parser.add_argument("--model", default=None, help="LLM 模型（默认 qwen-plus）")
    parser.add_argument(
        "--stream", action="store_true", default=True,
        help="流式输出中间步骤（默认开启）",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[错误] 请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)

    model_name = args.model or os.environ.get("CHARLES_MODEL", "qwen-plus")
    root = _get_project_root()
    skills_info = _discover_skills(root)
    agent = create_charles_agent(model=model_name)

    skill_names = " | ".join(s["name"] for s in skills_info)
    print("=" * 60)
    print("  Charles 投研情报官已就绪 (DeepAgents + Skills)")
    print(f"  模型: {model_name}")
    print(f"  技能: {skill_names}")
    print("  输入问题后按回车，输入 'quit' 退出")
    print("=" * 60)

    thread_id = "default"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit", "q"):
                print("Charles: 再见，祝投资顺利!")
                break

            print("  [思考中...]")

            if args.stream:
                _run_with_stream(agent, user_input, config)
            else:
                _run_with_invoke(agent, user_input, config)

        except KeyboardInterrupt:
            print("\n已退出")
            break
        except Exception as e:
            print(f"\n[错误] {type(e).__name__}: {e}")


def _run_with_stream(agent, user_input: str, config: dict):
    """流式运行，实时展示中间步骤"""
    import requests as _requests

    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            final_reply = ""
            for msg, metadata in agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                stream_mode="messages",
            ):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        name = tc.get("name", "")
                        raw_args = tc.get("args", {})
                        if name == "execute":
                            display = raw_args.get("command", "")
                        elif name == "read_file":
                            display = raw_args.get("path", "") or raw_args.get("file_path", "")
                        elif name in ("ls", "list_directory"):
                            display = raw_args.get("path", "/")
                        else:
                            display = str(raw_args)[:120]
                        print(f"  [{name}] {display}")

                elif getattr(msg, "type", "") == "tool":
                    content = str(getattr(msg, "content", ""))
                    if len(content) > 300:
                        content = content[:300] + "..."
                    print(f"  [结果] {content}")

                elif getattr(msg, "type", "") == "ai" and msg.content:
                    if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                        final_reply = msg.content

            if final_reply:
                print(f"\nCharles: {final_reply}")
            else:
                print("\nCharles: [无文本回复，请查看上方工具调用结果]")
            return

        except (_requests.exceptions.ConnectionError, ConnectionError, OSError) as e:
            if attempt < max_retries:
                print(f"  [网络波动，第 {attempt+1} 次重试...]")
                import time
                time.sleep(2)
            else:
                print(f"\n[网络错误] 多次重试后仍然失败: {e}")
        except Exception as e:
            print(f"\n[错误] {type(e).__name__}: {e}")
            return


def _run_with_invoke(agent, user_input: str, config: dict):
    """非流式运行"""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )
    messages = result.get("messages", [])
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, "content") and last_msg.content:
            print(f"\nCharles: {last_msg.content}")
        else:
            print("\nCharles: [无文本回复，请查看工具调用结果]")
    else:
        print("\nCharles: [无回复]")


if __name__ == "__main__":
    main()
