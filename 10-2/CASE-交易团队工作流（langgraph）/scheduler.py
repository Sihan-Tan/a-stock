# -*- coding: utf-8 -*-
"""
APScheduler 定时调度器 -- 盘前 9:25 自动跑一次团队工作流

工作流模式相对 Agent 模式的另一个杀手特性：
  Agent 模式靠用户对话驱动；工作流模式可以挂在 cron / scheduler 上自动触发，
  在用户睡觉、开会、出差时也能稳定执行（这正是机构量化的标准姿势）。

用法：
    python scheduler.py                          # 默认每个交易日 9:25 跑一次 600519.SH
    python scheduler.py --stock 002594.SZ        # 指定标的
    python scheduler.py --cron "30 8 * * 1-5"    # 自定义 cron 表达式
    python scheduler.py --once                   # 立刻跑一次然后退出（用于手动测试）

输出：
    outputs/runs/<stock>_<YYYY-MM-DD>.json   每次运行的最终 state
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.env import setup_utf8

setup_utf8()

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from apscheduler.schedulers.blocking import BlockingScheduler  # noqa: E402
from apscheduler.triggers.cron import CronTrigger  # noqa: E402

from main import run_workflow  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("trading-scheduler")

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _save_run(stock: str, state: dict):
    """把最终 state 序列化落盘，便于事后审计 / 复盘"""
    today = datetime.now().strftime("%Y-%m-%d_%H%M")
    fname = f"{stock.replace('.', '_')}_{today}.json"
    out_file = OUTPUT_DIR / fname

    serializable = {
        "stock_code": state.get("stock_code"),
        "capital": state.get("capital"),
        "investment_view": {k: v for k, v in (state.get("investment_view") or {}).items()
                            if k != "raw_report"},  # 正文已落盘 .md/.html，state 里不重复
        "trade_signal": state.get("trade_signal"),
        "risk_verdict": state.get("risk_verdict"),
        "approved": state.get("approved"),
        "trade_result": state.get("trade_result"),
        "messages": state.get("messages", []),
    }
    out_file.write_text(json.dumps(serializable, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    log.info("运行结果已落盘: %s", out_file)


def scheduled_job(stock: str, capital: float, auto_approve: bool):
    """定时任务：跑一次完整工作流"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.info("=" * 70)
    log.info("[定时触发] %s -- 标的 %s | 资金 %s | auto_approve=%s",
             now_str, stock, f"{capital:,.0f}", auto_approve)
    log.info("=" * 70)

    question = f"请用国泰君安五步法分析 {stock} 当前的投资机会和催化剂。"

    try:
        state = run_workflow(stock, capital, auto_approve, question)
        _save_run(stock, state)
        log.info("[定时任务完成] %s", stock)
    except Exception as e:
        log.exception("[定时任务异常] %s: %s", type(e).__name__, e)


def main():
    parser = argparse.ArgumentParser(description="团队工作流 APScheduler 调度器")
    parser.add_argument("--stock", default="600519.SH", help="股票代码")
    parser.add_argument("--capital", type=float, default=1_000_000, help="可用资金")
    parser.add_argument("--cron", default="25 9 * * 1-5",
                        help="cron 表达式，默认 '25 9 * * 1-5' = 工作日 9:25")
    parser.add_argument("--once", action="store_true",
                        help="立即执行一次然后退出（用于手动测试，不挂调度器）")
    parser.add_argument("--auto-approve", action="store_true", default=True,
                        help="定时任务默认自动放行人审")
    args = parser.parse_args()

    if not os.environ.get("DASHSCOPE_API_KEY"):
        log.error("请设置 DASHSCOPE_API_KEY")
        sys.exit(1)

    if args.once:
        log.info("[--once] 立即执行一次")
        scheduled_job(args.stock, args.capital, args.auto_approve)
        return

    scheduler = BlockingScheduler(timezone="Asia/Shanghai")
    trigger = CronTrigger.from_crontab(args.cron, timezone="Asia/Shanghai")

    scheduler.add_job(
        scheduled_job,
        trigger=trigger,
        kwargs={"stock": args.stock, "capital": args.capital,
                "auto_approve": args.auto_approve},
        id=f"trading-team-{args.stock}",
        name=f"团队工作流 {args.stock}",
        max_instances=1,
        coalesce=True,
    )

    log.info("=" * 70)
    log.info("调度器已启动")
    log.info("  cron: %s (Asia/Shanghai)", args.cron)
    log.info("  标的: %s", args.stock)
    log.info("  资金: %s", f"{args.capital:,.0f}")
    log.info("  按 Ctrl+C 退出（scheduler.start 后阻塞，下次触发会自动打印）")
    log.info("=" * 70)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("调度器已退出")


if __name__ == "__main__":
    main()
