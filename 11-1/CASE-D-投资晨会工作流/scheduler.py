# -*- coding: utf-8 -*-
# 21-CASE-D: 投资晨会调度器
"""
MorningScheduler -- 简化版定时调度器, 让晨会工作流每天 9:00 自动跑

用 APScheduler 的 BlockingScheduler + cron 触发器, 跨平台:
    - Windows: 直接 python scheduler.py 跑前台
    - Linux:   nohup python scheduler.py > sched.log 2>&1 &

两个 cron job:
    08:30 (周一到周五)  job_data_refresh   -> CASE-A run_daily.py 增量更新昨日数据
    09:00 (周一到周五)  job_pre_market     -> CASE-D graph.py 跑晨会工作流并推送

用法:
    python scheduler.py               # 启动调度器, 阻塞运行 (Ctrl+C 退出)
    python scheduler.py --simulate    # 模拟模式: 立刻把两个 job 各跑一次, 验证流程通
    python scheduler.py --job pre     # 只注册 9:00 那个 job (测试用)
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

# 项目根目录 = 当前 CASE-D 的上级
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("morning-scheduler")


# ============================================================
# 业务任务定义
# ============================================================

def job_data_refresh():
    """8:30 触发: 调 CASE-A 的 run_daily.py 做昨日数据增量"""
    log.info("[JOB] job_data_refresh 触发")
    run_daily = PROJECT_ROOT / "CASE-A-板块数据准备" / "run_daily.py"
    if not run_daily.exists():
        log.error(f"找不到 {run_daily}")
        return
    ret = subprocess.run([sys.executable, str(run_daily)], cwd=str(run_daily.parent))
    log.info(f"[JOB] job_data_refresh 完成 (returncode={ret.returncode})")


def job_pre_market():
    """9:00 触发: 调 CASE-D 的 graph.py 跑晨会工作流"""
    log.info("[JOB] job_pre_market 触发")
    # 直接 import build_graph, 比 subprocess 启动 Python 解释器快得多
    sys.path.insert(0, str(THIS_DIR))
    from graph import build_graph
    graph = build_graph()
    result = graph.invoke({
        "trigger_time":     datetime.now().isoformat(timespec="seconds"),
        "industry_level":   2,
        "top_n_industries": 5,
        "top_n_stocks":     10,
        "lookback_days":    90,
        "sample_stocks":    20,
        "messages":         [],
    })
    log.info(f"[JOB] job_pre_market 完成, 推送结果: {result.get('push_result')}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="投资晨会调度器 (教学简化版)")
    parser.add_argument("--simulate", action="store_true",
                        help="模拟模式: 立刻把每个 job 跑一次然后退出")
    parser.add_argument("--job", choices=["data", "pre", "all"], default="all",
                        help="只注册某一个 job (data=8:30 / pre=9:00 / all=两个都注册)")
    args = parser.parse_args()

    if args.simulate:
        log.info("=" * 60)
        log.info("[SIMULATE] 模拟模式: 立刻跑每个 job 一次")
        log.info("=" * 60)
        if args.job in ("data", "all"):
            job_data_refresh()
        if args.job in ("pre", "all"):
            job_pre_market()
        return

    sched = BlockingScheduler(timezone="Asia/Shanghai")

    if args.job in ("data", "all"):
        sched.add_job(
            job_data_refresh, id="data_refresh", name="CASE-A 数据增量",
            trigger=CronTrigger(hour=8, minute=30, day_of_week="mon-fri",
                                timezone="Asia/Shanghai"),
        )
        log.info("[REG] 08:30 (周一到周五)  job_data_refresh   -> CASE-A run_daily.py")

    if args.job in ("pre", "all"):
        sched.add_job(
            job_pre_market, id="pre_market", name="9:00 晨会工作流",
            trigger=CronTrigger(hour=9, minute=0, day_of_week="mon-fri",
                                timezone="Asia/Shanghai"),
        )
        log.info("[REG] 09:00 (周一到周五)  job_pre_market     -> CASE-D graph.py")

    log.info("=" * 60)
    log.info("[BOOT] 调度器启动, 阻塞运行 (Ctrl+C 退出)")
    log.info("=" * 60)

    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        log.info("[EXIT] 调度器已退出")


if __name__ == "__main__":
    main()
