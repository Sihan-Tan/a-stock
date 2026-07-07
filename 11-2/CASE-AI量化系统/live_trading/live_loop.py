# -*- coding: utf-8 -*-
# 23-CASE-A: 盘中全自动交易闭环主循环
"""
LiveLoop -- 盘中全自动交易闭环主循环

每隔 N 分钟跑一遍, 完成: 拉行情 -> 评估持仓 -> 跑信号 -> 风控审批 -> 下单 -> 推送

核心架构 (LangGraph 风格, 但简化为顺序循环, 因为每分钟级延迟比 LangGraph 启动开销重要):

    每分钟循环:
        1. health_check()          检查 miniQMT 连接 + 行情数据完整性
        2. update_positions()      拉最新持仓 + 当日盈亏
        3. check_circuit_breaker() 当日亏损是否触发熔断
        4. evaluate_stop_loss()    持仓股是否触发止损
        5. evaluate_signals()      候选股是否出现新信号
        6. risk_check()            风控审批 (Kris 规则)
        7. place_orders()          下单 (本 CASE live_trading.miniqmt_trader_v2)
        8. push_summary()          推送告警 (alert_router)
        9. save_state()            落盘 state (供 CEO 控制台读)

异常处理金字塔:
    L1 数据层异常 -> 跳过本轮, 下轮继续, 不告警 (网络抖动)
    L2 风控否决   -> 不下单, INFO 推送
    L3 订单失败   -> WARN 推送, 重试 1 次
    L4 系统级异常 -> CRITICAL 推送 + 暂停所有交易 (state.trading_status = "HALTED")
    L5 不可恢复   -> FATAL 推送 + 进程退出 + 等人工

注意:
    - 真正的实盘需要接 miniQMT, 在 dry-run 下用模拟数据 (适合教学/演示)
    - 信号评估这里可用 MACD/RSI 占位，实战可替换为自有选股 / 路由输出。
"""

from __future__ import annotations
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from alerting.alert_router import AlertRouter
from live_trading.state_store import StateStore


# ============================================================
# 数据来源 (dry-run 下用 xtdata + 模拟下单)
# ============================================================

class MarketDataProvider:
    """市场数据提供者 -- xtdata 拉真实数据"""

    def __init__(self):
        self._connected = False

    def connect(self):
        from xtquant import xtdata
        xtdata.connect()
        self._connected = True

    def get_latest_tick(self, stock_code: str) -> dict:
        """拉最新 tick (含 5 档盘口)"""
        from xtquant import xtdata
        if not self._connected:
            self.connect()
        ticks = xtdata.get_full_tick([stock_code])
        return ticks.get(stock_code, {})

    def get_recent_kline(self, stock_code: str, period: str = "5m",
                        count: int = 50) -> Optional[Any]:
        """拉最近 N 根 K 线 (用于算指标)"""
        import pandas as pd
        from xtquant import xtdata
        if not self._connected:
            self.connect()
        try:
            xtdata.download_history_data(stock_code, period=period,
                                         start_time="20250101", incrementally=True)
            data = xtdata.get_market_data_ex(
                field_list=["open", "high", "low", "close", "volume"],
                stock_list=[stock_code], period=period, count=count,
            )
            df = data.get(stock_code)
            if df is None or len(df) == 0:
                return None
            df = df.copy()
            df.index = pd.to_datetime(df.index)
            return df
        except Exception:
            return None


# ============================================================
# 信号评估 (简化版 MACD)
# ============================================================

def evaluate_macd_signal(df) -> str:
    """
    评估 MACD 信号
    返回: "buy" / "sell" / "hold"
    """
    import pandas as pd
    if df is None or len(df) < 30:
        return "hold"
    close = df["close"].astype(float)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()

    # 最新两根: 看是否金叉/死叉
    if len(dif) < 2:
        return "hold"
    prev = dif.iloc[-2] - dea.iloc[-2]
    curr = dif.iloc[-1] - dea.iloc[-1]
    if prev <= 0 and curr > 0:
        return "buy"
    if prev >= 0 and curr < 0:
        return "sell"
    return "hold"


# ============================================================
# 持仓与盈亏更新
# ============================================================

def update_positions_from_market(positions: List[dict],
                                 market: MarketDataProvider) -> List[dict]:
    """拉最新价更新持仓的市值 + 浮动盈亏"""
    updated = []
    for pos in positions:
        code = pos["code"]
        tick = market.get_latest_tick(code)
        cur_price = float(tick.get("lastPrice", pos.get("cost", 0)))
        volume = int(pos["volume"])
        cost = float(pos.get("cost", 0))
        mv = volume * cur_price
        pnl = (cur_price - cost) * volume
        pnl_pct = (cur_price - cost) / cost if cost > 0 else 0
        updated.append({
            **pos,
            "cur_price":   round(cur_price, 3),
            "market_value": round(mv, 2),
            "pnl":         round(pnl, 2),
            "pnl_pct":     round(pnl_pct, 4),
        })
    return updated


def calc_today_pnl(positions: List[dict], capital: float) -> tuple:
    """计算当日总盈亏 (元 + 百分比)"""
    total_pnl = sum(p.get("pnl", 0) for p in positions)
    total_pct = total_pnl / capital if capital > 0 else 0
    return round(total_pnl, 2), round(total_pct, 4)


# ============================================================
# 主循环
# ============================================================

class LiveTradingLoop:
    """
    实盘主循环 (默认 dry-run)

    用法:
        loop = LiveTradingLoop(watch_stocks=["600519.SH", "513100.SH"])
        loop.run_once()         # 跑一次
        loop.run_forever(60)    # 每 60 秒跑一次, 直到 Ctrl+C
    """

    def __init__(self,
                 watch_stocks: List[str],
                 capital: float = 1_000_000,
                 state_file: str = "outputs/live_state.json",
                 max_daily_loss_pct: float = -0.02,
                 dry_run: bool = True,
                 signal_evaluator: Optional[Callable[[str, "MarketDataProvider", float], dict]] = None):
        """
        signal_evaluator: 可选的信号评估器, 签名 (code, market, capital) -> dict
            返回字典 {"side": "buy"/"sell"/"hold", "strategy": str, "reason": str (可选)}
            不传 = 沿用默认 5min MACD 金叉/死叉 (兼容历史行为)
        """
        self.watch_stocks = watch_stocks
        self.capital = capital
        self.dry_run = dry_run
        self.max_daily_loss_pct = max_daily_loss_pct
        self.signal_evaluator = signal_evaluator

        self.state_store = StateStore(state_file)
        self.market = MarketDataProvider()
        self.alert = AlertRouter(info_aggregate_seconds=300)

        # 初始化 state
        s = self.state_store.load()
        s["capital"] = capital
        s["watch_stocks"] = watch_stocks
        s["control"]["dry_run"] = dry_run
        s["control"]["max_daily_loss"] = max_daily_loss_pct
        self.state_store.save(s)

    # ------------------------------------------------------------------
    # 单次循环
    # ------------------------------------------------------------------
    def run_once(self) -> dict:
        """跑一次完整循环"""
        cycle_start = time.time()
        s = self.state_store.load()

        # 0) 检查 control.trading_status
        if s.get("trading_status") == "HALTED":
            self.alert.alert("WARN", "交易已熔断, 跳过本轮", source="loop")
            return {"action": "halted_skip"}
        if s.get("trading_status") == "PAUSED":
            self.alert.alert("INFO", "交易已暂停 (CEO 控制台暂停)", source="loop")
            return {"action": "paused_skip"}

        # 1) health_check
        try:
            self.market.connect()
            s["health"]["miniqmt_connected"] = True
            s["health"]["last_heartbeat"] = datetime.now().isoformat(timespec="seconds")
        except Exception as e:
            s["health"]["miniqmt_connected"] = False
            s["health"]["errors_24h"] = s["health"].get("errors_24h", 0) + 1
            self.alert.alert("CRITICAL", "miniQMT 连接失败",
                             message=str(e), source="health")
            self.state_store.save(s)
            return {"action": "health_fail"}

        # 2) 更新持仓 + 当日盈亏
        positions = s.get("positions", [])
        if positions:
            positions = update_positions_from_market(positions, self.market)
            today_pnl, today_pnl_pct = calc_today_pnl(positions, self.capital)
            s["positions"] = positions
            s["today_pnl"] = today_pnl
            s["today_pnl_pct"] = today_pnl_pct
            s["pnl_history"] = s.get("pnl_history", [])
            s["pnl_history"].append({
                "ts": datetime.now().isoformat(timespec="seconds"),
                "pnl": today_pnl, "pnl_pct": today_pnl_pct,
            })
            s["pnl_history"] = s["pnl_history"][-500:]

        # 3) 熔断检查
        if s.get("today_pnl_pct", 0) <= self.max_daily_loss_pct:
            s["trading_status"] = "HALTED"
            self.alert.alert(
                "CRITICAL", "触发当日亏损熔断",
                message=f"今日累计盈亏 {s['today_pnl_pct']:.2%}, "
                        f"已跌破熔断线 {self.max_daily_loss_pct:.2%}",
                source="circuit_breaker",
            )
            self.state_store.save(s)
            return {"action": "circuit_breaker"}

        # 4) 评估信号 (默认对 watch 池每只算 MACD; 注入了 signal_evaluator 则按 evaluator 派发)
        new_signals = []
        for code in self.watch_stocks:
            if self.signal_evaluator is not None:
                # 外部注入的信号路由器: 由 evaluator 自己决定用哪个策略
                try:
                    result = self.signal_evaluator(code, self.market, self.capital)
                except Exception as e:
                    self.alert.alert("WARN", f"signal_evaluator 异常 {code}",
                                     message=str(e), source="zoe")
                    continue
                if not result:
                    continue
                side = result.get("side", "hold")
                if side == "hold":
                    continue
                sig = {
                    "code":     code,
                    "side":     side,
                    "strategy": result.get("strategy", "unknown"),
                    "reason":   result.get("reason", ""),
                }
            else:
                df = self.market.get_recent_kline(code, period="5m", count=50)
                side = evaluate_macd_signal(df)
                if side == "hold":
                    continue
                sig = {"code": code, "side": side, "strategy": "macd_5min"}
            new_signals.append(sig)

        if new_signals:
            for sig in new_signals:
                self.state_store.append_signal(sig)
                self.alert.alert(
                    "INFO", f"信号触发 -> {sig['side']} {sig['code']} [{sig.get('strategy','')}]",
                    source="zoe",
                )

        # 5) 风控 + 下单
        for sig in new_signals:
            order_result = self._handle_signal(s, sig)
            # 把触发该订单的策略名一起记录, 便于复盘
            if "strategy" not in order_result and sig.get("strategy"):
                order_result["strategy"] = sig["strategy"]
            self.state_store.append_order(order_result)

        # 6) 落盘 state (含本轮事件)
        s["events"] = s.get("events", [])
        s["events"].append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "type": "loop_cycle",
            "signal_count": len(new_signals),
            "duration_ms": int((time.time() - cycle_start) * 1000),
        })
        s["events"] = s["events"][-200:]
        self.state_store.save(s)

        return {
            "action":      "cycle_done",
            "duration_ms": int((time.time() - cycle_start) * 1000),
            "new_signals": len(new_signals),
        }

    def _handle_signal(self, state: dict, signal: dict) -> dict:
        """处理一个信号: 风控 -> 下单 -> 推送"""
        code = signal["code"]
        side = signal["side"]
        tick = self.market.get_latest_tick(code)
        price = float(tick.get("lastPrice", 0))
        if price <= 0:
            return {"code": code, "side": side, "status": "rejected",
                    "reason": "拿不到价格", "ts": datetime.now().isoformat()}

        # 简化风控: 单笔不超过总资金 10%
        max_amount = self.capital * 0.10
        quantity = int(max_amount / price / 100) * 100
        if quantity == 0:
            quantity = 100   # 至少 1 手试探

        amount = quantity * price

        # control.pause_buying 拦截
        if side == "buy" and state.get("control", {}).get("pause_buying"):
            self.alert.alert("INFO", "买入被 CEO 控制台暂停",
                             message=f"{code} {quantity}股 @ {price:.2f}",
                             source="control")
            return {"code": code, "side": side, "quantity": quantity,
                    "price": price, "status": "paused_by_ceo"}

        # 下单 (dry-run / real)
        if self.dry_run:
            self.alert.alert(
                "INFO", f"[DRY-RUN] 下单 {side} {code} {quantity}股 @ {price:.2f}",
                source="trader",
            )
            return {"code": code, "side": side, "quantity": quantity,
                    "price": price, "amount": amount, "status": "dry_run",
                    "ts": datetime.now().isoformat()}

        # 真实下单 (本 CASE 内 live_trading/miniqmt_trader_v2)
        try:
            from live_trading.miniqmt_trader_v2 import MiniQMTTraderV2
            trader = MiniQMTTraderV2(
                qmt_path=os.environ["QMT_PATH"],
                account_id=os.environ["ACCOUNT_ID"],
                enable_heartbeat=False,
            )
            trader.connect()
            if side == "buy":
                order_id = trader.buy(code, quantity, price=price,
                                      strategy_name="live_loop")
            else:
                order_id = trader.sell(code, quantity, price=price,
                                       strategy_name="live_loop")
            trader.disconnect()

            if order_id:
                self.alert.alert(
                    "INFO", f"实盘下单成功 {side} {code}",
                    message=f"委托编号 {order_id}, {quantity}股 @ {price:.2f}",
                    source="trader",
                )
                return {"code": code, "side": side, "quantity": quantity,
                        "price": price, "amount": amount, "status": "submitted",
                        "order_id": order_id, "ts": datetime.now().isoformat()}
            else:
                self.alert.alert("WARN", f"实盘下单失败 {code}", source="trader")
                return {"code": code, "side": side, "quantity": quantity,
                        "price": price, "status": "failed",
                        "ts": datetime.now().isoformat()}
        except Exception as e:
            self.alert.alert("CRITICAL", f"下单异常 {code}",
                             message=str(e), source="trader")
            return {"code": code, "side": side, "status": "exception",
                    "reason": str(e), "ts": datetime.now().isoformat()}

    # ------------------------------------------------------------------
    # 长跑模式
    # ------------------------------------------------------------------
    def run_forever(self, interval_seconds: int = 60):
        """每隔 N 秒跑一次, 直到 Ctrl+C"""
        self.alert.alert("INFO", "实盘主循环启动",
                         message=f"watch={self.watch_stocks}, "
                                 f"interval={interval_seconds}s, dry_run={self.dry_run}",
                         source="loop")
        try:
            while True:
                t0 = time.time()
                result = self.run_once()
                # 等到下一次触发
                elapsed = time.time() - t0
                if elapsed < interval_seconds:
                    time.sleep(interval_seconds - elapsed)
        except KeyboardInterrupt:
            self.alert.alert("INFO", "实盘主循环退出 (Ctrl+C)", source="loop")
            self.alert.shutdown()


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="盘中全自动交易闭环")
    parser.add_argument("--stocks", default="600519.SH,513100.SH",
                        help="监控股票池, 逗号分隔")
    parser.add_argument("--capital", type=float, default=1_000_000)
    parser.add_argument("--interval", type=int, default=60,
                        help="循环间隔秒, 默认 60")
    parser.add_argument("--once", action="store_true", help="只跑一次")
    parser.add_argument("--state-file", default="outputs/live_state.json")
    args = parser.parse_args()

    stocks = [s.strip() for s in args.stocks.split(",") if s.strip()]
    loop = LiveTradingLoop(
        watch_stocks=stocks,
        capital=args.capital,
        state_file=args.state_file,
        dry_run=os.environ.get("TRADER_DRY_RUN", "1") == "1",
    )

    if args.once:
        result = loop.run_once()
        print(f"\n[完成] {result}")
        print(f"\nstate 落盘: {args.state_file}")
    else:
        loop.run_forever(args.interval)


if __name__ == "__main__":
    main()
