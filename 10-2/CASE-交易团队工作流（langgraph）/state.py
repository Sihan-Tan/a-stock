# -*- coding: utf-8 -*-
"""
TradingState -- 团队工作流共享状态

LangGraph 的精髓是 "State as a contract"：
  每个节点读取它关心的字段，写回它产出的字段；
  字段名一旦稳定，节点之间就完全解耦，可以独立替换实现。

约定字段所有权：
  Charles 写 investment_view
  Zoe     写 trade_signal
  Kris    写 risk_verdict（也可能改 trade_signal.quantity，例如 ATR 降仓）
  Human   写 approved
  Trader  写 trade_result
  公共    retry_count / messages（每个节点都可 append 一条审计日志）
"""

from typing import Annotated, Optional, TypedDict
from operator import add


class InvestmentView(TypedDict, total=False):
    """Charles 的投研观点"""
    stance: str          # bullish / neutral / bearish
    confidence: float    # 0-1
    summary: str         # 一句话核心观点
    catalysts: list      # 短期/中期催化剂
    risks: list          # 主要风险
    raw_report: str      # Charles 输出的完整研报（Markdown）
    report_md_path: str  # 落盘的 Markdown 研报路径（附件）
    report_html_path: str  # 落盘的 HTML 研报路径（附件，方便人工评阅）


class TradeSignal(TypedDict, total=False):
    """Zoe 的交易信号"""
    stock_code: str      # 600519.SH
    direction: str       # buy / sell / hold
    quantity: int        # 股数（100 整数倍）
    price: float         # 限价；0 表示市价
    reason: str          # 为什么这个时点出这个单
    strategy: str        # MACD / 双均线
    latest_signal: str   # golden_cross / death_cross / bullish / bearish
    latest_close: float  # 最新收盘价
    backtest_winrate: float
    backtest_total_return: float


class RiskVerdict(TypedDict, total=False):
    """Kris 的风控决议"""
    decision: str        # approve / warn / reject / halt
    is_approved: bool
    reason: str
    rule_name: str
    suggested_max_pct: float    # 建议仓位上限比例（< 1.0 时 Zoe 应缩量重发）


class TradeResult(TypedDict, total=False):
    """Trader 的下单回执"""
    dry_run: bool
    order_id: Optional[int]
    submitted_at: str
    note: str


class TradingState(TypedDict, total=False):
    # === 输入 ===
    stock_code: str          # 标的，如 600519.SH
    capital: float           # 可用资金
    user_question: str       # 用户的原始问题（投给 Charles）

    # === 各节点产出 ===
    investment_view: InvestmentView
    trade_signal: TradeSignal
    risk_verdict: RiskVerdict
    approved: Optional[bool]
    trade_result: TradeResult

    # === 控制流 ===
    retry_count: int         # Kris 否决后回到 Zoe 重生成的次数
    max_retry: int

    # === 审计日志（用 Annotated + add 让多个节点可以 append）===
    messages: Annotated[list, add]
