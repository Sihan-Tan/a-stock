# -*- coding: utf-8 -*-
"""
CASE: 风控引擎

风控官 Kris 的核心 -- 8 条核心规则 + 一票否决

8 条规则分布:
    事前 (4): 单笔金额上限 / 价格偏离 / ST 黑名单 / ATR 仓位
    事中 (2): 单日最大亏损熔断 / ATR 止损
    外部 (2): 事件关键词 / 宏观 VIX 门控

设计原则:
    "少而精, 不堆砌" -- 这 8 条覆盖了账户安全 90% 的场景, 一票否决保证执行力。
    其他常见规则 (集中度 / 防抖 / 回撤 / 连亏 / PE / 负债率 / OVX / 10Y 国债 / 滑点等)
    思想都在课件里讲, 代码不实现 -- 避免 "看起来很全, 实际很冗余"。

调用方式:
    kris = RiskManager()
    decision = kris.approve(order, portfolio, context)
    if decision.is_approved:
        execute_order(order)
    else:
        print(decision.reason)
"""
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict


# ============================================================
# 第一部分: 数据结构
# ============================================================

class Decision(Enum):
    """风控决策类型"""
    APPROVE = "approve"        # 批准
    WARN = "warn"              # 警告 (可执行, 但建议降低仓位)
    REJECT = "reject"          # 拒绝当前订单
    HALT = "halt"              # 熔断 (停止所有交易)


@dataclass
class RiskDecision:
    """风控审批结果"""
    decision: Decision
    reason: str
    rule_name: str
    max_position_pct: float = 1.0   # 建议仓位上限比例
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def is_approved(self):
        return self.decision in (Decision.APPROVE, Decision.WARN)

    @property
    def is_rejected(self):
        return self.decision in (Decision.REJECT, Decision.HALT)

    def __repr__(self):
        icon = {
            Decision.APPROVE: "[PASS]",
            Decision.WARN: "[WARN]",
            Decision.REJECT: "[REJECT]",
            Decision.HALT: "[HALT]",
        }[self.decision]
        return f"{icon} {self.rule_name}: {self.reason}"


@dataclass
class Order:
    """订单对象"""
    stock_code: str
    direction: str             # "buy" / "sell"
    amount: float              # 金额 (元)
    price: float               # 委托价格
    quantity: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def __post_init__(self):
        if self.quantity == 0 and self.price > 0:
            self.quantity = int(self.amount / self.price / 100) * 100


# ============================================================
# 第二部分: 事前预防 (4 条)
# ============================================================

class PreTradeGuard:
    """
    事前预防

    4 条核心规则:
      1. 单笔金额上限 -- 防 bug / fat finger
      2. 价格偏离 (Price Collar) -- 防输错价
      3. ST/黑名单 -- 直接拦掉退市风险股
      4. ATR 仓位 -- 海龟法则核心: 让每笔交易承担恒定风险
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.max_order_amount = cfg.get('max_order_amount', 200_000)
        self.price_collar_pct = cfg.get('price_collar_pct', 0.05)
        self.blacklist = set(cfg.get('blacklist', []))
        self.atr_risk_pct = cfg.get('atr_risk_pct', 0.01)        # 单笔风险 1% (海龟标准)
        self.atr_overshoot_ratio = cfg.get('atr_overshoot_ratio', 2.0)  # 超出 ATR 建议 N 倍则警告

    def check_all(self, order: Order, portfolio: dict) -> List[RiskDecision]:
        """跑完所有事前检查"""
        results = [
            self.check_order_amount(order),
            self.check_price_collar(order, portfolio),
            self.check_blacklist(order),
        ]
        atr_result = self.check_atr_position(order, portfolio)
        if atr_result:
            results.append(atr_result)
        return results

    def check_order_amount(self, order: Order) -> RiskDecision:
        """规则1: 单笔金额上限"""
        if order.amount > self.max_order_amount:
            return RiskDecision(
                decision=Decision.REJECT,
                reason=f"单笔金额 {order.amount:,.0f} 超过上限 {self.max_order_amount:,.0f}",
                rule_name="单笔金额上限",
            )
        return RiskDecision(
            decision=Decision.APPROVE,
            reason=f"单笔金额 {order.amount:,.0f} 在限额内",
            rule_name="单笔金额上限",
        )

    def check_price_collar(self, order: Order, portfolio: dict) -> RiskDecision:
        """规则2: 价格偏离检查 (防 fat finger)"""
        prices = portfolio.get('prices', {})
        current_price = prices.get(order.stock_code, 0)
        if current_price <= 0:
            return RiskDecision(
                decision=Decision.APPROVE,
                reason="无现价信息, 跳过价格偏离检查",
                rule_name="价格偏离检查",
            )

        deviation = abs(order.price - current_price) / current_price
        if deviation > self.price_collar_pct:
            return RiskDecision(
                decision=Decision.REJECT,
                reason=f"委托价 {order.price:.3f} 偏离现价 {current_price:.3f} 达 {deviation:.1%}, "
                       f"超过 {self.price_collar_pct:.0%} 限制",
                rule_name="价格偏离检查",
            )
        return RiskDecision(
            decision=Decision.APPROVE,
            reason=f"价格偏离 {deviation:.2%}, 正常",
            rule_name="价格偏离检查",
        )

    def check_blacklist(self, order: Order) -> RiskDecision:
        """规则3: ST/黑名单"""
        # 代码后缀含 _ST 或代码本身在黑名单
        is_st = '_ST' in order.stock_code.upper() or 'ST' in order.stock_code.upper()
        if is_st or order.stock_code in self.blacklist:
            return RiskDecision(
                decision=Decision.REJECT,
                reason=f"{order.stock_code} 命中 ST/黑名单, 禁止买入",
                rule_name="ST黑名单",
            )
        return RiskDecision(
            decision=Decision.APPROVE,
            reason=f"{order.stock_code} 不在黑名单",
            rule_name="ST黑名单",
        )

    def check_atr_position(self, order: Order, portfolio: dict) -> Optional[RiskDecision]:
        """
        规则4: ATR 仓位 (海龟交易法则核心)

        海龟公式: 建议仓位 = (总资产 * risk_pct) / ATR * 价格
        含义:    价格波动 1 个 ATR 时, 账户恰好变动 risk_pct (默认 1%)
        止损:    入场价 - 2 * ATR (海龟 2N 止损)

        实际仓位 > 建议仓位 * overshoot_ratio 时, 发出 WARN 并建议降仓。
        """
        atr_data = portfolio.get('atr', {})
        atr_value = atr_data.get(order.stock_code, 0)
        if atr_value <= 0 or order.direction != 'buy':
            return None

        total_asset = portfolio.get('total_asset', 0)
        if total_asset <= 0:
            return None

        suggested_position = (total_asset * self.atr_risk_pct) / atr_value * order.price
        atr_stop_price = order.price - 2 * atr_value

        if order.amount > suggested_position * self.atr_overshoot_ratio:
            return RiskDecision(
                decision=Decision.WARN,
                reason=f"ATR={atr_value:.3f}, 建议仓位 {suggested_position:,.0f}元, "
                       f"实际 {order.amount:,.0f}元 (超 {order.amount/suggested_position:.1f}倍). "
                       f"ATR止损价: {atr_stop_price:.3f}",
                rule_name="ATR仓位检查",
                max_position_pct=suggested_position / order.amount,
            )
        return RiskDecision(
            decision=Decision.APPROVE,
            reason=f"ATR={atr_value:.3f}, 建议仓位 {suggested_position:,.0f}元, "
                   f"实际 {order.amount:,.0f}元, 风险可控. ATR止损参考: {atr_stop_price:.3f}",
            rule_name="ATR仓位检查",
        )


# ============================================================
# 第三部分: 事中熔断 (2 条)
# ============================================================

class CircuitBreaker:
    """
    事中熔断

    2 条核心规则:
      1. 单日最大亏损熔断
         -- 当日净值跌幅触及阈值 (默认 2%) 时, 暂停所有买入。
            这一条覆盖了 "日内回撤" / "连续亏损笔数" 等多种账户安全场景。
      2. ATR 自适应止损 (海龟 2N 止损)
         -- 持仓中价格触及 入场价 - 2*ATR 时, 强制平仓。
            高波动时止损宽 (给趋势空间), 低波动时止损紧 (快速止损)。
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.max_daily_loss_pct = cfg.get('max_daily_loss_pct', 0.02)
        self.atr_stop_multiplier = cfg.get('atr_stop_multiplier', 2.0)

        # 当日状态
        self.daily_start_nav = 0.0
        self.current_nav = 0.0
        self.is_halted = False
        self.halt_reason = ""
        # 持仓的 ATR 止损价: {stock_code: {entry_price, atr, stop_price}}
        self.atr_stops: Dict[str, dict] = {}

    def reset_daily(self, start_nav: float):
        """每日开盘前重置"""
        self.daily_start_nav = start_nav
        self.current_nav = start_nav
        self.is_halted = False
        self.halt_reason = ""

    def update_nav(self, nav: float) -> Optional[RiskDecision]:
        """每笔成交后更新净值, 检查日亏损熔断"""
        self.current_nav = nav
        if self.daily_start_nav <= 0:
            return None
        loss_pct = (self.daily_start_nav - nav) / self.daily_start_nav
        if loss_pct >= self.max_daily_loss_pct:
            self.is_halted = True
            self.halt_reason = f"单日亏损 {loss_pct:.2%} 触发熔断线 {self.max_daily_loss_pct:.2%}"
            return RiskDecision(
                decision=Decision.HALT,
                reason=self.halt_reason,
                rule_name="单日亏损熔断",
            )
        return None

    def register_position(self, stock_code: str, entry_price: float, atr: float):
        """开仓时登记 ATR 止损位"""
        if atr <= 0:
            return
        self.atr_stops[stock_code] = {
            'entry_price': entry_price,
            'atr': atr,
            'stop_price': entry_price - self.atr_stop_multiplier * atr,
        }

    def remove_position(self, stock_code: str):
        """平仓时移除登记"""
        self.atr_stops.pop(stock_code, None)

    def check_atr_stop(self, stock_code: str, current_price: float) -> Optional[RiskDecision]:
        """
        检查 ATR 止损: 价格跌破 入场价 - 2*ATR 时强制平仓
        返回 REJECT 表示触发, 上层应执行平仓。
        """
        info = self.atr_stops.get(stock_code)
        if not info:
            return None
        if current_price <= info['stop_price']:
            return RiskDecision(
                decision=Decision.REJECT,
                reason=f"{stock_code} 触发ATR止损: 入场价 {info['entry_price']:.3f}, "
                       f"ATR={info['atr']:.3f}, 止损价 {info['stop_price']:.3f}, "
                       f"现价 {current_price:.3f}",
                rule_name="ATR止损",
            )
        return None

    def get_status(self) -> dict:
        daily_pnl_pct = ((self.current_nav - self.daily_start_nav) / self.daily_start_nav
                         if self.daily_start_nav > 0 else 0)
        return {
            'daily_start_nav': self.daily_start_nav,
            'current_nav': self.current_nav,
            'daily_pnl_pct': daily_pnl_pct,
            'is_halted': self.is_halted,
            'halt_reason': self.halt_reason,
            'atr_stops': dict(self.atr_stops),
        }


# ============================================================
# 第四部分: 外部信号 (2 条)
# ============================================================

class EventKeywordChecker:
    """
    事件风控 (外部信号 1)

    关键词匹配 -- 只保留 reject 一档, 避免误伤。
    课件中会讲 reject + warn 两档思想, 代码这里只实现 reject。

    使用方式:
        # 用户层把新闻文本传进来 (硬编码 / 实时抓取均可)
        ek = EventKeywordChecker()
        decision = ek.check(stock_code, news_text)
    """

    BEARISH_KEYWORDS = [
        '退市', '暂停上市', '终止上市', '*ST',
        '立案调查', '立案侦查', '行政处罚',
        '涉嫌违法', '涉嫌犯罪', '财务造假',
    ]

    def __init__(self, keywords: List[str] = None):
        self.keywords = keywords or list(self.BEARISH_KEYWORDS)

    def check(self, stock_code: str, news_text: str) -> RiskDecision:
        if not news_text:
            return RiskDecision(
                decision=Decision.APPROVE,
                reason=f"{stock_code} 无近期新闻, 跳过",
                rule_name="事件关键词",
            )
        for kw in self.keywords:
            if kw in news_text:
                return RiskDecision(
                    decision=Decision.REJECT,
                    reason=f"{stock_code} 新闻命中重大利空: {kw}",
                    rule_name="事件关键词",
                )
        return RiskDecision(
            decision=Decision.APPROVE,
            reason=f"{stock_code} 新闻未发现重大利空",
            rule_name="事件关键词",
        )


class EventLLMChecker:
    """
    事件风控 (升级版) -- 用大模型替代关键词

    关键词体系的局限:
        1) 早期模糊信号拦不住 (獐子岛 2014 年第一次"扇贝跑了")
        2) 同义词 / 变体表达漏掉 ("被立案" vs "立案调查")
        3) 假阳性 ("建议立案管理"等无关新闻偶尔触发)

    LLM 解决思路:
        把新闻喂给通义千问, 让它回答 REJECT/WARN/APPROVE + 一句理由。
        这是 "关键词体系" 的进化形态, 也是和第 16 讲舆情系统的完整衔接点。

    使用方式:
        ek_llm = EventLLMChecker()
        decision = ek_llm.check('600519', news_text)

    依赖:
        pip install dashscope
        export DASHSCOPE_API_KEY=sk-xxx   (Windows: $env:DASHSCOPE_API_KEY)
    """

    PROMPT_TEMPLATE = """你是 A 股专业风控官 Kris, 现在审批一笔买入订单。
请只根据下面的新闻文本, 判断这只股票是否存在重大利空, 决定是否允许买入。

判定口径 (从严):
  - REJECT: 命中重大利空 (立案调查 / 退市 / 财务造假 / 重大违规 / 高管被抓 / 重大债务违约 / 实控人被立案 等)
  - WARN:   存在中等风险信号 (大额减持 / 业绩大幅下滑 / 商誉减值 / 重要诉讼 / 监管问询)
  - APPROVE: 新闻中性或正面, 无明显风险

输出格式 (严格遵守, 不要任何额外解释):
  第一行: REJECT / WARN / APPROVE 之一
  第二行: 一句话理由 (不超过 40 字)

股票代码: {stock_code}
新闻文本:
\"\"\"
{news_text}
\"\"\"
"""

    def __init__(self, model: str = "qwen-turbo", api_key: str = None):
        self.model = model
        self.api_key = api_key  # None 时 dashscope 会自动读 DASHSCOPE_API_KEY 环境变量

    def check(self, stock_code: str, news_text: str) -> RiskDecision:
        if not news_text or not news_text.strip():
            return RiskDecision(
                decision=Decision.APPROVE,
                reason=f"{stock_code} 无近期新闻, 跳过",
                rule_name="事件LLM审批",
            )

        from dashscope import Generation
        prompt = self.PROMPT_TEMPLATE.format(
            stock_code=stock_code,
            news_text=news_text[:3000],  # 截断防止超长
        )
        kwargs = {'model': self.model, 'prompt': prompt}
        if self.api_key:
            kwargs['api_key'] = self.api_key
        resp = Generation.call(**kwargs)

        if resp.status_code != 200:
            raise RuntimeError(f"DashScope 调用失败: {resp.code} {resp.message}")

        text = resp.output.text.strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        verdict = lines[0].upper() if lines else ""
        reason = lines[1] if len(lines) > 1 else text

        # 兼容大模型偶尔混入 markdown 标记
        verdict = verdict.replace('*', '').replace('#', '').strip()

        if 'REJECT' in verdict:
            d_type = Decision.REJECT
        elif 'WARN' in verdict:
            d_type = Decision.WARN
        elif 'APPROVE' in verdict:
            d_type = Decision.APPROVE
        else:
            # 模型没按格式输出 -> 视为最严, 拒绝 (一票否决原则)
            d_type = Decision.REJECT
            reason = f"模型输出无法解析, 按从严处理: {text[:60]}"

        return RiskDecision(
            decision=d_type,
            reason=f"{stock_code} LLM判定: {reason}",
            rule_name="事件LLM审批",
        )


class MacroGate:
    """
    宏观门控 (外部信号 2)

    只用 VIX 一个指标 -> 仓位系数。
    其他可选指标 (OVX 原油波动率 / 美 10Y 国债收益率) 在课件里作为 "可扩展点" 介绍, 代码不实现。

    VIX 区间映射 (阶梯型, 边界用线性插值平滑):
        VIX < 20      -> 100%   正常
        20-25 (线性)  -> 70%    焦虑
        25-35 (线性)  -> 30%    恐慌
        35-50 (线性)  -> 10%    极度恐慌
        VIX >= 50     -> 0%     末日级别 (2008-10 / 2020-03), 暂停开仓
    """

    def __init__(self):
        self.current_vix = None
        self.position_coefficient = 1.0
        self.risk_level = "未知"

    def update_vix(self, vix: float) -> float:
        """传入 VIX 值, 返回当前仓位系数"""
        self.current_vix = vix
        if vix >= 50:
            self.position_coefficient = 0.0
            self.risk_level = "末日级别"
        elif vix >= 35:
            # 35-50 线性: 30% -> 10%
            self.position_coefficient = 0.30 - (vix - 35) / 15 * 0.20
            self.risk_level = "极度恐慌"
        elif vix >= 25:
            # 25-35 线性: 70% -> 30%
            self.position_coefficient = 0.70 - (vix - 25) / 10 * 0.40
            self.risk_level = "恐慌"
        elif vix >= 20:
            # 20-25 线性: 100% -> 70%
            self.position_coefficient = 1.00 - (vix - 20) / 5 * 0.30
            self.risk_level = "焦虑"
        else:
            self.position_coefficient = 1.0
            self.risk_level = "正常"
        return self.position_coefficient

    def check(self) -> RiskDecision:
        if self.current_vix is None:
            return RiskDecision(
                decision=Decision.APPROVE,
                reason="未提供 VIX, 跳过宏观门控",
                rule_name="宏观VIX门控",
            )
        if self.position_coefficient <= 0:
            return RiskDecision(
                decision=Decision.HALT,
                reason=f"VIX={self.current_vix:.1f} 极度恐慌, 暂停所有开仓",
                rule_name="宏观VIX门控",
                max_position_pct=0.0,
            )
        if self.position_coefficient < 1.0:
            return RiskDecision(
                decision=Decision.WARN,
                reason=f"VIX={self.current_vix:.1f} ({self.risk_level}), "
                       f"仓位系数降至 {self.position_coefficient:.0%}",
                rule_name="宏观VIX门控",
                max_position_pct=self.position_coefficient,
            )
        return RiskDecision(
            decision=Decision.APPROVE,
            reason=f"VIX={self.current_vix:.1f} 正常, 仓位系数 100%",
            rule_name="宏观VIX门控",
        )


# ============================================================
# 第五部分: 风控官 Kris 统一入口
# ============================================================

class RiskManager:
    """
    风控官 Kris -- 统一入口

    审批流程 (一票否决):
        宏观门控 -> 事件关键词 -> 事前 4 条 -> (熔断状态)
        任一返回 REJECT/HALT, 整笔订单否决。

    使用方式:
        kris = RiskManager()
        # 1) 把宏观信号灌进去 (每天 / 每盘前一次)
        kris.macro.update_vix(19.5)
        # 2) 审批订单
        decision = kris.approve(order, portfolio,
                                context={'news_text': '...近期新闻...'})
        # 3) 成交后回报净值
        halt = kris.on_trade_complete(new_nav)
        # 4) 持仓时检查 ATR 止损
        stop = kris.check_atr_stop(stock_code, current_price)
    """

    def __init__(self, config: dict = None, event_checker=None):
        """
        参数:
            config:        各模块配置 dict
            event_checker: 可选, 注入自定义事件检查器 (如 EventLLMChecker)
                           不传时默认用 EventKeywordChecker (关键词模式)
        """
        cfg = config or {}
        self.pre_trade = PreTradeGuard(cfg.get('pre_trade', {}))
        self.circuit_breaker = CircuitBreaker(cfg.get('circuit_breaker', {}))
        self.event = event_checker or EventKeywordChecker(cfg.get('event_keywords'))
        self.macro = MacroGate()
        self.audit_log: List[dict] = []

    # ----- 主审批 -----
    def approve(self, order: Order, portfolio: dict,
                context: dict = None) -> RiskDecision:
        """
        审批订单

        参数:
            order:     待审批订单
            portfolio: {'total_asset': 总资产, 'prices': {code: price},
                        'atr': {code: atr_value}}
            context:   {'news_text': 该股近期新闻全文} 可选
        """
        context = context or {}

        # 1) 已熔断: 直接拒
        if self.circuit_breaker.is_halted:
            d = RiskDecision(Decision.HALT,
                             f"交易已熔断: {self.circuit_breaker.halt_reason}",
                             rule_name="熔断状态")
            self._log(order, d)
            return d

        # 2) 宏观门控
        macro_d = self.macro.check()
        if macro_d.decision == Decision.HALT:
            self._log(order, macro_d)
            return macro_d

        all_checks: List[RiskDecision] = [macro_d]

        # 3) 事件关键词 (买入才检查)
        if order.direction == 'buy':
            news_text = context.get('news_text', '')
            event_d = self.event.check(order.stock_code, news_text)
            all_checks.append(event_d)
            if event_d.decision == Decision.REJECT:
                final = self._get_strictest(all_checks)
                self._log(order, final)
                return final

        # 4) 事前 4 条
        all_checks.extend(self.pre_trade.check_all(order, portfolio))

        # 5) 取最严格的决策
        final = self._get_strictest(all_checks)
        self._log(order, final)
        return final

    # ----- 事中回调 -----
    def on_trade_complete(self, nav: float) -> Optional[RiskDecision]:
        """每笔成交后更新净值, 返回 RiskDecision 表示触发熔断"""
        return self.circuit_breaker.update_nav(nav)

    def register_position(self, stock_code: str, entry_price: float, atr: float):
        """成交后登记 ATR 止损位"""
        self.circuit_breaker.register_position(stock_code, entry_price, atr)

    def remove_position(self, stock_code: str):
        """平仓后移除"""
        self.circuit_breaker.remove_position(stock_code)

    def check_atr_stop(self, stock_code: str,
                       current_price: float) -> Optional[RiskDecision]:
        """盘中调用: 检查 ATR 止损是否触发"""
        return self.circuit_breaker.check_atr_stop(stock_code, current_price)

    def start_day(self, start_nav: float):
        """每日开盘前调用"""
        self.circuit_breaker.reset_daily(start_nav)

    # ----- 内部工具 -----
    def _get_strictest(self, decisions: List[RiskDecision]) -> RiskDecision:
        """从多个决策中取最严的 (HALT > REJECT > WARN > APPROVE)"""
        severity = {Decision.HALT: 4, Decision.REJECT: 3,
                    Decision.WARN: 2, Decision.APPROVE: 1}
        decisions_sorted = sorted(decisions,
                                  key=lambda d: severity.get(d.decision, 0),
                                  reverse=True)
        strictest = decisions_sorted[0]

        rejections = [d for d in decisions if d.is_rejected]
        warnings = [d for d in decisions if d.decision == Decision.WARN]

        if rejections:
            reasons = "; ".join(f"[{d.rule_name}] {d.reason}" for d in rejections)
            return RiskDecision(
                decision=strictest.decision,
                reason=reasons,
                rule_name="综合审批",
            )
        if warnings:
            min_pct = min(d.max_position_pct for d in warnings)
            reasons = "; ".join(f"[{d.rule_name}] {d.reason}" for d in warnings)
            return RiskDecision(
                decision=Decision.WARN,
                reason=reasons,
                rule_name="综合审批",
                max_position_pct=min_pct,
            )
        return RiskDecision(
            decision=Decision.APPROVE,
            reason="所有检查通过",
            rule_name="综合审批",
        )

    def _log(self, order: Order, decision: RiskDecision):
        self.audit_log.append({
            'time': decision.timestamp,
            'stock': order.stock_code,
            'direction': order.direction,
            'amount': order.amount,
            'decision': decision.decision.value,
            'rule': decision.rule_name,
            'reason': decision.reason,
        })

    def print_audit_log(self, last_n: int = 20):
        print("\n" + "=" * 70)
        print(f"  Kris 审批日志 (最近 {last_n} 条)")
        print("=" * 70)
        icon = {'approve': 'OK', 'warn': '!!', 'reject': 'XX', 'halt': 'XX'}
        for entry in self.audit_log[-last_n:]:
            tag = icon.get(entry['decision'], '??')
            print(f"  [{tag}] {entry['time']} | {entry['stock']} {entry['direction']} "
                  f"{entry['amount']:>10,.0f} | {entry['rule']}: {entry['reason'][:60]}")
        print("=" * 70)

    def get_summary(self) -> dict:
        total = len(self.audit_log)
        approved = sum(1 for e in self.audit_log if e['decision'] == 'approve')
        warned = sum(1 for e in self.audit_log if e['decision'] == 'warn')
        rejected = sum(1 for e in self.audit_log if e['decision'] in ('reject', 'halt'))
        return {
            'total': total,
            'approved': approved,
            'warned': warned,
            'rejected': rejected,
            'rejection_rate': rejected / total if total > 0 else 0,
            'circuit_breaker': self.circuit_breaker.get_status(),
            'macro': {
                'vix': self.macro.current_vix,
                'coefficient': self.macro.position_coefficient,
                'risk_level': self.macro.risk_level,
            },
        }


# ============================================================
# 主程序: 跑一遍 8 条规则的最小演示
# ============================================================

if __name__ == '__main__':
    print("=" * 70)
    print("CASE: 风控引擎 -- 8 条核心规则演示")
    print("=" * 70)

    kris = RiskManager(config={
        'pre_trade': {
            'max_order_amount': 200_000,
            'price_collar_pct': 0.05,
            'blacklist': [],
        },
        'circuit_breaker': {
            'max_daily_loss_pct': 0.02,
        },
    })
    kris.start_day(1_000_000)

    # 提前灌一个正常的 VIX
    kris.macro.update_vix(18.5)

    portfolio = {
        'total_asset': 1_000_000,
        'prices': {'510050.SH': 3.00, '600519.SH': 1700},
        'atr': {'510050.SH': 0.05, '600519.SH': 30.0},
    }

    test_cases = [
        ("正常订单",
         Order('510050.SH', 'buy', 100_000, 3.00),
         {'news_text': '50ETF成交活跃, 资金面平稳'}),
        ("超限额",
         Order('510050.SH', 'buy', 500_000, 3.00),
         {}),
        ("价格偏离 (fat finger)",
         Order('510050.SH', 'buy', 50_000, 3.40),
         {}),
        ("ST 黑名单",
         Order('000001.SZ_ST', 'buy', 30_000, 10.0),
         {}),
        ("ATR 仓位超标",
         Order('600519.SH', 'buy', 800_000, 1700),
         {'news_text': '茅台业绩稳定增长'}),
        ("事件利空 (立案调查)",
         Order('600519.SH', 'buy', 50_000, 1700),
         {'news_text': '据悉, 公司昨日被证监会立案调查'}),
    ]

    for desc, order, ctx in test_cases:
        print(f"\n--- {desc} ---")
        d = kris.approve(order, portfolio, ctx)
        print(f"  {d}")

    # 模拟事中熔断: 净值跌到 97 万 (亏 3%)
    print("\n--- 净值跌到 970,000 (-3%) ---")
    halt = kris.on_trade_complete(970_000)
    if halt:
        print(f"  {halt}")

    # 熔断后再下单 -> 直接 HALT
    print("\n--- 熔断后再下单 ---")
    d = kris.approve(Order('510050.SH', 'buy', 50_000, 3.00), portfolio, {})
    print(f"  {d}")

    # 切换 VIX 到极度恐慌
    print("\n--- 重启日, VIX=42 极度恐慌 ---")
    kris.start_day(970_000)
    kris.macro.update_vix(42)
    d = kris.approve(Order('510050.SH', 'buy', 50_000, 3.00), portfolio, {})
    print(f"  {d}")

    kris.print_audit_log(20)

    summary = kris.get_summary()
    print(f"\n统计: 共 {summary['total']} 笔, "
          f"通过 {summary['approved']}, 警告 {summary['warned']}, "
          f"拒绝/熔断 {summary['rejected']}, 拒绝率 {summary['rejection_rate']:.1%}")
