# -*- coding: utf-8 -*-
"""
经典海龟交易策略 - ATR 仓位管理 + 金字塔加仓

====================================================================
海龟交易法则核心理念
====================================================================

海龟交易法则之所以在交易界享有盛誉，不是因为它发现了什么"神奇指标"，
而是因为它对交易的本质有深刻理解:

  普通交易者问: "什么时候买？"   (关注入场时机)
  海龟交易者问: "买多少？"       (关注风险管理)

这就是海龟法则的核心洞见: 长期来看，资金管理比入场信号更重要。

====================================================================
四大组件详解
====================================================================

1. 唐奇安通道 (Donchian Channel) - 趋势识别
   - 入场: 价格突破过去 20 日最高价 -> 开仓做多
   - 出场: 价格跌破过去 10 日最低价 -> 平仓
   - 为什么入场周期(20)比出场周期(10)长？
     因为要让"入场门槛"更高，确保突破是真信号；
     而"出场门槛"更低，让利润奔跑（截断亏损，让利润奔跑）

2. ATR (Average True Range, 平均真实波幅) - 波动率度量
   - 计算周期: 20 日
   - ATR 衡量市场平均每天的波动幅度
   - ATR 大 -> 市场波动剧烈 -> 减少仓位
   - ATR 小 -> 市场平稳 -> 增加仓位
   - 关键原理: 仓位与波动率成反比，实现"风险恒定"

3. ATR 仓位管理 - 海龟策略的核心创新
   公式: 单位大小 = (账户总值 * 风险比例) / ATR
   举例:
     账户 100 万, ATR = 2 元, risk_pct = 1%
     单位大小 = (1,000,000 * 1%) / 2 = 5,000 股
   含义: 如果价格反向波动 1 个 ATR(2 元)，账户亏损 1 万元 = 账户的 1%
   这保证了: 无论股票是高价股还是低价股，每笔交易承担的风险是恒定的！

4. 金字塔加仓
   - 最多持有 4 个单位
   - 每上涨 0.5 个 ATR，加仓 1 个单位
   - 每次加仓后，止损线上移至 最新入场价 - 2 个 ATR
   - 为什么金字塔加仓？
     初始仓位较小（风险可控），趋势确认后逐步加仓（顺势而为）

====================================================================
本文件结构与运行说明
====================================================================

Part 1: SimpleTurtle vs FullTurtle 对比
  展示 ATR 仓位管理带来的价值差异
  - SimpleTurtle: 只有唐奇安通道信号 + 固定 95% 仓位（对照组）
  - FullTurtle:   唐奇安通道 + ATR 仓位 + 金字塔加仓 + 2N 止损

Part 2: 多标的横向对比
  展示趋势跟踪策略对市场环境的依赖:
  - 下跌趋势(茅台): 频繁假突破
  - 震荡趋势(沪深300): 在趋势不明时表现一般
  - 上涨趋势(纳指ETF): 表现优异

运行: python 1-经典海龟策略.py
"""
import numpy as np
import backtrader as bt
from data_loader import run_and_report, calc_buy_and_hold


# ============================================================
# 简单海龟策略 - 仅信号，固定仓位 (对照组)
# ============================================================
# 这个策略只使用唐奇安通道的突破信号，不进行任何仓位管理。
# 每次信号出现时，使用默认的 PercentSizer 将 95% 资金投入。
# 目的是和完整海龟对比，量化展示"仓位管理"到底贡献了多少价值。

class SimpleTurtleStrategy(bt.Strategy):
    """
    简单版海龟策略（对照组）

    这个版本只有"信号系统"，没有"仓位管理系统"。
    用来回答一个问题: "如果我只用唐奇安通道信号，但满仓操作，效果如何？"

    参数:
        entry_period: 入场通道周期，默认 20（突破 20 日最高价入场）
        exit_period:  出场通道周期，默认 10（跌破 10 日最低价出场）

    结果预期:
      相比完整海龟，简单海龟在趋势行情中收益更高（因为仓位重），
      但在震荡行情中回撤也更大（同样的原因）。
      这印证了海龟法则的核心思想: 仓位管理决定生死。
    """
    params = (
        ('entry_period', 20),  # 唐奇安入场通道周期：突破过去 N 日最高价时入场
        ('exit_period', 10),   # 唐奇安出场通道周期：跌破过去 N 日最低价时出场
    )

    def __init__(self):
        # 计算价格通道:
        #   Highest: 过去 entry_period 个 bar 的最高价
        #   Lowest:  过去 exit_period 个 bar 的最低价
        #   注意: Backtrader 的 Highest/Lowest 默认包含当前 bar，
        #         所以实际取的是 (当前 + 前 N-1 个) 的最高/最低
        self.entry_high = bt.ind.Highest(self.data.high, period=self.p.entry_period)
        self.exit_low = bt.ind.Lowest(self.data.low, period=self.p.exit_period)

    def next(self):
        """
        每个 K 线到来时执行一次

        逻辑非常简单:
          空仓时: 价格突破通道上轨 -> 买入（仓位由 PercentSizer 管理，默认 95%）
          持仓时: 价格跌破通道下轨 -> 全部卖出
        """
        if not self.position:
            # 入场信号: 收盘价突破唐奇安通道上轨
            if self.data.close[0] > self.entry_high[-1]:
                self.buy()
        else:
            # 出场信号: 收盘价跌破唐奇安通道下轨
            if self.data.close[0] < self.exit_low[-1]:
                self.close()  # close() = 平掉所有仓位


# ============================================================
# 完整海龟策略 - ATR 仓位 + 金字塔加仓 + 2N 止损
# ============================================================
# 这是海龟交易法则的完整实现，包含全部四个核心组件。
# 相比简单版，增加了:
#   1. ATR 动态仓位计算（风险恒定）
#   2. 金字塔加仓（最多 4 个单位）
#   3. 移动止损（2N 止损线）
#   4. 双重出场机制（止损 + 通道出场）

class TurtleStrategy(bt.Strategy):
    """
    完整海龟交易策略

    =============================================================
    仓位公式详解
    =============================================================

    单位大小(股) = (账户资金 * risk_pct) / ATR

    这个公式的设计哲学是: 风险预算制

    假设账户 100 万, risk_pct = 1% (即每笔交易愿意亏损 1 万元):
      - 如果股票波动剧烈 (ATR=5 元) -> 只能买 2000 股
        (每下跌 1 个 ATR = 5 元，亏损 2000*5 = 1 万 = 1%)
      - 如果股票波动平稳 (ATR=1 元) -> 可以买 10000 股
        (每下跌 1 个 ATR = 1 元，亏损 10000*1 = 1 万 = 1%)

    无论股票波动率如何，每笔交易承担的风险始终是账户的 1%。

    =============================================================
    加仓规则
    =============================================================

    第一次入场后，如果趋势延续:
      - 价格涨过 入场价 + 0.5 * ATR -> 加 1 个单位
      - 价格再涨 0.5 * ATR         -> 再加 1 个单位
      - ...最多加到 4 个单位

    为什么是 0.5N？这是海龟法则的经验值。
    太小的加仓间隔(0.3N) -> 加仓太频繁，容易在回调时全部止损
    太大的加仓间隔(1N)  -> 加仓太少，趋势来了吃不到足够仓位

    =============================================================
    止损规则
    =============================================================

    每次加仓后，止损线上移至:
      最新入场价 - 2 * ATR

    这意味着:
      - 第一次入场: 止损在入场价下方 2ATR
      - 第二次加仓: 止损上移到第二次入场价下方 2ATR
      - 随着加仓次数增加，止损线不断上移

    效果: 如果趋势持续，止损线不断上移保护利润；
          如果趋势反转，尽早离场控制亏损。

    =============================================================
    出场机制
    =============================================================

    海龟策略有双重出场:
      1. 止损出场: 价格跌破 2N 止损线 -> 无条件平仓
      2. 趋势出场: 价格跌破 10 日低点 -> 趋势可能结束, 平仓

    止损出场是"风控底线"，趋势出场是"跟随趋势"。
    两者谁先触发就按谁执行。
    """
    params = (
        ('entry_period', 20),   # 唐奇安入场通道: 突破过去 20 日最高价入场
        ('exit_period', 10),    # 唐奇安出场通道: 跌破过去 10 日最低价出场
        ('atr_period', 20),     # ATR 计算周期: 取过去 20 日平均真实波幅
        ('risk_pct', 0.01),     # 单笔风险比例: 每笔交易承担 1% 账户净值的风险
                                # 海龟原版建议 2%，这里用 1% 更保守
        ('max_units', 4),       # 最大持仓单位数: 最多加仓到 4 个单位
                                # 海龟原版允许 4 个单位（突破、加仓、再加仓、再加仓）
        ('add_n', 0.5),         # 加仓步长: 每上涨 0.5 个 ATR 加一个单位
                                # 这是"让利润奔跑"的具体实现
        ('stop_n', 2.0),        # 止损倍数: 止损线设在入场价下方 2 个 ATR
                                # 海龟原版使用 2N (2 倍 ATR) 作为止损
    )

    def __init__(self):
        # ---- 信号系统 ----
        self.entry_high = bt.ind.Highest(self.data.high, period=self.p.entry_period)
        self.exit_low = bt.ind.Lowest(self.data.low, period=self.p.exit_period)

        # ---- 波动率度量 ----
        # Backtrader 内置 ATR 指标，使用 Wilder 的平滑方法
        # 和 TA-Lib 的 ATR 计算结果略有差异（平滑系数不同），但趋势一致
        self.atr = bt.ind.ATR(period=self.p.atr_period)

        # ---- 状态变量 ----
        self.units = 0              # 当前持仓单位数 (0~max_units)
        self.entry_prices = []      # 各单位的入场价格列表，用于计算平均持仓成本
                                    # 例: [10.00, 10.50, 11.00] 表示三次入场价格
        self.stop_price = 0.0       # 当前止损线（跌破此价格平仓）
        self.last_add_price = 0.0   # 最近一次加仓价格（用于判断下一次加仓条件）
        self.order = None           # 当前待处理的订单（防止重复下单）

    def notify_order(self, order):
        """
        订单状态变化回调

        当订单成交时，更新持仓状态:
          - 买入成交: 记录入场价，更新单位数，设置止损线
          - 卖出成交: 重置所有状态（全部平仓）

        注意:
          order 从 submitted -> accepted -> completed 会触发三次，
          我们只在 completed 时处理。Submitted/Accepted 直接跳过。
        """
        if order.status in [order.Submitted, order.Accepted]:
            return
        if order.status == order.Completed:
            if order.isbuy():
                # 买入成交：记录入场价格
                fp = order.executed.price         # 实际成交价（可能和下单价格不同）
                self.entry_prices.append(fp)       # 添加到入场价格列表
                self.units = len(self.entry_prices) # 更新持仓单位数
                # 止损线 = 最新入场价 - 2 * ATR
                # 随着加仓，止损线逐步上移（保护利润）
                self.stop_price = fp - self.p.stop_n * self.atr[0]
                self.last_add_price = fp           # 更新最近加仓价
            elif order.issell():
                # 卖出成交（平仓）：重置所有持仓状态
                self.units = 0
                self.entry_prices = []
                self.stop_price = 0.0
                self.last_add_price = 0.0
        self.order = None

    def _calc_unit_size(self):
        """
        海龟仓位公式: 计算一个"单位"的持仓数量

        公式: 单位大小(股) = (账户总值 * 风险比例) / ATR

        数学推导:
          假设:
            - 账户总值 = P
            - 风险比例 = r (如 1%)
            - ATR = N
            - 单位大小 = S（股）
          等式:
            每波动 1ATR 的盈亏 = S * N
            我们希望每笔风险 = P * r
          所以: S * N = P * r
                -> S = (P * r) / N

        A 股规则:
          交易单位必须是 100 股（1 手）的整数倍
          最少买入 100 股（1 手）
          所以最后做取整处理: int(S // 100) * 100

        返回:
            int 单位大小（股数），至少 100 股
        """
        portfolio_value = self.broker.getvalue()  # 当前账户总净值
        atr_val = self.atr[0]                     # 当前 ATR 值
        if atr_val <= 0:
            return 0  # ATR=0 说明数据异常，不交易
        # 核心仓位公式
        unit_size = (portfolio_value * self.p.risk_pct) / atr_val
        # 取整到 100 股（A 股最小交易单位）
        unit_size = int(unit_size // 100) * 100
        return max(unit_size, 100)

    def next(self):
        """
        每个交易日的核心逻辑

        执行顺序:
          1. 检查是否有未完成订单（有则跳过，不重复下单）
          2. 检查 ATR 有效性（无效则跳过）
          3. 空仓状态: 检查入场信号（突破通道 + 计算仓位 -> 买入）
          4. 持仓状态:
             a. 检查止损线（跌破止损 -> 平仓）
             b. 检查出场通道（跌破通道 -> 平仓）
             c. 检查加仓条件（满足条件 -> 加仓）
        """
        # 防重复下单: 如果还有未完成的订单，不做任何操作
        if self.order:
            return

        # 数据有效性检查: ATR 必须是有效正数
        atr_val = self.atr[0]
        if np.isnan(atr_val) or atr_val <= 0:
            return

        close = self.data.close[0]

        # ---- 空仓状态: 寻找入场机会 ----
        if not self.position:
            # 突破信号: 收盘价 > 唐奇安通道上轨
            # entry_high[-1] 取前一个 bar 的值，避免 look-ahead bias (未来数据)
            # 如果取 entry_high[0]，则包含了当前 bar 的最高价，会造成未来数据泄露
            if close > self.entry_high[-1]:
                size = self._calc_unit_size()
                if size > 0:
                    # 使用 buy() 创建买入订单
                    self.order = self.buy(size=size)

        # ---- 持仓状态: 止损/出场/加仓 ----
        else:
            # 1. 止损出场: 价格跌破 2N 止损线
            #    这是最严格的底线，必须无条件执行
            if close < self.stop_price:
                self.order = self.close()
                return

            # 2. 趋势出场: 价格跌破 10 日低点
            #    意味着当前的上升趋势可能已经结束
            if close < self.exit_low[-1]:
                self.order = self.close()
                return

            # 3. 金字塔加仓: 趋势延续，逐步增加仓位
            if self.units < self.p.max_units:
                # 加仓条件: 价格 >= 最近加仓价 + 0.5 * ATR
                if close >= self.last_add_price + self.p.add_n * atr_val:
                    size = self._calc_unit_size()
                    cash = self.broker.getcash()
                    # 检查可用现金是否足够（预留 1% 应对滑点）
                    # 避免因现金不足导致加仓失败
                    if size > 0 and cash > close * size * 1.01:
                        self.order = self.buy(size=size)


# ============================================================
# 主程序
# ============================================================

def _print_three_strategy_table(bh_ret, r_simple, r_full):
    """
    打印三策略对比表格: 买入持有 vs 简单海龟 vs 完整海龟

    表格包含:
      - 总收益: 区间总收益率
      - 最大回撤: 衡量风险
      - 夏普比率: 风险调整后收益
      - 交易次数: 策略活跃度
      - 胜率: 盈交易占比（海龟通常 30-40%）
      - 盈亏比: 平均盈利/平均亏损（海龟通常 > 2）

    参数:
        bh_ret: float, 买入持有收益率
        r_simple: dict, 简单海龟回测结果
        r_full: dict, 完整海龟回测结果
    """
    bh_val = (bh_ret or 0) * 100
    print(f"  {'指标':<12} {'买入持有':>12} {'简单海龟':>12} {'完整海龟':>12}")
    print(f"  {'-' * 52}")
    print(f"  {'总收益':<12} {bh_val:>+11.2f}% {r_simple['total_return']*100:>+11.2f}% {r_full['total_return']*100:>+11.2f}%")
    print(f"  {'最大回撤':<12} {'--':>12} {r_simple['max_drawdown']*100:>11.2f}% {r_full['max_drawdown']*100:>11.2f}%")
    print(f"  {'夏普比率':<12} {'--':>12} {r_simple['sharpe_ratio']:>12.2f} {r_full['sharpe_ratio']:>12.2f}")
    print(f"  {'交易次数':<12} {'--':>12} {r_simple['total_trades']:>12d} {r_full['total_trades']:>12d}")
    print(f"  {'胜率':<12} {'--':>12} {r_simple['win_rate']*100:>11.1f}% {r_full['win_rate']*100:>11.1f}%")
    print(f"  {'盈亏比':<12} {'--':>12} {r_simple['profit_loss_ratio']:>12.2f} {r_full['profit_loss_ratio']:>12.2f}")


if __name__ == '__main__':
    # ================================================================
    # 全局参数设置
    # ================================================================
    start_date = '2024-01-01'
    end_date = '2025-12-31'

    print("=" * 70)
    print("海龟交易法则 - 经典策略实战")
    print("=" * 70)
    print("\n海龟四大组件:")
    print("  1. 唐奇安通道: 突破20日最高价入场, 跌破10日最低价出场")
    print("  2. ATR(N值):   衡量市场波动幅度")
    print("  3. ATR仓位:    单位大小 = (账户资金 * 1%) / ATR")
    print("  4. 金字塔加仓: 最多4个单位, 每上涨0.5N加仓, 2N止损")

    # ================================================================
    # Part 1: 三策略对比 - 买入持有 vs 简单海龟 vs 完整海龟
    # ================================================================
    # 先用沪深 300 ETF (510300.SH) 做演示
    # 选择理由: ETF 流动性好、没有涨跌停限制带来的极端波动、代表市场整体走势
    demo_stock = '510300.SH'
    print(f"\n{'=' * 70}")
    print(f"Part 1: 买入持有 vs 简单海龟 vs 完整海龟 ({demo_stock} 沪深300ETF)")
    print(f"{'=' * 70}")

    # 简单海龟: 只有信号，固定仓位
    print(f"\n[简单海龟] 仅唐奇安通道信号, 固定仓位95%:")
    r_simple = run_and_report(
        SimpleTurtleStrategy, demo_stock, start_date, end_date,
        label='简单海龟', plot=True, use_sizer=True,
        # use_sizer=True: 使用默认 PercentSizer，每次投入 95% 资金
    )

    # 完整海龟: 信号 + ATR 仓位 + 金字塔加仓 + 2N 止损
    print(f"\n[完整海龟] ATR仓位管理 + 金字塔加仓 + 2N止损:")
    r_full = run_and_report(
        TurtleStrategy, demo_stock, start_date, end_date,
        label='完整海龟', plot=True, use_sizer=False,
        # use_sizer=False: 海龟自行管理仓位，不需要默认 Sizer
    )

    # 买入持有基准
    bh = calc_buy_and_hold(demo_stock, start_date, end_date)

    # 三策略对比
    print(f"\n{'  三策略对比 ':=^60}")
    _print_three_strategy_table(bh, r_simple, r_full)

    # ================================================================
    # Part 2: 多标的横向对比
    # ================================================================
    # 为什么需要对比不同标的？
    # 海龟策略是趋势跟踪策略，它的表现高度依赖于市场环境:
    #   - 强趋势市场（如 2024-2025 的纳指）-> 大赚
    #   - 震荡市场（如沪深 300 的区间波动）-> 小亏
    #   - 下跌市场（如茅台的持续下跌）-> 频繁假突破，持续亏损
    #
    # 通过多标的对比，我们能清楚看到: 没有万能的策略，选择合适的战场很重要。
    print(f"\n{'=' * 70}")
    print("Part 2: 三策略在不同标的上的表现")
    print(f"{'=' * 70}")
    print("  趋势跟踪策略的核心前提: 市场存在趋势")
    print("  下面对比: 下跌股 vs 上涨ETF, 看三种策略的差异\n")

    stocks = [
        ('600519.SH', '贵州茅台'),     # 下跌趋势股：检验策略在逆境中的表现
        ('510300.SH', '沪深300ETF'),   # 震荡标的：检验策略在平庸行情中的表现
        ('159941.SZ', '纳指ETF'),      # 上涨趋势ETF：检验策略在顺境中的表现
    ]

    results = {}
    for code, name in stocks:
        try:
            print(f"\n--- {name}({code}) ---")
            r_s = run_and_report(
                SimpleTurtleStrategy, code, start_date, end_date,
                label=f'{name}-简单海龟', plot=True, use_sizer=True,
            )
            r_f = run_and_report(
                TurtleStrategy, code, start_date, end_date,
                label=f'{name}-完整海龟', plot=True, use_sizer=False,
            )
            bh_ret = calc_buy_and_hold(code, start_date, end_date)
            results[code] = {
                'name': name, 'simple': r_s, 'full': r_f, 'bh': bh_ret,
            }
        except ValueError as e:
            print(f"  {name}({code}): 跳过 - {e}")

    # ---- 每个标的的三策略对比 ----
    for code, data in results.items():
        print(f"\n{'=' * 70}")
        print(f"{data['name']}({code}) 三策略对比")
        print(f"{'=' * 70}")
        _print_three_strategy_table(data['bh'], data['simple'], data['full'])

    # ---- 多标的汇总表: 按策略分组 ----
    # 这样方便横向比较: 同一个策略在不同标的上的表现差异
    if results:
        print(f"\n{'=' * 70}")
        print("多标的汇总对比")
        print(f"{'=' * 70}")

        names = [data['name'] for data in results.values()]
        col_width = 12
        header = f"  {'指标':<16}"
        for n in names:
            header += f" {n:>{col_width}}"
        sep_len = 16 + (col_width + 1) * len(names)

        for strategy_label, key in [('买入持有', 'bh'), ('简单海龟', 'simple'), ('完整海龟', 'full')]:
            print(f"\n  [{strategy_label}]")
            print(f"  {header.strip()}")
            print(f"  {'-' * sep_len}")

            if key == 'bh':
                # 买入持有只有收益率一项指标
                row = f"  {'总收益':<16}"
                for data in results.values():
                    bh_val = (data['bh'] or 0) * 100
                    row += f" {bh_val:>+{col_width-1}.1f}%"
                print(row)
            else:
                # 策略有完整的绩效指标
                rows_cfg = [
                    ('总收益',   lambda r: f"{r['total_return']*100:>+{col_width-1}.1f}%"),
                    ('最大回撤', lambda r: f"{r['max_drawdown']*100:>{col_width-1}.1f}%"),
                    ('夏普比率', lambda r: f"{r['sharpe_ratio']:>{col_width}.2f}"),
                    ('交易次数', lambda r: f"{r['total_trades']:>{col_width}d}"),
                    ('胜率',     lambda r: f"{r['win_rate']*100:>{col_width-1}.1f}%"),
                    ('盈亏比',   lambda r: f"{r['profit_loss_ratio']:>{col_width}.2f}"),
                ]
                for label, fmt_fn in rows_cfg:
                    row = f"  {label:<16}"
                    for data in results.values():
                        row += f" {fmt_fn(data[key])}"
                    print(row)

    # ================================================================
    # 实验结论
    # ================================================================
    print("\n关键发现:")
    print("  - 海龟策略在有明确趋势的市场中表现优异 (纳指ETF 夏普最高)")
    print("  - 在下跌/震荡市场中(茅台), 趋势跟踪策略会频繁假突破而亏损")
    print("  - 简单海龟(95%仓位)收益更高, 但回撤也更大 -- 高仓位是双刃剑")
    print("  - 完整海龟通过ATR仓位管理控制风险, 回撤更小, 但牺牲了收益弹性")
    print("  - 核心启示: 趋势策略不是万能的, 选择合适的市场比优化参数更重要")
