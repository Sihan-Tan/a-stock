# -*- coding: utf-8 -*-
"""
经典网格交易策略 - 固定网格 + ETF入门
====================================

核心理念:
---------
海龟和缠论教我们如何在趋势市中赚钱, 但市场 70% 以上的时间在震荡。
网格策略就是震荡市的"收租"模型:

  把价格区间切成 N 个格子
  每跌一格买一份 (越跌越买, 摊低成本)
  每涨一格卖一份 (越涨越卖, 落袋为安)
  不预测涨跌, 只要价格在区间内波动就赚差价

网格四要素:
----------
1. 网格区间 [lower, upper]: 在哪里做网格?
   - 本脚本用过去 N 日 (60日) 的最高/最低价 + 2% 缓冲

2. 格子数量 (num_grids): 切多细?
   - 越多: 交易越频繁, 每笔利润越小, 但胜率更高
   - 本脚本用 8 格

3. 每格仓位 (capital_per_grid): 每格买多少?
   - 总资金 / 格子数 = 每格分配资金
   - 本脚本: 90% 总资金 / 8 = 约 11.25% / 格

4. 出界处理: 价格涨破上界或跌破下界怎么办?
   - 突破上界: 已全部卖出, 空仓等待价格回归
   - 跌破下界: 已满仓被套, 只能被动持有

本案例:
-------
  Part 1: 在沪深300ETF上运行固定网格 (展示网格原理)
  Part 2: 多标的横向对比 (网格 vs 买入持有)
  Part 3: 多标的对比表格 (展示网格策略对市场环境的依赖)

运行方式:
  python 1-经典网格策略.py
"""
import numpy as np
import backtrader as bt
from data_loader import load_stock_data, run_and_report, calc_buy_and_hold
from grid_engine import GridEngine
from db_config import INITIAL_CASH


# ============================================================
# 固定网格策略
# ============================================================

class SimpleGridStrategy(bt.Strategy):
    """
    固定网格策略 - Backtrader 策略类。

    策略逻辑:
      1. 积累 lookback 根 K 线后, 用历史最高/最低价确定网格区间
      2. 创建 GridEngine, 在区间内做网格交易
      3. 每日调用 grid.update(price) 检查是否有交易信号
      4. 根据信号执行买入/卖出

    参数:
      lookback: 回看周期, 用于计算网格区间 (默认 60 日)
      num_grids: 格子数量 (默认 8 格)
      margin_pct: 上下缓冲比例 (默认 2%), 防止区间刚好在边界上
    """
    params = (
        ('lookback', 60),      # 回看周期: 用过去60天的数据确定网格区间
        ('num_grids', 8),      # 格子数量: 将区间切成8格
        ('margin_pct', 0.02),  # 缓冲比例: 上下各留2%的缓冲空间
    )

    def __init__(self):
        self.grid = None             # GridEngine 实例
        self.order = None            # 当前挂单 (防止重复下单)
        self.grid_initialized = False  # 网格是否已初始化

    def notify_order(self, order):
        """
        订单状态回调: 订单完成时重置 self.order。
        Submitted/Accepted 是中间状态, 忽略。
        """
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

    def _init_grid(self):
        """
        用历史数据初始化网格区间。

        步骤:
          1. 获取过去 lookback 日的最高价和最低价
          2. 区间上下界 = 极值 ± 缓冲
          3. 分配 90% 的资金给网格 (留10%防意外)
          4. 创建 GridEngine 实例
        """
        size = min(len(self.data), self.p.lookback)
        # 获取历史最高/最低 (从近到远遍历)
        highs = [self.data.high[-i] for i in range(1, size + 1)]
        lows = [self.data.low[-i] for i in range(1, size + 1)]

        hist_high = max(highs)
        hist_low = min(lows)
        margin = (hist_high - hist_low) * self.p.margin_pct
        upper = hist_high + margin
        lower = hist_low - margin

        # 分配资金的90%给网格, 留10%应对滑点和意外
        grid_capital = self.broker.getvalue() * 0.90
        self.grid = GridEngine(
            upper=upper,
            lower=lower,
            num_grids=self.p.num_grids,
            total_capital=grid_capital,
        )
        self.grid_initialized = True

    def next(self):
        """
        每个 Bar 的回调: 核心交易逻辑。

        流程:
          1. 如果有挂单, 等待成交
          2. 如果数据不够 lookback 天, 继续积累
          3. 如果网格未初始化, 初始化
          4. 获取网格交易信号, 执行买卖
        """
        if self.order:
            return

        if len(self.data) < self.p.lookback:
            return

        if not self.grid_initialized:
            self._init_grid()
            return

        price = self.data.close[0]
        signals = self.grid.update(price)

        for sig in signals:
            if sig['action'] == 'BUY':
                # 买入: 检查现金是否足够 (预留1%防滑点)
                cash = self.broker.getcash()
                cost = price * sig['size'] * 1.01
                if cash >= cost and sig['size'] > 0:
                    self.order = self.buy(size=sig['size'])
            elif sig['action'] == 'SELL':
                # 卖出: 检查持仓是否足够
                if self.position.size >= sig['size']:
                    self.order = self.sell(size=sig['size'])

    def stop(self):
        """
        回测结束时调用: 打印网格运行报告。
        """
        if self.grid:
            self.grid.summary()


# ============================================================
# 主程序
# ============================================================

def _print_comparison_table(results, bh_results, stocks):
    """
    打印多标的对比表格, 横向对比网格策略和买入持有。

    参数:
        results: dict {code: run_and_report 结果}
        bh_results: dict {code: 买入持有收益率}
        stocks: list of (code, name)
    """
    names = [name for _, name in stocks]
    col_width = 14
    header = f"  {'指标':<12}"
    for n in names:
        header += f" {n:>{col_width}}"
    sep_len = 12 + (col_width + 1) * len(names)

    # 买入持有
    print(f"\n  [买入持有]")
    print(f"  {header.strip()}")
    print(f"  {'-' * sep_len}")
    row = f"  {'总收益':<12}"
    for code, _ in stocks:
        bh = bh_results.get(code)
        val = (bh or 0) * 100
        row += f" {val:>+{col_width-1}.1f}%"
    print(row)

    # 网格策略
    print(f"\n  [网格策略]")
    print(f"  {header.strip()}")
    print(f"  {'-' * sep_len}")

    rows_cfg = [
        ('总收益',   lambda r: f"{r['total_return']*100:>+{col_width-1}.1f}%"),
        ('最大回撤', lambda r: f"{r['max_drawdown']*100:>{col_width-1}.1f}%"),
        ('夏普比率', lambda r: f"{r['sharpe_ratio']:>{col_width}.2f}"),
        ('交易次数', lambda r: f"{r['total_trades']:>{col_width}d}"),
        ('胜率',     lambda r: f"{r['win_rate']*100:>{col_width-1}.1f}%"),
        ('盈亏比',   lambda r: f"{r['profit_loss_ratio']:>{col_width}.2f}"),
    ]
    for label, fmt_fn in rows_cfg:
        row = f"  {label:<12}"
        for code, _ in stocks:
            r = results.get(code)
            if r:
                row += f" {fmt_fn(r)}"
            else:
                row += f" {'--':>{col_width}}"
        print(row)


if __name__ == '__main__':
    # 回测时间区间: 2024-01-01 ~ 2025-12-31 (2年)
    start_date = '2024-01-01'
    end_date = '2025-12-31'

    print("=" * 70)
    print("网格交易策略 - 经典固定网格")
    print("=" * 70)
    print("\n网格四要素:")
    print("  1. 网格区间: 过去60日最高/最低价 (+2%缓冲)")
    print("  2. 格子数量: 8格")
    print("  3. 每格仓位: 总资金的90% / 8 = 每格约11.25%")
    print("  4. 出界处理: 超出范围时停止买入/卖出, 等待价格回归")

    # ================================================================
    # Part 1: 沪深300ETF 网格演示
    # ================================================================
    demo_stock = '510300.SH'  # 沪深300ETF, 典型的震荡标的
    print(f"\n{'=' * 70}")
    print(f"Part 1: 固定网格演示 ({demo_stock} 沪深300ETF)")
    print(f"{'=' * 70}")

    print(f"\n[网格策略]")
    r_grid = run_and_report(
        SimpleGridStrategy, demo_stock, start_date, end_date,
        label='固定网格', plot=True, use_sizer=False,
    )

    bh = calc_buy_and_hold(demo_stock, start_date, end_date)
    print(f"\n  买入持有: {(bh or 0)*100:+.1f}%")
    print(f"  网格策略: {r_grid['total_return']*100:+.1f}%")

    # ================================================================
    # Part 2: 多标的横向对比
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Part 2: 多标的横向对比")
    print(f"{'=' * 70}")
    print("  网格策略的核心前提: 市场在区间内震荡")
    print("  下面对比不同标的的网格策略表现\n")

    stocks = [
        ('510300.SH', '沪深300ETF'),   # 大盘指数ETF, 震荡偏多
        ('600519.SH', '贵州茅台'),      # 高价蓝筹, 趋势性强
        ('000001.SZ', '平安银行'),      # 银行股, 低波动震荡
        ('159941.SZ', '纳指ETF'),      # 美股ETF, 长期上涨趋势
    ]

    results = {}
    bh_results = {}

    for code, name in stocks:
        try:
            print(f"\n--- {name}({code}) ---")
            r = run_and_report(
                SimpleGridStrategy, code, start_date, end_date,
                label=f'{name}-网格', plot=True, use_sizer=False,
            )
            results[code] = r
            bh_results[code] = calc_buy_and_hold(code, start_date, end_date)
        except ValueError as e:
            print(f"  {name}({code}): 跳过 - {e}")

    if results:
        print(f"\n{'=' * 70}")
        print("多标的汇总对比")
        print(f"{'=' * 70}")
        _print_comparison_table(results, bh_results, stocks)

    print("\n关键发现:")
    print("  - 网格策略胜率极高(70-80%), 但盈亏比低(每次赚固定一格)")
    print("  - 在震荡市(如沪深300ETF)中, 网格策略稳定赚差价")
    print("  - 在趋势市(如纳指ETF)中, 网格容易踏空或深套")
    print("  - 核心风险: 价格单边下跌时持续加仓 → 满仓被套")
    print("  - 下一步: 用缠论中枢替代固定区间, 让网格边界更科学")
