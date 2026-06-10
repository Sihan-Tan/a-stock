# -*- coding: utf-8 -*-
"""
缠论中枢网格策略 - 用缠论中枢的 ZG/ZD 作为网格边界
================================================

核心创新:
---------
脚本1的固定网格用"过去60日最高/最低价"作为网格区间 -- 这是人为拍脑袋的。
本脚本用缠论中枢的 ZG/ZD 作为网格区间 -- 这是市场自己"选出来"的震荡区间。

为什么中枢适合做网格?
  缠论中枢的定义: 至少3笔价格区间有重叠 → 价格在此区间内反复震荡。
  这不就是网格策略最理想的运行环境吗? 在中枢内低买高卖,
  价格突破中枢时停止交易, 等待新中枢形成。

策略逻辑:
---------
1. 用 chan.py 分析K线, 识别所有中枢 (价格震荡区间)
2. 当价格处于某个中枢内:
   - 以 ZG (中枢上沿) 为网格上界
   - 以 ZD (中枢下沿) 为网格下界
   - 在中枢内等距切格, 执行网格交易
3. 当价格突破中枢 (出中枢):
   - 停止网格, 清空网格持仓
   - 等待价格回到中枢或新中枢形成
4. 新中枢形成:
   - 用新的 ZG/ZD 自动重建网格

本案例:
-------
  Part 1: 单标的中枢网格演示 (贵州茅台)
  Part 2: 中枢网格 vs 固定网格 vs 买入持有 (三策略对比)
  Part 3: 多标的横向对比

运行方式:
  python 2-缠论中枢网格策略.py
"""
import numpy as np
import backtrader as bt
from data_loader import (
    load_stock_data, ChanPandasData,
    run_and_report, calc_buy_and_hold,
)
from chanpy_wrapper import run_chan, chan_to_signal_df, draw_chan_chart
from grid_engine import ChanGridEngine, GridEngine
from db_config import INITIAL_CASH


# ============================================================
# 缠论中枢网格策略
# ============================================================

class ChanGridStrategy(bt.Strategy):
    """
    缠论中枢网格策略。

    利用 ChanPandasData 中预计算的 chan_zg/chan_zd 信号线:
      - 当 chan_zg > 0 且 chan_zd > 0: 有中枢, 可以做网格
      - 价格在 [chan_zd, chan_zg] 内: 网格模式, 低买高卖
      - 价格突破 chan_zg 或跌破 chan_zd: 停止网格, 清仓等待

    中枢切换:
      当 chan_zg/chan_zd 值发生变化时, 说明进入了新的中枢,
      自动重建网格参数。ZG/ZD 变化超过 0.1% 视为新中枢。

    三种模式:
      WAIT:    等待中枢形成
      GRID:    中枢内做网格
      BREAKOUT:价格突破中枢, 停止交易

    参数:
      num_grids: 中枢内格子数 (默认6格, 中枢区间比固定网格小)
      capital_ratio: 分配给网格的资金比例 (默认80%)
      exit_on_breakout: 出中枢时是否清仓 (默认True)
    """
    params = (
        ('num_grids', 6),           # 中枢内格子数, 比固定网格少
        ('capital_ratio', 0.80),    # 80%资金分配给网格
        ('exit_on_breakout', True), # 出中枢时清仓
    )

    def __init__(self):
        self.grid = None           # ChanGridEngine 实例
        self.order = None          # 当前挂单
        self.current_zg = 0.0      # 当前中枢上沿
        self.current_zd = 0.0      # 当前中枢下沿
        self.mode = 'WAIT'         # WAIT / GRID / BREAKOUT
        self.mode_changes = []     # 模式切换记录

    def notify_order(self, order):
        """订单状态回调: 订单完成后重置挂单标志"""
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

    def _build_grid(self, zg, zd):
        """
        用新的中枢参数构建 (或重建) 网格。

        如果 grid 不存在, 创建新实例;
        如果 grid 已存在 (中枢切换), 调用 switch_zhongshu() 重建。

        参数:
            zg: 中枢上沿 (网格上界)
            zd: 中枢下沿 (网格下界)
        """
        if zg <= zd or zg <= 0:
            return
        grid_capital = self.broker.getvalue() * self.p.capital_ratio
        if self.grid is None:
            self.grid = ChanGridEngine(
                zg=zg, zd=zd,
                num_grids=self.p.num_grids,
                total_capital=grid_capital,
            )
        else:
            # 中枢切换: 重建网格
            self.grid.switch_zhongshu(zg, zd)
            self.grid.total_capital = grid_capital
            self.grid.capital_per_grid = grid_capital / self.p.num_grids

        self.current_zg = zg
        self.current_zd = zd
        self.mode = 'GRID'

    def _close_all_grid_positions(self):
        """平掉所有网格持仓, 停用网格。"""
        if self.position.size > 0:
            self.order = self.close()  # 市价全平
        if self.grid:
            self.grid.deactivate()
            # 重置持仓记录, 等待新中枢
            self.grid.position_at = [0] * (self.grid.num_grids + 1)

    def next(self):
        """
        每个 Bar 的回调: 核心交易逻辑。

        流程:
          1. 获取当前中枢 ZG/ZD 和价格
          2. 如果没有中枢 → 停止网格, 进入 WAIT 模式
          3. 如果中枢切换 → 平仓 → 重建网格
          4. 如果出中枢 → 平仓, 进入 BREAKOUT
          5. 在中枢内 → 执行网格交易
        """
        if self.order:
            return

        zg = self.data.chan_zg[0]    # 当前Bar的中枢上沿
        zd = self.data.chan_zd[0]    # 当前Bar的中枢下沿
        price = self.data.close[0]

        # 检查中枢是否有效: ZG>0, ZD>0, 且不是 NaN
        has_zhongshu = (not np.isnan(zg)) and (not np.isnan(zd)) and zg > 0 and zd > 0

        if not has_zhongshu:
            # 没有中枢 → 如果之前在做网格, 平仓退出
            if self.mode == 'GRID':
                self._close_all_grid_positions()
                self.mode = 'WAIT'
            return

        # 检查中枢是否切换: ZG/ZD 变化超过 0.1% 视为新中枢
        zg_changed = abs(zg - self.current_zg) > self.current_zg * 0.001 if self.current_zg > 0 else True
        zd_changed = abs(zd - self.current_zd) > self.current_zd * 0.001 if self.current_zd > 0 else True

        if zg_changed or zd_changed:
            # 中枢切换: 先平仓, 再重建网格
            if self.mode == 'GRID' and self.position.size > 0:
                self._close_all_grid_positions()
                return    # 等下一个Bar重建, 避免同Bar平仓又建仓
            self._build_grid(zg, zd)

        if self.mode != 'GRID' or self.grid is None:
            return

        # ------- 检查是否出中枢 -------
        # 向上突破 ZG (+0.5%确认): 趋势向上, 停止网格
        if price > zg * 1.005:
            if self.p.exit_on_breakout:
                self._close_all_grid_positions()
            self.mode = 'BREAKOUT'
            return

        # 向下跌破 ZD (-0.5%确认): 趋势向下, 停止网格
        if price < zd * 0.995:
            if self.p.exit_on_breakout:
                self._close_all_grid_positions()
            self.mode = 'BREAKOUT'
            return

        # ------- 在中枢内: 执行网格交易 -------
        signals = self.grid.update(price)

        for sig in signals:
            if sig['action'] == 'BUY':
                cash = self.broker.getcash()
                cost = price * sig['size'] * 1.01
                if cash >= cost and sig['size'] > 0:
                    self.order = self.buy(size=sig['size'])
                    break  # 一次只下一个单
            elif sig['action'] == 'SELL':
                if self.position.size >= sig['size']:
                    self.order = self.sell(size=sig['size'])
                    break

    def stop(self):
        """回测结束时打印网格运行报告"""
        if self.grid:
            self.grid.summary()


# ============================================================
# 固定网格策略 (对照组, 从脚本1简化)
# ============================================================

class FixedGridStrategy(bt.Strategy):
    """
    固定网格策略(对照组), 用于和缠论中枢网格对比。

    逻辑与脚本1的 SimpleGridStrategy 相同, 只是简化了代码。
    """
    params = (
        ('lookback', 60),
        ('num_grids', 8),
        ('margin_pct', 0.02),
    )

    def __init__(self):
        self.grid = None
        self.order = None
        self.grid_initialized = False

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        self.order = None

    def _init_grid(self):
        size = min(len(self.data), self.p.lookback)
        highs = [self.data.high[-i] for i in range(1, size + 1)]
        lows = [self.data.low[-i] for i in range(1, size + 1)]
        hist_high = max(highs)
        hist_low = min(lows)
        margin = (hist_high - hist_low) * self.p.margin_pct
        grid_capital = self.broker.getvalue() * 0.90
        self.grid = GridEngine(
            upper=hist_high + margin, lower=hist_low - margin,
            num_grids=self.p.num_grids, total_capital=grid_capital,
        )
        self.grid_initialized = True

    def next(self):
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
                cash = self.broker.getcash()
                if cash >= price * sig['size'] * 1.01 and sig['size'] > 0:
                    self.order = self.buy(size=sig['size'])
                    break
            elif sig['action'] == 'SELL':
                if self.position.size >= sig['size']:
                    self.order = self.sell(size=sig['size'])
                    break

    def stop(self):
        if self.grid:
            self.grid.summary()


# ============================================================
# 缠论可视化 (中枢 + 网格线)
# ============================================================

def plot_chan_with_grid(df, chan_data, stock_code, title=''):
    """
    绘制缠论分析图 + 中枢内的网格线。

    上图: 缠论结构 (笔/中枢/买卖点) + 中枢内的网格水平线
    下图: 成交量柱状图

    参数:
        df: K线 DataFrame
        chan_data: run_chan() 的结果
        stock_code: 股票代码
        title: 图表标题
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    import os

    os.makedirs('outputs', exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12),
                                    gridspec_kw={'height_ratios': [4, 1]})

    # 在每个中枢内计算网格线: ZG~ZD 之间等距切格
    all_grid_levels = []
    num_grids = 6  # 与策略参数保持一致
    for zs in chan_data['zs_list']:
        zg, zd = zs['ZG'], zs['ZD']
        gs = (zg - zd) / num_grids
        for i in range(num_grids + 1):
            all_grid_levels.append(zd + i * gs)

    draw_chan_chart(ax1, df, chan_data,
                   show_bi=True, show_seg=False, show_zs=True, show_bsp=True,
                   show_grid_levels=all_grid_levels)

    ax1.set_title(f'{title}  {stock_code} (中枢内网格线)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格')
    ax1.grid(True, alpha=0.3)

    # 下图: 成交量柱 (红色=阳线, 绿色=阴线)
    n = len(df)
    vol_colors = ['#e74c3c' if df['close'].iloc[i] >= df['open'].iloc[i] else '#27ae60'
                  for i in range(n)]
    ax2.bar(range(n), df['volume'], color=vol_colors, alpha=0.6, width=0.8)
    step = max(1, n // 12)
    tick_pos = list(range(0, n, step))
    if (n - 1) not in tick_pos:
        tick_pos.append(n - 1)
    total_days = (df.index[-1] - df.index[0]).days if n > 1 else 365
    if total_days <= 180:
        tick_lbl = [df.index[i].strftime('%m-%d') for i in tick_pos]
    else:
        tick_lbl = [df.index[i].strftime('%Y-%m') for i in tick_pos]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=8)
    ax2.set_xlim(-1, n)
    ax2.set_ylabel('成交量')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = title.replace(' ', '_').replace('/', '_')
    plot_file = f'outputs/{safe_name}.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {plot_file}")
    plt.close()


# ============================================================
# 主程序
# ============================================================

def _print_three_strategy_table(bh_ret, r_fixed, r_chan):
    """
    打印三策略对比表格: 买入持有 vs 固定网格 vs 中枢网格。

    参数:
        bh_ret: 买入持有收益率 (float)
        r_fixed: 固定网格回测结果 (dict)
        r_chan: 中枢网格回测结果 (dict)
    """
    bh_val = (bh_ret or 0) * 100
    print(f"  {'指标':<12} {'买入持有':>12} {'固定网格':>12} {'中枢网格':>12}")
    print(f"  {'-' * 52}")
    print(f"  {'总收益':<12} {bh_val:>+11.2f}% {r_fixed['total_return']*100:>+11.2f}% {r_chan['total_return']*100:>+11.2f}%")
    print(f"  {'最大回撤':<12} {'--':>12} {r_fixed['max_drawdown']*100:>11.2f}% {r_chan['max_drawdown']*100:>11.2f}%")
    print(f"  {'夏普比率':<12} {'--':>12} {r_fixed['sharpe_ratio']:>12.2f} {r_chan['sharpe_ratio']:>12.2f}")
    print(f"  {'交易次数':<12} {'--':>12} {r_fixed['total_trades']:>12d} {r_chan['total_trades']:>12d}")
    print(f"  {'胜率':<12} {'--':>12} {r_fixed['win_rate']*100:>11.1f}% {r_chan['win_rate']*100:>11.1f}%")
    print(f"  {'盈亏比':<12} {'--':>12} {r_fixed['profit_loss_ratio']:>12.2f} {r_chan['profit_loss_ratio']:>12.2f}")


if __name__ == '__main__':
    start_date = '2023-01-01'
    end_date = '2025-12-31'

    print("=" * 70)
    print("缠论中枢网格策略")
    print("=" * 70)
    print("\n核心思路:")
    print("  缠论中枢 = 市场自己选出来的震荡区间")
    print("  中枢的ZG/ZD = 网格的天然上下界")
    print("  在中枢内做网格收租, 出中枢后停止等待")
    print("  vs 固定网格: 不再拍脑袋定区间, 让市场结构说话")

    # ================================================================
    # Part 1: 单标的演示 - 贵州茅台
    # ================================================================
    demo_stock = '600519.SH'
    demo_name = '贵州茅台'

    print(f"\n{'=' * 70}")
    print(f"Part 1: 缠论中枢网格演示 ({demo_name})")
    print(f"{'=' * 70}")

    # 加载数据 & 运行缠论分析
    print(f"\n[1] 加载 {demo_stock} 并运行 chan.py 缠论分析...")
    df = load_stock_data(demo_stock, start_date, end_date)
    print(f"    {len(df)} 根K线")

    chan_data = run_chan(df, symbol=demo_stock)
    print(f"    笔: {len(chan_data['bi_list'])} | "
          f"中枢: {len(chan_data['zs_list'])} | "
          f"买卖点: {len(chan_data['bsp_list'])}")

    # 展示中枢列表
    if chan_data['zs_list']:
        print(f"\n    中枢列表:")
        for i, zs in enumerate(chan_data['zs_list']):
            sd = zs['start_date'].strftime('%Y-%m-%d') if zs['start_date'] else '?'
            ed = zs['end_date'].strftime('%Y-%m-%d') if zs['end_date'] else '?'
            width = zs['ZG'] - zs['ZD']
            print(f"      [{i+1}] {sd} ~ {ed} | "
                  f"ZG={zs['ZG']:.2f} ZD={zs['ZD']:.2f} | "
                  f"宽度={width:.2f} ({width/zs['center']*100:.1f}%)")

    # 绘制缠论结构 + 网格线
    print(f"\n[2] 绘制缠论结构 + 中枢内网格线...")
    plot_chan_with_grid(df, chan_data, demo_stock, title=f'缠论中枢网格-{demo_name}')

    # 转换为 ChanPandasData 格式 (含 chan_signal/chan_zg/chan_zd 列)
    signal_df = chan_to_signal_df(df, chan_data)

    # 回测: 中枢网格策略
    print(f"\n[3] 回测: 缠论中枢网格策略")
    r_chan = run_and_report(
        ChanGridStrategy, stock_code=demo_stock,
        label='中枢网格', plot=True, use_sizer=False,
        df=signal_df, data_class=ChanPandasData,
    )

    # 回测: 固定网格策略 (对照组)
    print(f"\n[4] 回测: 固定网格策略 (对照组)")
    r_fixed = run_and_report(
        FixedGridStrategy, demo_stock, start_date, end_date,
        label='固定网格', plot=True, use_sizer=False,
    )

    bh = calc_buy_and_hold(demo_stock, start_date, end_date)

    print(f"\n{'  三策略对比 ':=^60}")
    _print_three_strategy_table(bh, r_fixed, r_chan)

    # ================================================================
    # Part 2: 多标的横向对比
    # ================================================================
    print(f"\n{'=' * 70}")
    print("Part 2: 多标的横向对比")
    print(f"{'=' * 70}")

    stocks = [
        ('600519.SH', '贵州茅台'),
        ('688981.SH', '中芯国际'),
        ('000001.SZ', '平安银行'),
        ('159941.SZ', '纳指ETF'),
    ]

    all_results = {}
    for code, name in stocks:
        try:
            print(f"\n--- {name}({code}) ---")

            stock_df = load_stock_data(code, start_date, end_date)
            stock_chan = run_chan(stock_df, symbol=code)
            stock_signal = chan_to_signal_df(stock_df, stock_chan)

            zs_count = len(stock_chan['zs_list'])
            print(f"  chan.py分析: {len(stock_chan['bi_list'])}笔, {zs_count}个中枢")

            r_c = run_and_report(
                ChanGridStrategy, stock_code=code,
                label=f'{name}-中枢网格', plot=True, use_sizer=False,
                df=stock_signal, data_class=ChanPandasData,
            )
            r_f = run_and_report(
                FixedGridStrategy, code, start_date, end_date,
                label=f'{name}-固定网格', plot=True, use_sizer=False,
            )
            bh_r = calc_buy_and_hold(code, start_date, end_date)

            all_results[code] = {
                'name': name, 'chan': r_c, 'fixed': r_f, 'bh': bh_r,
                'zs_count': zs_count,
            }
        except Exception as e:
            print(f"  {name}({code}): 跳过 - {e}")

    # 汇总对比
    if all_results:
        print(f"\n{'=' * 70}")
        print("多标的汇总对比")
        print(f"{'=' * 70}")

        for code, data in all_results.items():
            print(f"\n{data['name']}({code}) - {data['zs_count']}个中枢:")
            _print_three_strategy_table(data['bh'], data['fixed'], data['chan'])

    print("\n关键发现:")
    print("  - 中枢网格的区间由市场结构决定, 比固定区间更精准")
    print("  - 中枢多的标的(震荡多): 网格机会更多, 收益更稳")
    print("  - 中枢少的标的(趋势强): 网格机会少, 但出中枢停止避免了大亏")
    print("  - 中枢切换时自动重建网格, 适应市场变化")
    print("  - 下一步: 出中枢后不空仓, 而是切换为趋势跟踪模式")
