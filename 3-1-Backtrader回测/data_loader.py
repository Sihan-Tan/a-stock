# -*- coding: utf-8 -*-
"""
数据加载与回测工具模块

功能:
  - 从MySQL读取K线数据 (trade_stock_daily 表)
  - 统一配置Cerebro引擎 (初始资金/手续费/仓位 来自 .env)
  - 自动记录交易和净值 (包装任意策略类)
  - 计算完整绩效指标 (夏普、回撤、胜率、盈亏比等)
  - 生成可视化图表 (K线+买卖点, 净值曲线, 回撤曲线)
"""
import os
import pandas as pd
import numpy as np
import backtrader as bt
from db_config import execute_query, INITIAL_CASH, COMMISSION, POSITION_PCT


# ============================================================
# 数据加载
# ============================================================

def load_stock_data(stock_code, start_date=None, end_date=None):
    """
    从MySQL加载日K线数据，返回 Backtrader 可用的 DataFrame

    参数:
        stock_code: 股票代码，如 '600519.SH' (贵州茅台)
        start_date: 开始日期，如 '2024-01-01'
        end_date:   结束日期，如 '2025-12-31'

    返回:
        pandas DataFrame，索引为日期(trade_date)，列为 open/high/low/close/volume

    处理流程:
      1. 构建动态SQL查询 (按日期范围筛选)
      2. 执行查询获取原始数据
      3. 清洗数据: 重命名列、转换类型、过滤无效价格
    """
    # 动态构建 WHERE 条件，支持可选日期范围
    conditions = ["stock_code = %s"]
    params = [stock_code]

    if start_date:
        conditions.append("trade_date >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= %s")
        params.append(end_date)

    sql = f"""
        SELECT trade_date, open_price, high_price, low_price, close_price, volume
        FROM trade_stock_daily
        WHERE {' AND '.join(conditions)}
        ORDER BY trade_date ASC
    """
    rows = execute_query(sql, params)
    if not rows:
        raise ValueError(f"没有找到 {stock_code} 的数据，请检查数据库或先运行数据采集")

    # 将查询结果转为 DataFrame 并清洗
    df = pd.DataFrame(rows)
    df['trade_date'] = pd.to_datetime(df['trade_date'])       # 日期列转为 datetime 类型
    df.set_index('trade_date', inplace=True)                   # 将日期设为行索引
    df.columns = ['open', 'high', 'low', 'close', 'volume']   # 重命名为 OHLCV 标准列名

    # 确保所有价格列为数值类型 (非数值转为 NaN)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 过滤无效价格: 负价/零价多见于后复权异常，会误导策略信号和收益计算
    valid_mask = (df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)
    df = df.loc[valid_mask]
    if df.empty:
        raise ValueError(f"{stock_code} 过滤无效价格后无有效数据，请检查数据源")
    return df


# ============================================================
# 策略包装器 - 自动记录交易和净值
# ============================================================

def _wrap_strategy(strategy_class):
    """
    包装任意策略类，在不修改原策略逻辑的前提下，自动记录:
      - _trade_log: 每笔买卖交易的日期、方向、价格、数量
      - _nav_log: 每个交易日的账户净值

    原理: 通过继承原策略类，重写 notify_order 和 next 方法，
          在调用父类方法前后插入记录逻辑 (装饰器模式的OOP版本)
    """
    class WrappedStrategy(strategy_class):
        def __init__(self):
            super().__init__()
            self._trade_log = []   # 交易记录列表 [(date, BUY/SELL, price, size), ...]
            self._nav_log = []     # 净值记录列表 [(date, nav), ...]

        def notify_order(self, order):
            """订单状态变化时触发，这里只记录已成交的订单"""
            if order.status == order.Completed:
                self._trade_log.append({
                    'date': self.data.datetime.date(0),              # 成交日期
                    'type': 'BUY' if order.isbuy() else 'SELL',      # 买卖方向
                    'price': round(order.executed.price, 2),          # 成交价格(保留2位小数)
                    'size': abs(int(order.executed.size)),            # 成交数量(取绝对值)
                })
            # 如果原策略也定义了 notify_order，继续调用它 (方法解析顺序 MRO)
            if hasattr(super(), 'notify_order'):
                super().notify_order(order)

        def next(self):
            """每个交易日触发，先记录净值再执行原策略逻辑"""
            self._nav_log.append({
                'date': self.data.datetime.date(0),                  # 当前日期
                'nav': self.broker.getvalue(),                       # 当前账户总价值(资金+持仓市值)
            })
            super().next()

    # 保留原策略类的元信息 (方便调试时看到正确的类名)
    WrappedStrategy.__name__ = strategy_class.__name__
    WrappedStrategy.__qualname__ = strategy_class.__qualname__
    WrappedStrategy.__module__ = strategy_class.__module__
    return WrappedStrategy


# ============================================================
# Cerebro 配置 - 一站式配置回测引擎
# ============================================================

def setup_cerebro(strategy_class, stock_code, start_date=None, end_date=None, **strategy_kwargs):
    """
    创建并配置好 Cerebro 引擎，包含策略、数据、资金、分析器

    参数:
        strategy_class: Backtrader 策略类
        stock_code:     股票代码
        start_date:     数据起始日期
        end_date:       数据结束日期
        **strategy_kwargs: 传给策略类的额外参数 (如周期、阈值等)

    返回:
        (cerebro, df) - Cerebro 引擎实例 和 原始K线 DataFrame
    """
    # 加载K线数据
    df = load_stock_data(stock_code, start_date, end_date)

    # 创建 Cerebro 大脑 - Backtrader 的核心调度引擎
    cerebro = bt.Cerebro()

    # 注入策略和数据
    cerebro.addstrategy(strategy_class, **strategy_kwargs)           # 添加策略(可多个)
    cerebro.adddata(bt.feeds.PandasData(dataname=df))                # 添加数据(Pandas转Backtrader格式)

    # 配置资金和交易参数
    cerebro.broker.setcash(INITIAL_CASH)                             # 初始资金
    cerebro.broker.setcommission(commission=COMMISSION)              # 交易手续费率
    cerebro.addsizer(bt.sizers.PercentSizer, percents=POSITION_PCT)  # 仓位管理器(占总资金百分比)

    # 添加绩效分析器 (回测结束后自动计算)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)  # 夏普比率
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')                       # 最大回撤
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')                     # 交易分析

    return cerebro, df


# ============================================================
# 绩效计算
# ============================================================

def _calc_metrics(cerebro, strat, df):
    """从Backtrader回测结果中提取并计算完整绩效指标

    计算的指标包括:
      - 收益率: 总收益、年化收益
      - 风险: 最大回撤、回撤持续天数
      - 风险调整: 夏普比率、卡玛比率
      - 交易: 总次数、胜率、盈亏比、利润因子、最大连亏
      - 基准: 买入持有收益率（用于对比策略是否跑赢大盘）
    """
    # --- 收益率 ---
    final_value = cerebro.broker.getvalue()                    # 回测结束后的账户总价值
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH # 总收益率

    # 年化收益率 = (1 + 总收益) ^ (1/年数) - 1
    trading_days = len(df)
    years = trading_days / 252                                  # 一年约252个交易日
    if years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = total_return

    # --- 夏普比率 ---
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0

    # --- 最大回撤 ---
    # 优先用净值序列手动计算 (比Backtrader内置分析器更可靠，避免>100%的异常值)
    max_drawdown = 0.0
    max_dd_len = 0                                               # 最大回撤持续天数
    nav_log = getattr(strat, '_nav_log', [])
    if nav_log:
        navs = [x['nav'] for x in nav_log]                      # 提取每日净值列表
        peak = navs[0]                                           # 历史最高净值
        dd_len = 0
        for v in navs:
            if v > peak:
                peak = v                                         # 创新高，重置峰值
                dd_len = 0                                       # 回撤天数归零
            else:
                dd_len += 1
                if peak > 0 and v > 0:
                    dd_pct = (peak - v) / peak                   # 当前回撤比例
                    max_drawdown = max(max_drawdown, min(dd_pct, 1.0))  # 长仓回撤不超过100%
                max_dd_len = max(max_dd_len, dd_len)
    if not nav_log:
        # 降级方案: 使用Backtrader内置分析器
        dd = strat.analyzers.drawdown.get_analysis()
        bt_dd = dd.get('max', {}).get('drawdown', 0) / 100      # Backtrader返回百分比，除以100转小数
        max_drawdown = min(bt_dd, 1.0)                          # 长仓策略回撤理论不超过100%
        max_dd_len = dd.get('max', {}).get('len', 0)

    # 卡玛比率 = 年化收益 / 最大回撤 (衡量每单位回撤带来的收益)
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # --- 交易统计 ---
    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get('total', {}).get('total', 0)          # 总交易笔数
    won_trades = ta.get('won', {}).get('total', 0)              # 盈利笔数
    lost_trades = ta.get('lost', {}).get('total', 0)            # 亏损笔数
    win_rate = won_trades / total_trades if total_trades > 0 else 0  # 胜率

    # 盈亏比 = |平均盈利| / |平均亏损| (>1 表示盈利大于亏损)
    avg_win = ta.get('won', {}).get('pnl', {}).get('average', 0) or 0    # 平均每笔盈利
    avg_loss = ta.get('lost', {}).get('pnl', {}).get('average', 0) or 0  # 平均每笔亏损
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # 利润因子 = 总盈利 / |总亏损| (>1 表示策略整体盈利)
    gross_profit = ta.get('won', {}).get('pnl', {}).get('total', 0) or 0   # 总盈利金额
    gross_loss = ta.get('lost', {}).get('pnl', {}).get('total', 0) or 0    # 总亏损金额
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

    # 最大连续亏损次数 (反映策略连续回撤的忍耐度)
    max_consecutive_losses = _calc_max_consecutive_losses(ta)

    # 期望值 = 胜率 × 平均盈利 + (1-胜率) × 平均亏损 (正数则策略有正期望)
    expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss if total_trades > 0 else 0

    # --- 买入持有基准 ---
    # 用于对比：如果直接买入持有同一只股票，收益是多少？
    valid_close = df['close'][df['close'] > 0]
    if len(valid_close) >= 2:
        close_start = float(valid_close.iloc[0])                 # 期初收盘价
        close_end = float(valid_close.iloc[-1])                  # 期末收盘价
        benchmark_return = (close_end / close_start - 1) if close_start > 0 else 0
    else:
        benchmark_return = 0

    return {
        'final_value': round(final_value, 2),
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'max_dd_len': max_dd_len,
        'sharpe_ratio': round(sharpe_ratio, 4),
        'calmar_ratio': round(calmar_ratio, 4),
        'total_trades': total_trades,
        'won_trades': won_trades,
        'lost_trades': lost_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': round(profit_loss_ratio, 2),
        'profit_factor': round(profit_factor, 2),
        'max_consecutive_losses': max_consecutive_losses,
        'expected_value': round(expected_value, 2),
        'years': round(years, 2),
        'trading_days': trading_days,
        'benchmark_return': benchmark_return,
    }


def _calc_max_consecutive_losses(ta):
    """从TradeAnalyzer的streak分析中提取最大连续亏损次数"""
    streak = ta.get('streak', {})
    lost_streak = streak.get('lost', {})
    return lost_streak.get('longest', 0) if lost_streak else 0


# ============================================================
# 运行回测 + 输出报告
# ============================================================

def run_and_report(strategy_class, stock_code, start_date=None, end_date=None,
                   label='', plot=False, **strategy_kwargs):
    """
    一站式接口: 运行回测并打印绩效报告

    参数:
        strategy_class: 策略类 (bt.Strategy 的子类)
        stock_code:     股票代码，如 '600519.SH'
        start_date:     回测起始日期
        end_date:       回测结束日期
        label:          策略显示名称 (用于打印和图表标题)
        plot:           是否输出可视化图表到 outputs/ 目录
        **strategy_kwargs: 传给策略类的额外参数

    返回:
        dict 包含所有绩效指标、原始K线数据、交易记录、净值序列
    """
    # 包装策略 (自动记录交易+净值)
    wrapped = _wrap_strategy(strategy_class)
    # 创建并配置 Cerebro 引擎
    cerebro, df = setup_cerebro(wrapped, stock_code, start_date, end_date, **strategy_kwargs)

    if label:
        print(f"{label} | {stock_code} | {df.index[0].strftime('%Y-%m-%d')} ~ "
              f"{df.index[-1].strftime('%Y-%m-%d')} | {len(df)}个交易日")

    # 运行回测
    results = cerebro.run()
    strat = results[0]

    # 计算所有绩效指标
    m = _calc_metrics(cerebro, strat, df)

    # 打印绩效摘要
    print(f"  总收益: {m['total_return']*100:+.2f}% | 年化: {m['annual_return']*100:+.2f}% | "
          f"最大回撤: {m['max_drawdown']*100:.2f}% | 夏普: {m['sharpe_ratio']:.2f} | "
          f"卡玛: {m['calmar_ratio']:.2f}")
    print(f"  交易: {m['total_trades']}次 | 胜率: {m['win_rate']*100:.1f}% | "
          f"盈亏比: {m['profit_loss_ratio']:.2f} | 利润因子: {m['profit_factor']:.2f} | "
          f"最大连亏: {m['max_consecutive_losses']}次")
    print(f"  [基准] 买入持有: {m['benchmark_return']*100:+.2f}%")

    # 组装返回结果
    result = {**m, 'df': df, 'trades': strat._trade_log, 'nav': strat._nav_log}

    if plot:
        chart_name = label or strategy_class.__name__
        plot_backtest(result, stock_code, chart_name)

    return result


# ============================================================
# 可视化图表
# ============================================================

def plot_backtest(result, stock_code='', title=''):
    """
    绘制回测结果的三栏图表并保存到 outputs/ 目录

    三栏布局:
      上图: K线(收盘价折线) + 买卖点标记 (红色三角=买, 绿色三角=卖)
      中图: 策略净值曲线 vs 买入持有基准线
      下图: 回撤填充图 (红色区域 = 从历史最高点的回撤幅度)

    参数:
        result:     run_and_report 的返回值
        stock_code: 股票代码（用于图表标题）
        title:      图表标题
    """
    import matplotlib.pyplot as plt
    import matplotlib

    # 设置中文字体和负号显示 (Windows 下使用 SimHei 黑体)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    os.makedirs('outputs', exist_ok=True)

    df = result['df']
    trades = result.get('trades', [])
    nav_data = result.get('nav', [])

    if not nav_data:
        print("没有净值数据，跳过绘图")
        return

    # ---- 构建净值 DataFrame ----
    nav_df = pd.DataFrame(nav_data)
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df.set_index('date', inplace=True)
    nav_df['nav_pct'] = nav_df['nav'] / INITIAL_CASH                 # 归一化净值 (初始=1.0)
    nav_df['peak'] = nav_df['nav'].cummax()                          # 历史最高净值 (滚动最大值)
    nav_df['drawdown'] = (nav_df['nav'] - nav_df['peak']) / nav_df['peak'] * 100  # 回撤百分比

    # ---- 买入持有基准 ----
    close_start = float(df['close'].iloc[0])                         # 期初收盘价
    benchmark = df['close'] / close_start                             # 归一化基准线

    # ---- 分离买卖点 ----
    buy_dates = [t['date'] for t in trades if t['type'] == 'BUY']
    buy_prices = [t['price'] for t in trades if t['type'] == 'BUY']
    sell_dates = [t['date'] for t in trades if t['type'] == 'SELL']
    sell_prices = [t['price'] for t in trades if t['type'] == 'SELL']

    m = result
    # 创建3行子图，高度比 3:2:1
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                         gridspec_kw={'height_ratios': [3, 2, 1]})

    # ============================================================
    # 上图: K线(收盘价) + 买卖点标记
    # ============================================================
    ax1.plot(df.index, df['close'], 'gray', linewidth=1, alpha=0.8, label='收盘价')
    if buy_dates:
        ax1.scatter(buy_dates, buy_prices, color='#e74c3c', marker='^', s=80,
                    zorder=5, label=f'买入({len(buy_dates)}次)')      # 红色上三角
    if sell_dates:
        ax1.scatter(sell_dates, sell_prices, color='#2ecc71', marker='v', s=80,
                    zorder=5, label=f'卖出({len(sell_dates)}次)')     # 绿色下三角
    ax1.set_ylabel('价格')
    ax1.set_title(f'{title}  {stock_code}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 右上角绩效指标注释框
    info_text = (
        f"Return:    {m['total_return']*100:+.2f}%\n"
        f"Benchmark: {m.get('benchmark_return', 0)*100:+.2f}%\n"
        f"Annual:    {m['annual_return']*100:+.2f}%\n"
        f"MaxDD:     {m['max_drawdown']*100:.2f}%\n"
        f"Sharpe:    {m['sharpe_ratio']:.2f}\n"
        f"Calmar:    {m['calmar_ratio']:.2f}\n"
        f"WinRate:   {m['win_rate']*100:.1f}%\n"
        f"P/L Ratio: {m['profit_loss_ratio']:.2f}\n"
        f"ProfitF:   {m['profit_factor']:.2f}"
    )
    ax1.text(0.98, 0.97, info_text, transform=ax1.transAxes,
             fontsize=9, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.8),
             family='monospace')

    # ============================================================
    # 中图: 净值曲线 vs 买入持有基准
    # ============================================================
    ax2.plot(nav_df.index, nav_df['nav_pct'], '#2980b9', linewidth=1.5, label='策略净值')
    ax2.plot(benchmark.index, benchmark, 'gray', linewidth=1, alpha=0.6, label='买入持有')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)       # 初始净值水平线
    ax2.set_ylabel('净值 (初始=1.0)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ============================================================
    # 下图: 回撤曲线
    # ============================================================
    ax3.fill_between(nav_df.index, nav_df['drawdown'], 0, color='#e74c3c', alpha=0.4)  # 红色填充
    ax3.plot(nav_df.index, nav_df['drawdown'], '#c0392b', linewidth=0.8)               # 深红折线
    ax3.set_ylabel('回撤(%)')
    ax3.set_xlabel('日期')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表 (文件名去除空格和斜杠)
    safe_name = title.replace(' ', '_').replace('/', '_')
    plot_file = os.path.join('outputs', f'{safe_name}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {plot_file}")
    plt.close()
