# -*- coding: utf-8 -*-
"""
数据加载与回测工具模块

本模块是整个缠论量化的"基础设施层", 提供三大核心功能:

1. 数据加载: 从 MySQL 读取 K 线数据, 封装为 Pandas DataFrame
   - load_stock_data(): 加载单只股票日K线
   - calc_buy_and_hold(): 计算买入持有收益率作为基准

2. Backtrader 集成:
   - ChanPandasData: 自定义数据类, 扩展了缠论信号列 (chan_signal/zg/zd/weekly_trend)
     让 Backtrader 策略可以像访问价格一样直接访问这些信号
   - setup_cerebro(): 一站式配置 Backtrader 引擎 (数据/资金/手续费/分析器)

3. 回测运行与绩效评估:
   - _wrap_strategy(): 策略包装器, 自动记录每笔交易和每日净值
   - run_and_report(): 一键运行回测 + 打印绩效报告 + 生成图表
   - plot_backtest(): 三面板可视化 (K线+买卖点 / 净值曲线 / 回撤曲线)
   - _calc_metrics(): 计算完整的绩效指标体系 (10+个指标)

使用流程:
  from data_loader import load_stock_data, ChanPandasData, run_and_report

  df = load_stock_data('600519.SH', '2024-01-01', '2025-12-31')
  # ... 执行缠论分析, 生成 signal_df ...
  result = run_and_report(MyStrategy, df=signal_df, data_class=ChanPandasData)
"""
import pandas as pd
import numpy as np
import backtrader as bt
import os
from db_config import execute_query, INITIAL_CASH, COMMISSION, POSITION_PCT


# ============================================================
# 缠论专用 PandasData（支持信号列）
# ============================================================

class ChanPandasData(bt.feeds.PandasData):
    """
    扩展 Backtrader 的 PandasData, 增加缠论专用信号线

    为什么需要自定义数据类:
      Backtrader 默认只识别 open/high/low/close/volume 等标准列。
      而我们分析出的缠论信号 (chan_signal, chan_zg, chan_zd, weekly_trend)
      需要让策略在 next() 中通过 self.data.chan_signal[0] 访问。

    新增的数据线:
      - chan_signal:  缠论信号 (0=无, 1=一买, 2=二买, 3=三买, -3=三卖)
      - chan_zg:      最近中枢的上沿 (ZG), 用于止损判断
      - chan_zd:      最近中枢的下沿 (ZD), 用于突破判断
      - weekly_trend: 周线趋势方向 (1=上升, -1=下跌, 0=震荡), 用于多周期过滤

    params 中的 -1 表示"在 DataFrame 中查找同名列"。
    如果 DataFrame 中有名为 'chan_signal' 的列, 自动映射到此数据线。
    """
    lines = ('chan_signal', 'chan_zg', 'chan_zd', 'weekly_trend',)
    params = (
        ('chan_signal', -1),
        ('chan_zg', -1),
        ('chan_zd', -1),
        ('weekly_trend', -1),
    )


# ============================================================
# 数据加载
# ============================================================

def load_stock_data(stock_code, start_date=None, end_date=None):
    """
    从 MySQL 加载日 K 线数据

    参数:
        stock_code: 股票代码, 如 '600519.SH' (必须包含后缀 .SH/.SZ)
        start_date: 开始日期, 如 '2024-01-01' (None 表示不限制)
        end_date:   结束日期, 如 '2025-12-31' (None 表示不限制)

    返回:
        pandas DataFrame, 索引为日期 (DatetimeIndex), 列为:
        open/high/low/close/volume (全部为 float 类型)

    注意事项:
      - 数据库中的字段名 (trade_date, open_price 等) 与策略中的字段名不同,
        函数中通过 .columns 重命名完成映射
      - 使用 pd.to_numeric(errors='coerce') 确保所有价格列为数值类型,
        非数值数据会被转为 NaN, 避免后续计算报错
      - 按日期升序排列, 因为 Backtrader 要求数据时间正序
    """
    # 动态构建 WHERE 条件: 如果有日期过滤, 追加到 SQL 中
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

    df = pd.DataFrame(rows)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    # 将数据库字段名映射为策略使用的标准名称
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def calc_buy_and_hold(stock_code, start_date, end_date):
    """
    计算区间买入持有 (Buy & Hold) 收益率

    用于与策略收益进行对比: 如果策略连买入持有都跑不赢, 说明策略没有价值。

    计算方式:
      (期末收盘价 / 期初收盘价) - 1

    返回:
        float: 收益率 (如 0.15 表示 +15%), 数据不足或异常返回 None
    """
    try:
        df = load_stock_data(stock_code, start_date, end_date)
        if len(df) < 2:
            return None
        return float(df['close'].iloc[-1] / df['close'].iloc[0] - 1)
    except Exception:
        return None


# ============================================================
# 策略包装器 - 自动记录交易和净值
# ============================================================

def _wrap_strategy(strategy_class):
    """
    包装任意策略类, 自动记录交易日志和净值曲线

    为什么需要包装器而不是在策略类中直接写:
      每个策略类都要写重复的 _trade_log / _nav_log 逻辑,
      通过包装器可以一次实现, 所有策略复用。

    包装器功能:
      1. _trade_log: 记录每笔已完成的买卖 (日期/方向/价格/数量)
      2. _nav_log:   记录每个交易日的账户净值 (用于绘制净值曲线和计算回撤)

    实现方式:
      运行时动态创建一个继承自原始策略的新类, 重写 notify_order 和 next 方法。
      使用 __name__ / __qualname__ / __module__ 保持原始策略的名称信息,
      避免 Backtrader 在报告中显示奇怪的类名。
    """
    class WrappedStrategy(strategy_class):
        def __init__(self):
            super().__init__()
            self._trade_log = []   # 交易记录列表: [{'date':..., 'type':'BUY'/'SELL', 'price':..., 'size':...}]
            self._nav_log = []     # 净值记录列表: [{'date':..., 'nav': 账户总价值}]

        def notify_order(self, order):
            """
            订单状态回调: 仅在订单完成时记录交易

            注意这里记录的是 executed 价格(实际成交价)而非下单价格,
            因为限价单可能存在滑点。abs(int(size)) 确保数量为正数。
            """
            if order.status == order.Completed:
                self._trade_log.append({
                    'date': self.data.datetime.date(0),
                    'type': 'BUY' if order.isbuy() else 'SELL',
                    'price': round(order.executed.price, 2),
                    'size': abs(int(order.executed.size)),
                })
            # 调用父类的 notify_order, 如果父类有定义的话
            if hasattr(super(), 'notify_order'):
                super().notify_order(order)

        def next(self):
            """每个交易日记录账户净值, 然后执行原始策略逻辑"""
            self._nav_log.append({
                'date': self.data.datetime.date(0),
                'nav': self.broker.getvalue(),
            })
            super().next()

    # 保持原始策略的元信息, 使 Backtrader 输出正确的策略名称
    WrappedStrategy.__name__ = strategy_class.__name__
    WrappedStrategy.__qualname__ = strategy_class.__qualname__
    WrappedStrategy.__module__ = strategy_class.__module__
    return WrappedStrategy


# ============================================================
# Cerebro 配置
# ============================================================

def setup_cerebro(strategy_class, stock_code=None, start_date=None, end_date=None,
                  use_sizer=True, df=None, data_class=None, **strategy_kwargs):
    """
    创建并配置好 Backtrader Cerebro 引擎

    Cerebro 是 Backtrader 的核心: 它连接数据、策略、经纪商和分析器,
    然后驱动回测循环 (逐个交易日调用策略的 next 方法)。

    参数:
        stock_code: 股票代码 (当 df 为 None 时, 从数据库加载)
        df: 直接传入的 DataFrame (优先于 stock_code), 可包含信号列
        data_class: 自定义数据类 (如有信号列, 必须用 ChanPandasData)
        use_sizer: 是否自动添加仓位管理器 (PercentSizer, 按百分比下单)
        **strategy_kwargs: 传递给策略的额外参数 (如 take_profit_pct=0.15)

    返回:
        (cerebro, df) 元组: cerebro 是配置好的引擎, df 是实际使用的 DataFrame
    """
    # 优先使用传入的 DataFrame, 否则从数据库加载
    if df is None:
        df = load_stock_data(stock_code, start_date, end_date)

    cerebro = bt.Cerebro()

    # 添加策略 (通过 **strategy_kwargs 传递策略参数)
    cerebro.addstrategy(strategy_class, **strategy_kwargs)

    # 添加数据 (使用指定的数据类或默认 PandasData)
    feed_class = data_class or bt.feeds.PandasData
    cerebro.adddata(feed_class(dataname=df))

    # 配置经纪商: 初始资金和手续费率
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

    # 添加百分比仓位管理器: 每次交易使用账户的 POSITION_PCT% 资金
    if use_sizer:
        cerebro.addsizer(bt.sizers.PercentSizer, percents=POSITION_PCT)

    # 添加三个内置分析器, 用于后续的绩效评估
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)  # 夏普比率 (无风险利率2%)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')                      # 回撤分析
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')                   # 交易统计

    return cerebro, df


# ============================================================
# 绩效计算
# ============================================================

def _calc_metrics(cerebro, strat, df):
    """
    从 Backtrader 运行结果中提取完整绩效指标

    指标体系的构建思路:
      1. 基础指标: 总收益、年化收益、最大回撤
      2. 风险调整: 夏普比率、卡玛比率
      3. 交易统计: 总交易次数、胜率、盈亏比
      4. 稳定性: 利润因子、最大连续亏损、期望值

    这样设计的目的是从多个维度评估策略:
      - 收益高但回撤大 → 持有体验差
      - 胜率高但盈亏比低 → 赚小亏大, 不可持续
      - 利润因子 < 1 → 策略在亏钱

    参数:
        cerebro: 运行后的 Cerebro 引擎 (含最终资金)
        strat:   Backtrader 运行返回的第一个策略实例 (含分析器结果)
        df:      原始 DataFrame (用于计算交易天数)

    返回:
        dict: 包含 15+ 个绩效指标的字典
    """
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH

    # 年化收益: (1+总收益)^(1/年数) - 1
    # 假设一年约 252 个交易日
    trading_days = len(df)
    years = trading_days / 252
    if years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = total_return

    # 夏普比率: 衡量每单位风险获得的超额收益
    # 通常 > 1 可接受, > 2 很好, > 3 极好
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0

    # 最大回撤: 从峰值到谷底的最大跌幅, 衡量策略的"最差体验"
    dd = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd.get('max', {}).get('drawdown', 0) / 100  # 转小数
    max_dd_len = dd.get('max', {}).get('len', 0)               # 最大回撤持续天数

    # 卡玛比率: 年化收益 / 最大回撤, 衡量"每单位回撤产生的收益"
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # 交易统计
    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get('total', {}).get('total', 0)
    won_trades = ta.get('won', {}).get('total', 0)
    lost_trades = ta.get('lost', {}).get('total', 0)
    win_rate = won_trades / total_trades if total_trades > 0 else 0

    # 平均盈亏
    avg_win = ta.get('won', {}).get('pnl', {}).get('average', 0) or 0
    avg_loss = ta.get('lost', {}).get('pnl', {}).get('average', 0) or 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0  # 盈亏比, > 2 较好

    # 利润因子: 总盈利 / 总亏损, > 1 策略盈利, > 2 优秀
    gross_profit = ta.get('won', {}).get('pnl', {}).get('total', 0) or 0
    gross_loss = ta.get('lost', {}).get('pnl', {}).get('total', 0) or 0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

    # 最大连续亏损次数: 衡量策略的"至暗时刻", 影响实盘的心理承受能力
    max_consecutive_losses = _calc_max_consecutive_losses(ta)

    # 期望值: 每笔交易的平均盈利 = 胜率 * 平均盈利 - 败率 * 平均亏损
    # 必须 > 0 策略才有长期正期望
    expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss if total_trades > 0 else 0

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
    }


def _calc_max_consecutive_losses(ta):
    """
    从 TradeAnalyzer 提取最大连续亏损次数

    TradeAnalyzer 的 streak 字段记录了连续盈利/亏损的统计。
    这是一个容易忽略但对实盘至关重要的指标:
    如果最大连续亏损是 10 次, 你是否有心理准备承受?
    """
    streak = ta.get('streak', {})
    lost_streak = streak.get('lost', {})
    return lost_streak.get('longest', 0) if lost_streak else 0


# ============================================================
# 运行回测 + 输出报告
# ============================================================

def run_and_report(strategy_class, stock_code=None, start_date=None, end_date=None,
                   label='', plot=False, quiet=False, use_sizer=True,
                   df=None, data_class=None, **strategy_kwargs):
    """
    一站式回测函数: 配置 -> 运行 -> 评估 -> 可视化

    这是本模块最常用的函数, 它将 setup_cerebro → cerebro.run() → _calc_metrics
    整合为一步调用, 并支持可选的图表输出。

    参数:
        strategy_class: 策略类 (继承 bt.Strategy)
        stock_code: 股票代码 (df 为 None 时使用)
        df: 直接传入的 DataFrame (优先于 stock_code), 可含缠论信号列
        data_class: 自定义数据类 (如 ChanPandasData, 当 df 含信号列时必须传入)
        label: 策略显示名称 (用于打印和图表标题)
        plot: 是否输出可视化图表到 outputs/ 目录
        quiet: 为 True 时不打印绩效报告 (批量回测时使用)
        use_sizer: 是否使用默认仓位管理器
        **strategy_kwargs: 传递给策略构造函数的额外参数

    返回:
        dict: 包含所有绩效指标、DataFrame、交易记录、净值曲线
    """
    # 包装策略以自动记录交易和净值
    wrapped = _wrap_strategy(strategy_class)
    cerebro, df = setup_cerebro(wrapped, stock_code, start_date, end_date,
                                use_sizer=use_sizer, df=df, data_class=data_class,
                                **strategy_kwargs)

    # 打印回测头信息 (股票代码+时间范围)
    if not quiet and label:
        print(f"{label} | {stock_code or ''} | {df.index[0].strftime('%Y-%m-%d')} ~ "
              f"{df.index[-1].strftime('%Y-%m-%d')} | {len(df)}个交易日")

    # 运行回测
    results = cerebro.run()
    strat = results[0]
    m = _calc_metrics(cerebro, strat, df)

    # 打印绩效摘要
    if not quiet:
        print(f"  总收益: {m['total_return']*100:+.2f}% | 年化: {m['annual_return']*100:+.2f}% | "
              f"最大回撤: {m['max_drawdown']*100:.2f}% | 夏普: {m['sharpe_ratio']:.2f} | "
              f"卡玛: {m['calmar_ratio']:.2f}")
        print(f"  交易: {m['total_trades']}次 | 胜率: {m['win_rate']*100:.1f}% | "
              f"盈亏比: {m['profit_loss_ratio']:.2f} | 利润因子: {m['profit_factor']:.2f} | "
              f"最大连亏: {m['max_consecutive_losses']}次")

    # 组装返回结果 (包含交易记录和净值数据供后续分析绘图)
    result = {**m, 'df': df, 'trades': strat._trade_log, 'nav': strat._nav_log}

    # 可选: 输出图表
    if plot:
        chart_name = label or strategy_class.__name__
        plot_backtest(result, stock_code or '', chart_name)

    return result


# ============================================================
# 可视化图表
# ============================================================

def plot_backtest(result, stock_code='', title=''):
    """
    绘制回测结果三面板图表

    布局:
      上图: K线(收盘价) + 买卖点标记 (直观看到在什么位置交易)
      中图: 净值曲线 vs 买入持有基准 (策略到底有没有创造价值)
      下图: 回撤曲线 (让用户直观感受持有期的"痛苦程度")

    参数:
        result: run_and_report 返回的字典 (含 df/trades/nav)
        stock_code: 股票代码 (显示在标题)
        title: 图表标题
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']       # 设置中文字体
    matplotlib.rcParams['axes.unicode_minus'] = False          # 解决负号显示为方块的问题

    os.makedirs('outputs', exist_ok=True)

    df = result['df']
    trades = result.get('trades', [])
    nav_data = result.get('nav', [])

    if not nav_data:
        print("没有净值数据，跳过绘图")
        return

    # 处理净值数据: 计算百分比净值和回撤
    nav_df = pd.DataFrame(nav_data)
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df.set_index('date', inplace=True)
    nav_df['nav_pct'] = nav_df['nav'] / INITIAL_CASH  # 归一化到初始资金
    nav_df['peak'] = nav_df['nav'].cummax()           # 历史峰值
    nav_df['drawdown'] = (nav_df['nav'] - nav_df['peak']) / nav_df['peak'] * 100  # 回撤百分比

    # 买入持有基准: 以收盘价计算, 归一化到1
    close_start = float(df['close'].iloc[0])
    benchmark = df['close'] / close_start

    # 提取买卖点数据
    buy_dates = [t['date'] for t in trades if t['type'] == 'BUY']
    buy_prices = [t['price'] for t in trades if t['type'] == 'BUY']
    sell_dates = [t['date'] for t in trades if t['type'] == 'SELL']
    sell_prices = [t['price'] for t in trades if t['type'] == 'SELL']

    m = result
    # 创建三面板: 高度比例 3:2:1 (K线图最大, 净值次之, 回撤最小)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                         gridspec_kw={'height_ratios': [3, 2, 1]})

    # ============ 上图: 收盘价 + 买卖点标记 ============
    ax1.plot(df.index, df['close'], 'gray', linewidth=1, alpha=0.8, label='收盘价')
    if buy_dates:
        # 买入用红色上三角 (^), 因为买入是"向上"的预期
        ax1.scatter(buy_dates, buy_prices, color='#e74c3c', marker='^', s=80,
                    zorder=5, label=f'买入({len(buy_dates)}次)')
    if sell_dates:
        # 卖出用绿色下三角 (v), 因为卖出是"向下"的兑现
        ax1.scatter(sell_dates, sell_prices, color='#2ecc71', marker='v', s=80,
                    zorder=5, label=f'卖出({len(sell_dates)}次)')
    ax1.set_ylabel('价格')
    ax1.set_title(f'{title}  {stock_code}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 右上角的信息框: 汇总关键绩效指标
    info_text = (
        f"Return:    {m['total_return']*100:+.2f}%\n"
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

    # ============ 中图: 净值曲线 vs 买入持有基准 ============
    ax2.plot(nav_df.index, nav_df['nav_pct'], '#2980b9', linewidth=1.5, label='策略净值')
    ax2.plot(benchmark.index, benchmark, 'gray', linewidth=1, alpha=0.6, label='买入持有')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)  # 基准线
    ax2.set_ylabel('净值 (初始=1.0)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ============ 下图: 回撤曲线 ============
    ax3.fill_between(nav_df.index, nav_df['drawdown'], 0, color='#e74c3c', alpha=0.4)
    ax3.plot(nav_df.index, nav_df['drawdown'], '#c0392b', linewidth=0.8)
    ax3.set_ylabel('回撤(%)')
    ax3.set_xlabel('日期')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表到 outputs 目录
    safe_name = title.replace(' ', '_').replace('/', '_')
    plot_file = os.path.join('outputs', f'{safe_name}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {plot_file}")
    plt.close()
