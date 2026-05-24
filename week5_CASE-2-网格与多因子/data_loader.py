# -*- coding: utf-8 -*-
"""
数据加载与回测工具模块
======================

本模块是整个回测框架的核心工具层, 提供以下关键功能:

1. 数据加载:
   - load_stock_data(): 从 MySQL 读取日K线数据, 返回标准格式的 DataFrame
   - calc_buy_and_hold(): 计算买入持有收益率, 作为策略的基准对比

2. Backtrader 集成:
   - ChanPandasData: 扩展自 bt.feeds.PandasData, 增加了缠论信号线 (chan_signal/chan_zg/chan_zd)
   - setup_cerebro(): 统一配置 Cerebro 引擎 (数据/资金/手续费/分析器)
   - _wrap_strategy(): 自动包装策略类, 注入交易记录和净值记录功能

3. 绩效评估:
   - _calc_metrics(): 从回测结果中提取完整绩效指标 (收益率/回撤/夏普/胜率等)
   - run_and_report(): 一键运行回测 + 打印绩效报告

4. 可视化:
   - plot_backtest(): 生成三合一图表 (K线+买卖点 / 净值曲线 / 回撤曲线)

网格策略特别说明:
  网格策略自行管理仓位 (按格子分批买入), 不需要 Backtrader 的默认 Sizer。
  调用 run_and_report() 时应设置 use_sizer=False。

依赖:
  - db_config: 数据库配置和查询接口
  - backtrader: 回测框架
  - pandas/numpy: 数据处理
  - matplotlib: 绘图
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
    扩展自 Backtrader 标准 PandasData, 额外支持缠论信号列。

    Backtrader 的 PandasData 默认只识别 open/high/low/close/volume/interest 这6列。
    本类新增了 3 条自定义线 (lines):
      - chan_signal: 缠论买卖点信号 (0=无, 1=一买, 2=二买, 3=三买, -3=三卖等)
      - chan_zg:     中枢上沿 (ZG, 中枢高点)
      - chan_zd:     中枢下沿 (ZD, 中枢低点)

    参数说明:
      params 元组中的每个条目对应 columns 中的列索引。
      -1 表示该列在 DataFrame 中不存在, 此时该线的值始终为 NaN。

    使用方式:
        feed = ChanPandasData(dataname=signal_df)
        cerebro.adddata(feed)
    """
    # 定义新增的线 (lines), 策略中通过 self.data.chan_signal[0] 访问
    lines = ('chan_signal', 'chan_zg', 'chan_zd',)

    # 参数: (列名, 列索引), -1 表示 DataFrame 中不存在该列
    params = (
        ('chan_signal', -1),  # 缠论信号, 对应 DataFrame 中的 'chan_signal' 列
        ('chan_zg', -1),      # 中枢上沿, 对应 'chan_zg'
        ('chan_zd', -1),      # 中枢下沿, 对应 'chan_zd'
    )


# ============================================================
# 数据加载
# ============================================================

def load_stock_data(stock_code, start_date=None, end_date=None):
    """
    从 MySQL 数据库加载股票日K线数据。

    数据来源: trade_stock_daily 表, 包含每日的开高低收和成交量。

    参数:
        stock_code: 股票代码, 格式为 '600519.SH' (代码.交易所后缀)
        start_date: 开始日期, 格式为 '2024-01-01'。None 表示从最早的数据开始
        end_date:   结束日期, 格式为 '2025-12-31'。None 表示到最晚的数据结束

    返回值:
        pandas DataFrame, 索引为日期 (DatetimeIndex), 列为:
          open/ high/ low/ close/ volume (均为 float 类型)

    异常:
        ValueError: 如果数据库中没有该股票的数据

    典型用法:
        df = load_stock_data('600519.SH', '2024-01-01', '2025-12-31')
        # df 可直接传入 Backtrader 的 PandasData 使用
    """
    # 动态构建 WHERE 条件, 只有提供了日期参数时才加入过滤
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

    # 转换为 DataFrame 并统一列名
    # Backtrader 的 PandasData 期望的列名是小写的 open/high/low/close/volume
    df = pd.DataFrame(rows)
    df['trade_date'] = pd.to_datetime(df['trade_date'])  # 字符串转日期
    df.set_index('trade_date', inplace=True)              # 日期作为索引
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # 确保数值类型
    return df


def calc_buy_and_hold(stock_code, start_date, end_date):
    """
    计算"买入持有"策略在指定区间内的收益率。

    买入持有是最简单的被动策略: 期初买入, 期末卖出, 中间不做任何操作。
    通常用作主动策略(如网格/趋势跟踪)的基准对比。

    计算方法:
        return = (期末收盘价 - 期初收盘价) / 期初收盘价

    参数:
        stock_code: 股票代码
        start_date: 开始日期
        end_date:   结束日期

    返回值:
        float 收益率 (如 0.15 表示 +15%)。数据不足返回 None。
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
    包装任意策略类, 自动注入交易记录和净值记录功能。

    这是 Python 装饰器模式 / 动态继承的应用:
      1. 创建一个新的类 WrappedStrategy, 继承自 strategy_class
      2. 覆盖 __init__/notify_order/next 方法, 注入记录逻辑
      3. 保持类名和模块信息不变 (通过 __name__ 等属性赋值)

    注入的功能:
      - _trade_log: list[dict], 记录每笔已完成的交易 (日期/类型/价格/数量)
      - _nav_log:   list[dict], 记录每个交易日的资产净值

    为什么需要包装器而不是直接写在策略类里?
      因为 run_and_report() 要支持任意策略类,
      不能要求每个策略都手动写交易记录代码。
      包装器让"记录"和"策略逻辑"解耦。

    参数:
        strategy_class: 策略类 (继承自 bt.Strategy)

    返回值:
        一个新的类, 具有相同的类名和接口, 但增加了记录功能
    """
    class WrappedStrategy(strategy_class):
        def __init__(self):
            super().__init__()
            self._trade_log = []  # 交易记录: [{date, type, price, size}, ...]
            self._nav_log = []    # 净值记录: [{date, nav}, ...]

        def notify_order(self, order):
            """
            订单状态回调: 订单完成后记录交易。
            只记录 Completed 状态的订单 (Submitted/Accepted 等中间状态跳过)。
            """
            if order.status == order.Completed:
                self._trade_log.append({
                    'date': self.data.datetime.date(0),  # 订单执行日期
                    'type': 'BUY' if order.isbuy() else 'SELL',
                    'price': round(order.executed.price, 2),
                    'size': abs(int(order.executed.size)),
                })
            if hasattr(super(), 'notify_order'):
                super().notify_order(order)

        def next(self):
            """
            每个 Bar 的回调: 记录当前资产净值。
            broker.getvalue() 返回总资产 = 现金 + 持仓市值。
            """
            self._nav_log.append({
                'date': self.data.datetime.date(0),
                'nav': self.broker.getvalue(),
            })
            super().next()

    # 保持类名和模块信息不变, 确保 Backtrader 能正确识别
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
    创建并配置 Backtrader 的 Cerebro 引擎。

    这是连接"数据"和"策略"的装配函数, 统一完成以下配置:
      1. 加载数据 (从数据库或直接传入 DataFrame)
      2. 添加策略类
      3. 设置初始资金 / 手续费
      4. 设置仓位管理器 (Sizer)
      5. 添加绩效分析器 (夏普/回撤/交易分析)

    参数:
        strategy_class: 策略类 (使用原始类, 包装由 run_and_report 完成)
        stock_code: 股票代码 (当 df 为 None 时通过此参数从数据库加载数据)
        start_date: 开始日期
        end_date:   结束日期
        use_sizer:  是否使用默认仓位管理器。
                    True: 使用 PercentSizer 按固定比例下单 (适合普通策略)
                    False: 策略自行管理仓位 (网格策略必须设为 False)
        df:         直接传入的 DataFrame (优先于 stock_code, 支持含信号列的 DataFrame)
        data_class: 自定义数据类 (如 ChanPandasData), 默认 bt.feeds.PandasData

    返回值:
        (cerebro, df) 二元组
        - cerebro: 配置好的 Cerebro 引擎, 调用 run() 执行回测
        - df: 加载的 DataFrame
    """
    # 如果未提供 DataFrame, 从数据库加载
    if df is None:
        df = load_stock_data(stock_code, start_date, end_date)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, **strategy_kwargs)

    # 添加数据源: 使用指定的数据类或默认 PandasData
    feed_class = data_class or bt.feeds.PandasData
    cerebro.adddata(feed_class(dataname=df))

    # 设置初始资金和手续费
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

    # 仓位管理器: 网格策略不需要, 因为网格自行管理每格仓位
    if use_sizer:
        cerebro.addsizer(bt.sizers.PercentSizer, percents=POSITION_PCT)

    # 添加三个分析器, 用于回测后评估绩效
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    return cerebro, df


# ============================================================
# 绩效计算
# ============================================================

def _calc_metrics(cerebro, strat, df):
    """
    从 Backtrader 回测结果中提取完整的绩效指标。

    计算以下指标:
      - 总收益率 / 年化收益率
      - 最大回撤 (基于净值序列, 更精确)
      - 夏普比率 / 卡玛比率
      - 交易统计: 总次数 / 胜率 / 平均盈亏 / 盈亏比 / 利润因子
      - 最大连续亏损次数
      - 期望值 (每笔交易平均预期收益)

    参数:
        cerebro: 回测后的 Cerebro 引擎, 用于获取最终资产值
        strat:   cerebro.run() 返回的策略实例, 通过 analyzers 获取分析结果
        df:      原始 K 线 DataFrame, 用于计算交易天数

    返回值:
        dict, 包含上述所有指标的键值对

    实现细节:
      - 最大回撤优先使用 _nav_log 净值序列计算 (更精确, 可捕捉日内回撤)
      - 如果净值序列不存在, 则回退到 Backtrader 的 DrawDown 分析器结果
      - 所有比率计算都有分母为 0 的保护
    """
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH

    # 年化收益率: 用实际交易天数 / 252 (A股年交易约252天)
    trading_days = len(df)
    years = trading_days / 252
    if years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = total_return

    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0

    # ---- 最大回撤计算 ----
    # 方法1: 从净值序列计算 (优先, 更精确)
    max_drawdown = 0.0
    max_dd_len = 0
    nav_log = getattr(strat, '_nav_log', [])
    if nav_log:
        navs = [x['nav'] for x in nav_log]
        peak = navs[0]      # 初始峰值
        dd_len = 0          # 当前回撤持续天数
        for v in navs:
            if v > peak:
                peak = v    # 创新高, 重置峰值
                dd_len = 0
            else:
                dd_len += 1
                if peak > 0 and v > 0:
                    dd_pct = (peak - v) / peak
                    max_drawdown = max(max_drawdown, min(dd_pct, 1.0))
                max_dd_len = max(max_dd_len, dd_len)

    # 方法2: 从 Backtrader DrawDown 分析器获取 (备选)
    if not nav_log:
        dd = strat.analyzers.drawdown.get_analysis()
        bt_dd = dd.get('max', {}).get('drawdown', 0) / 100
        max_drawdown = min(bt_dd, 1.0)
        max_dd_len = dd.get('max', {}).get('len', 0)

    # 卡玛比率 = 年化收益 / 最大回撤, 衡量"收益/风险"的综合指标
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # ---- 交易统计 ----
    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get('total', {}).get('total', 0)
    won_trades = ta.get('won', {}).get('total', 0)
    lost_trades = ta.get('lost', {}).get('total', 0)
    win_rate = won_trades / total_trades if total_trades > 0 else 0

    avg_win = ta.get('won', {}).get('pnl', {}).get('average', 0) or 0
    avg_loss = ta.get('lost', {}).get('pnl', {}).get('average', 0) or 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0  # 盈亏比

    gross_profit = ta.get('won', {}).get('pnl', {}).get('total', 0) or 0
    gross_loss = ta.get('lost', {}).get('pnl', {}).get('total', 0) or 0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0  # 利润因子

    max_consecutive_losses = _calc_max_consecutive_losses(ta)

    # 期望值 = 胜率 * 平均盈利 + 败率 * 平均亏损
    # 正期望值意味着长期执行策略能赚钱
    expected_value = win_rate * avg_win + (1 - win_rate) * avg_loss if total_trades > 0 else 0

    return {
        'final_value': round(final_value, 2),
        'total_return': total_return,              # 总收益率
        'annual_return': annual_return,            # 年化收益率
        'max_drawdown': max_drawdown,              # 最大回撤 (0~1)
        'max_dd_len': max_dd_len,                  # 最大回撤持续天数
        'sharpe_ratio': round(sharpe_ratio, 4),    # 夏普比率
        'calmar_ratio': round(calmar_ratio, 4),    # 卡玛比率
        'total_trades': total_trades,              # 总交易次数
        'won_trades': won_trades,                  # 盈利交易次数
        'lost_trades': lost_trades,                # 亏损交易次数
        'win_rate': win_rate,                      # 胜率
        'avg_win': avg_win,                        # 平均盈利金额
        'avg_loss': avg_loss,                      # 平均亏损金额
        'profit_loss_ratio': round(profit_loss_ratio, 2),   # 盈亏比
        'profit_factor': round(profit_factor, 2),           # 利润因子 (总盈利/总亏损)
        'max_consecutive_losses': max_consecutive_losses,   # 最大连续亏损次数
        'expected_value': round(expected_value, 2),         # 期望值
        'years': round(years, 2),                # 回测年数
        'trading_days': trading_days,             # 交易天数
    }


def _calc_max_consecutive_losses(ta):
    """
    从 Backtrader TradeAnalyzer 中提取最大连续亏损次数。

    TradeAnalyzer 的 streak 字段记录了最长的连续盈利/亏损。
    这个指标对网格策略尤其重要: 连续亏损过多可能意味着网格区间设置不合理。

    参数:
        ta: TradeAnalyzer 的 get_analysis() 结果

    返回值:
        int: 最大连续亏损次数
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
    运行回测并打印绩效报告的"一站式"函数。

    这是最高层级的调用入口, 将以下步骤封装成一个函数:
      1. 包装策略类 (自动注入交易/净值记录)
      2. 配置 Cerebro 引擎
      3. 执行回测
      4. 计算绩效指标
      5. 打印报告
      6. 可选: 生成图表

    参数:
        strategy_class: 策略类 (原始的 bt.Strategy 子类)
        stock_code:     股票代码 (当 df 为 None 时使用)
        df:             直接传入含信号列的 DataFrame (优先于 stock_code)
        data_class:     自定义数据类, 如 ChanPandasData
        label:          显示名称, 用于区分多个回测结果
        plot:           是否输出可视化图表到 outputs/ 目录
        quiet:          为 True 时不打印报告, 仅返回结果字典
        use_sizer:      是否使用默认仓位管理器 (网格策略设为 False)

    返回值:
        dict, 包含:
          - 所有绩效指标 (_calc_metrics 的返回值)
          - 'df':     原始 K 线 DataFrame
          - 'trades': 交易记录列表
          - 'nav':    净值记录列表

    典型用法:
        result = run_and_report(
            SimpleGridStrategy, '600519.SH',
            '2024-01-01', '2025-12-31',
            label='网格策略', plot=True, use_sizer=False,
        )
        print(result['total_return'])
    """
    # Step 1: 包装策略, 注入记录功能
    wrapped = _wrap_strategy(strategy_class)

    # Step 2: 配置引擎
    cerebro, df = setup_cerebro(wrapped, stock_code, start_date, end_date,
                                use_sizer=use_sizer, df=df, data_class=data_class,
                                **strategy_kwargs)

    # Step 3: 打印基本信息
    if not quiet and label:
        print(f"{label} | {stock_code or ''} | {df.index[0].strftime('%Y-%m-%d')} ~ "
              f"{df.index[-1].strftime('%Y-%m-%d')} | {len(df)}个交易日")

    # Step 4: 执行回测
    results = cerebro.run()
    strat = results[0]  # Backtrader 总是返回列表, 取第一个策略

    # Step 5: 计算绩效
    m = _calc_metrics(cerebro, strat, df)

    # Step 6: 打印报告
    if not quiet:
        print(f"  总收益: {m['total_return']*100:+.2f}% | 年化: {m['annual_return']*100:+.2f}% | "
              f"最大回撤: {m['max_drawdown']*100:.2f}% | 夏普: {m['sharpe_ratio']:.2f} | "
              f"卡玛: {m['calmar_ratio']:.2f}")
        print(f"  交易: {m['total_trades']}次 | 胜率: {m['win_rate']*100:.1f}% | "
              f"盈亏比: {m['profit_loss_ratio']:.2f} | 利润因子: {m['profit_factor']:.2f} | "
              f"最大连亏: {m['max_consecutive_losses']}次")

    result = {**m, 'df': df, 'trades': strat._trade_log, 'nav': strat._nav_log}

    # Step 7: 可选生成图表
    if plot:
        chart_name = label or strategy_class.__name__
        plot_backtest(result, stock_code or '', chart_name)

    return result


# ============================================================
# 可视化图表
# ============================================================

def plot_backtest(result, stock_code='', title=''):
    """
    绘制回测结果的三合一图表。

    图表布局 (上中下三张子图):
      上图: K线收盘价曲线 + 买卖点标记 (红色三角=买入, 绿色三角=卖出)
           右上角显示绩效指标汇总框
      中图: 策略净值曲线 (初始=1.0) + 买入持有基准曲线
      下图: 回撤曲线 (从峰值回落的百分比)

    参数:
        result:     run_and_report() 返回的结果字典
        stock_code: 股票代码, 显示在图表标题中
        title:      图表标题
    """
    import matplotlib.pyplot as plt
    import matplotlib
    # matplotlib 中文字体配置: SimHei 为黑体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    os.makedirs('outputs', exist_ok=True)

    df = result['df']
    trades = result.get('trades', [])
    nav_data = result.get('nav', [])

    if not nav_data:
        print("没有净值数据，跳过绘图")
        return

    # ---- 准备净值数据 ----
    nav_df = pd.DataFrame(nav_data)
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df.set_index('date', inplace=True)
    nav_df['nav_pct'] = nav_df['nav'] / INITIAL_CASH  # 归一化到初始净值 = 1.0
    nav_df['peak'] = nav_df['nav'].cummax()            # 累计峰值
    nav_df['drawdown'] = (nav_df['nav'] - nav_df['peak']) / nav_df['peak'] * 100  # 回撤百分比

    # 买入持有基准: 从初始价格归一化
    close_start = float(df['close'].iloc[0])
    benchmark = df['close'] / close_start

    # ---- 分离买卖点 ----
    buy_dates = [t['date'] for t in trades if t['type'] == 'BUY']
    buy_prices = [t['price'] for t in trades if t['type'] == 'BUY']
    sell_dates = [t['date'] for t in trades if t['type'] == 'SELL']
    sell_prices = [t['price'] for t in trades if t['type'] == 'SELL']

    m = result
    # 三子图: 高度比例为 3:2:1
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                         gridspec_kw={'height_ratios': [3, 2, 1]})

    # ---- 上图: 价格 + 买卖点 ----
    ax1.plot(df.index, df['close'], 'gray', linewidth=1, alpha=0.8, label='收盘价')
    if buy_dates:
        ax1.scatter(buy_dates, buy_prices, color='#e74c3c', marker='^', s=80,
                    zorder=5, label=f'买入({len(buy_dates)}次)')
    if sell_dates:
        ax1.scatter(sell_dates, sell_prices, color='#2ecc71', marker='v', s=80,
                    zorder=5, label=f'卖出({len(sell_dates)}次)')
    ax1.set_ylabel('价格')
    ax1.set_title(f'{title}  {stock_code}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 右上角绩效摘要框
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

    # ---- 中图: 净值曲线 ----
    ax2.plot(nav_df.index, nav_df['nav_pct'], '#2980b9', linewidth=1.5, label='策略净值')
    ax2.plot(benchmark.index, benchmark, 'gray', linewidth=1, alpha=0.6, label='买入持有')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)  # 初始净值参考线
    ax2.set_ylabel('净值 (初始=1.0)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ---- 下图: 回撤曲线 ----
    ax3.fill_between(nav_df.index, nav_df['drawdown'], 0, color='#e74c3c', alpha=0.4)
    ax3.plot(nav_df.index, nav_df['drawdown'], '#c0392b', linewidth=0.8)
    ax3.set_ylabel('回撤(%)')
    ax3.set_xlabel('日期')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片到 outputs/ 目录
    safe_name = title.replace(' ', '_').replace('/', '_')
    plot_file = os.path.join('outputs', f'{safe_name}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {plot_file}")
    plt.close()
