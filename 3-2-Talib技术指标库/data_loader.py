# -*- coding: utf-8 -*-
"""
数据加载与回测工具模块 -- 整个项目的核心基础设施

功能:
  - 从 MySQL 读取 K 线数据 (trade_stock_daily 表)
  - 统一配置 Backtrader Cerebro 引擎（初始资金/手续费/仓位来自 .env）
  - 自动包装策略类，记录交易日志和每日净值
  - 计算完整绩效指标（收益、回撤、夏普、卡玛、胜率、盈亏比、利润因子等）
  - 生成可视化图表（K线+买卖点、净值曲线、回撤曲线）

设计思路:
  本模块将所有回测的"样板代码"集中管理，策略文件只需关注策略逻辑本身，
  通过调用 run_and_report() 即可完成从数据加载到结果输出的全流程。

依赖:
  - db_config.py: 数据库配置与查询
  - backtrader: 回测引擎
  - pandas/numpy: 数据处理
  - matplotlib: 可视化输出
"""
import pandas as pd
import numpy as np
import backtrader as bt
import os
from db_config import execute_query, INITIAL_CASH, COMMISSION, POSITION_PCT


# ============================================================
# 数据加载 -- 从 MySQL 数据库读取日 K 线数据
# ============================================================

def load_stock_data(stock_code, start_date=None, end_date=None):
    """
    从 MySQL 加载单只股票的日 K 线数据

    MySQL 的 trade_stock_daily 表存储原始字段名（open_price, close_price 等），
    而 Backtrader 要求列名为 open/high/low/close/volume，这里做统一映射。

    参数:
        stock_code: 股票代码，如 '600519.SH'（茅台）、'000001.SZ'（平安银行）
        start_date: 开始日期，如 '2024-01-01'（含边界）
        end_date:   结束日期，如 '2025-12-31'（含边界）

    返回:
        pandas DataFrame，索引为日期（DatetimeIndex），
        列为标准化的 open/high/low/close/volume

    异常:
        ValueError: 数据库中没有该股票数据
    """
    # 动态构建 SQL 查询条件，避免不必要的 WHERE 子句
    conditions = ["stock_code = %s"]   # 主条件：股票代码匹配
    params = [stock_code]

    if start_date:
        conditions.append("trade_date >= %s")   # 起始日期过滤
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= %s")   # 截止日期过滤
        params.append(end_date)

    # SQL 查询：按交易日升序排列，保证后续技术指标计算的时间顺序正确
    sql = f"""
        SELECT trade_date, open_price, high_price, low_price, close_price, volume
        FROM trade_stock_daily
        WHERE {' AND '.join(conditions)}
        ORDER BY trade_date ASC
    """
    rows = execute_query(sql, params)
    if not rows:
        # 给出明确提示：可能是数据库无数据，需要先运行行情数据采集脚本
        raise ValueError(f"没有找到 {stock_code} 的数据，请检查数据库或先运行数据采集")

    # 转换为 DataFrame 并标准化列名
    df = pd.DataFrame(rows)
    df['trade_date'] = pd.to_datetime(df['trade_date'])  # 字符串转日期类型
    df.set_index('trade_date', inplace=True)              # 以日期为索引，便于时间序列分析
    df.columns = ['open', 'high', 'low', 'close', 'volume']  # 重命名为 Backtrader 标准列名
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # 确保数值类型，无效值转 NaN
    return df


def list_available_symbols(start_date=None, end_date=None):
    """
    查询数据库中有数据的股票/ETF 代码列表

    用于策略的批量回测或形态选股雷达的全市场扫描。

    参数:
        start_date/end_date: 日期范围过滤

    返回:
        list[str] 股票代码列表，如 ['000001.SZ', '600519.SH', ...]
    """
    conditions, params = [], []
    if start_date:
        conditions.append("trade_date >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= %s")
        params.append(end_date)
    sql = "SELECT DISTINCT stock_code FROM trade_stock_daily"
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)
    sql += " ORDER BY stock_code"
    rows = execute_query(sql, params)
    return [r['stock_code'] for r in rows]


def get_symbol_data_summary(codes, start_date=None, end_date=None):
    """
    查询指定标的在各年份的数据天数

    用于了解数据质量：某只股票在哪些年份有数据、每年多少交易日。
    如果某年数据天数明显偏少（<200天），说明数据库可能缺失了部分数据。

    参数:
        codes: 标的代码列表，如 ['600519.SH', '000001.SZ']
        start_date/end_date: 日期范围

    返回:
        dict {code: {'total': N, 'yearly': {2022: N, 2023: N, ...}}}
        total 为总数据条数，yearly 为各年份的交易日计数
    """
    if not codes:
        return {}
    # 动态构建 IN 子句的参数占位符
    placeholders = ','.join(['%s'] * len(codes))
    conditions = [f"stock_code IN ({placeholders})"]
    params = list(codes)
    if start_date:
        conditions.append("trade_date >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("trade_date <= %s")
        params.append(end_date)
    # YEAR() 函数提取年份，按股票+年份分组统计
    sql = (f"SELECT stock_code, YEAR(trade_date) AS yr, COUNT(*) AS cnt "
           f"FROM trade_stock_daily WHERE {' AND '.join(conditions)} "
           f"GROUP BY stock_code, YEAR(trade_date)")
    rows = execute_query(sql, params)
    result = {}
    for r in rows or []:
        code = r['stock_code']
        if code not in result:
            result[code] = {'total': 0, 'yearly': {}}
        result[code]['yearly'][r['yr']] = r['cnt']
        result[code]['total'] += r['cnt']
    return result


def calc_buy_and_hold(stock_code, start_date, end_date):
    """
    计算区间买入持有收益率（基准收益）

    用于与策略收益对比，判断策略是否创造了超额收益。
    如果策略收益低于买入持有，说明这个策略在当前标的上没有价值。

    参数:
        stock_code: 标的代码
        start_date/end_date: 持有区间

    返回:
        float 收益率（如 0.15 表示 +15%），数据不足返回 None
    """
    try:
        df = load_stock_data(stock_code, start_date, end_date)
        if len(df) < 2:
            return None
        # 收益率 = 最终收盘价 / 最初收盘价 - 1
        return float(df['close'].iloc[-1] / df['close'].iloc[0] - 1)
    except Exception:
        return None


def get_instrument_names(codes):
    """
    获取标的显示名称（股票/ETF）。

    学员数据库 wucai_trade_charles 中没有 trade_stock_status 表，
    无法从数据库查询股票中文名称，因此直接返回代码本身作为名称。
    如果在生产环境中使用，可替换为从聚宽/akshare 等数据源获取名称。

    参数:
        codes: 标的代码列表

    返回:
        dict {code: display_name}
    """
    if not codes:
        return {}
    return {code: code for code in codes}


# ============================================================
# 策略包装器 -- 自动记录交易和净值
# ============================================================

def _wrap_strategy(strategy_class):
    """
    包装任意策略类，自动记录交易日志和每日净值

    原理（装饰器模式）:
      动态创建一个继承自原策略类的新类，重写 notify_order() 和 next() 方法，
      在不修改原策略代码的前提下注入日志记录功能。

    为什么需要这个方法？
      Backtrader 的策略类在 cerebro.run() 执行时才会实例化，
      无法在实例化前注入日志逻辑。通过动态子类化，我们可以在不修改
      策略代码的情况下统一记录交易数据。

    记录数据:
      - _trade_log: list[dict] 每笔交易的日期、类型(BUY/SELL)、价格、数量
      - _nav_log:   list[dict] 每日净值

    参数:
        strategy_class: Backtrader 策略类（不是实例）

    返回:
        WrappedStrategy: 功能增强后的策略类（保留了原类的 __name__ 等元信息）
    """
    class WrappedStrategy(strategy_class):
        def __init__(self):
            super().__init__()
            self._trade_log = []   # 交易日志：记录每笔买卖的时间、价格、数量
            self._nav_log = []     # 净值日志：记录每日账户净值

        def notify_order(self, order):
            """
            订单完成回调 -- 自动记录成交信息

            Backtrader 在订单状态变化时调用此方法。
            我们只关心成交（Completed）的订单。

            order.isbuy() 判断买卖方向
            order.executed.price 实际成交价（可能和发单价格不同）
            order.executed.size 成交数量（正数为买入，负数为卖出）
            """
            if order.status == order.Completed:
                self._trade_log.append({
                    'date': self.data.datetime.date(0),  # 当前 bar 的日期
                    'type': 'BUY' if order.isbuy() else 'SELL',
                    'price': round(order.executed.price, 2),
                    'size': abs(int(order.executed.size)),  # size 取绝对值
                })
            # 如果原策略有 notify_order，仍然调用它，不破坏原逻辑
            if hasattr(super(), 'notify_order'):
                super().notify_order(order)

        def next(self):
            """
            每根 K 线回调 -- 自动记录当日净值

            self.broker.getvalue() 返回当前账户总资产（现金+持仓市值）。
            记录在 _nav_log 中，后续用于绘制净值曲线和计算最大回撤。
            """
            self._nav_log.append({
                'date': self.data.datetime.date(0),  # 当前 bar 的日期
                'nav': self.broker.getvalue(),        # 当日账户总资产
            })
            # 调用原策略的 next()，保证策略逻辑正常执行
            super().next()

    # 保持包装后的类名与原始类一致，避免影响日志输出和调试
    WrappedStrategy.__name__ = strategy_class.__name__
    WrappedStrategy.__qualname__ = strategy_class.__qualname__
    WrappedStrategy.__module__ = strategy_class.__module__
    return WrappedStrategy


# ============================================================
# Cerebro 配置 -- 创建并配置回测引擎
# ============================================================

def setup_cerebro(strategy_class, stock_code, start_date=None, end_date=None, **strategy_kwargs):
    """
    创建并配置好 Backtrader Cerebro 引擎

    这是从"数据"到"引擎"的一站式配置函数，封装了以下步骤：
      1. 加载股票数据
      2. 创建 Cerebro 引擎
      3. 添加策略（支持传递参数）
      4. 添加数据源
      5. 设置初始资金、手续费、仓位
      6. 添加分析器（夏普比率、最大回撤、交易统计）

    参数:
        strategy_class: Backtrader 策略类
        stock_code: 股票代码
        start_date/end_date: 数据范围
        **strategy_kwargs: 传递给策略的关键字参数（即策略的 params）

    返回:
        (cerebro, df) 二元组
        cerebro: 配置好的 Backtrader Cerebro 实例
        df: 加载的 DataFrame（用于后续绩效计算和绘图）
    """
    df = load_stock_data(stock_code, start_date, end_date)

    cerebro = bt.Cerebro()
    # **strategy_kwargs 解包后传给策略的 params，例如 RSIStrategy(period=14)
    cerebro.addstrategy(strategy_class, **strategy_kwargs)
    # PandasData 适配器将 pandas DataFrame 转换为 Backtrader 内部数据格式
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.broker.setcash(INITIAL_CASH)                                    # 设置初始资金
    cerebro.broker.setcommission(commission=COMMISSION)                     # 设置手续费率
    cerebro.addsizer(bt.sizers.PercentSizer, percents=POSITION_PCT)        # 设置每次开仓比例

    # 添加内置分析器，用于后续计算绩效指标
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)   # 夏普比率（无风险利率2%）
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')                       # 最大回撤
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')                    # 交易统计

    return cerebro, df


# ============================================================
# 绩效计算 -- 从回测结果中提取完整绩效指标
# ============================================================

def _calc_metrics(cerebro, strat, df):
    """
    从 Backtrader 回测结果中提取完整的绩效指标体系

    计算指标：
      核心指标: 总收益、年化收益、最大回撤、夏普比率
      辅助指标: 卡玛比率、交易次数、胜率、盈亏比、利润因子
      风险指标: 最大连续亏损次数、期望值

    参数:
        cerebro: 运行完成后的 Cerebro 实例
        strat: 运行完成后的策略实例（通过 results[0] 获取）
        df: 原始 K 线 DataFrame

    返回:
        dict 包含上述所有绩效指标
    """
    final_value = cerebro.broker.getvalue()            # 最终账户资产
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH  # 总收益率

    # ---- 年化收益率 ----
    # A 股每年约 252 个交易日，用实际交易天数折算年化
    trading_days = len(df)
    years = trading_days / 252
    if years > 0 and total_return > -1:
        # 年化收益率 = (1+总收益)^(1/年数) - 1，考虑了复利效应
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = total_return  # 数据不足或亏损超过100%时直接用总收益

    # ---- 夏普比率 ----
    # 衡量每承担一单位风险获得多少超额收益
    # SharpRatio > 1 良好，> 2 优秀，> 3 极佳
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0

    # ---- 最大回撤 ----
    # 衡量策略在最坏情况下从峰值下跌了多少
    # 最大回撤 < 20% 良好，< 10% 优秀
    dd = strat.analyzers.drawdown.get_analysis()
    max_drawdown = dd.get('max', {}).get('drawdown', 0) / 100  # 转为小数
    max_dd_len = dd.get('max', {}).get('len', 0)  # 回撤持续天数

    # ---- 卡玛比率 = 年化收益 / 最大回撤 ----
    # 卡玛 > 1 良好，> 2 优秀。与夏普互补，更关注下行风险。
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # ---- 交易统计 ----
    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get('total', {}).get('total', 0)       # 总交易次数
    won_trades = ta.get('won', {}).get('total', 0)           # 盈利交易次数
    lost_trades = ta.get('lost', {}).get('total', 0)         # 亏损交易次数
    win_rate = won_trades / total_trades if total_trades > 0 else 0  # 胜率

    # ---- 盈亏比 = 平均盈利 / 平均亏损 ----
    avg_win = ta.get('won', {}).get('pnl', {}).get('average', 0) or 0
    avg_loss = ta.get('lost', {}).get('pnl', {}).get('average', 0) or 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # ---- 利润因子 = 总盈利 / 总亏损的绝对值 ----
    # 利润因子 > 1 说明整体盈利，> 2 说明盈利能力很强
    gross_profit = ta.get('won', {}).get('pnl', {}).get('total', 0) or 0
    gross_loss = ta.get('lost', {}).get('pnl', {}).get('total', 0) or 0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

    # ---- 最大连续亏损次数 ----
    # 衡量策略的"寒冬"有多长，对实盘心理承受能力有重要参考价值
    max_consecutive_losses = _calc_max_consecutive_losses(ta)

    # ---- 期望值 = 胜率 * 平均盈利 + 败率 * 平均亏损 ----
    # 期望值 > 0 说明策略长期有效，越大越好
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

    TradeAnalyzer 的 streak 字段记录了盈利/亏损的连续出现次数。
    连续亏损次数对资金管理至关重要——它决定了需要预留多少安全垫。

    参数:
        ta: TradeAnalyzer 的分析结果字典

    返回:
        int 最大连续亏损次数
    """
    streak = ta.get('streak', {})
    lost_streak = streak.get('lost', {})
    return lost_streak.get('longest', 0) if lost_streak else 0


# ============================================================
# 运行回测 + 输出报告 -- 一站式接口
# ============================================================

def run_and_report(strategy_class, stock_code, start_date=None, end_date=None,
                   label='', plot=False, quiet=False, **strategy_kwargs):
    """
    运行回测并打印绩效报告 -- 最常用的顶层接口

    这是整个模块的核心封装，调用方只需提供策略类、股票代码和日期范围，
    即可完成"数据加载 -> 引擎配置 -> 策略包装 -> 回测运行 -> 绩效计算 ->
    结果输出 -> 图表保存"的全流程。

    使用示例:
        run_and_report(MyStrategy, '600519.SH', '2024-01-01', '2025-12-31',
                       label='我的策略', plot=True)

    参数:
        strategy_class: 策略类（必须是 bt.Strategy 的子类）
        stock_code: 股票代码
        start_date/end_date: 数据范围
        label: 显示名称（用于图表标题和输出文件命名）
        plot: 是否输出可视化图表到 outputs/ 目录
        quiet: 为 True 时不打印，仅返回结果（用于批量回测时降低输出噪音）
        **strategy_kwargs: 传递给策略的额外参数

    返回:
        dict 包含绩效指标、K线数据、交易记录和净值记录
    """
    # 第1步：包装策略类，注入交易/净值记录功能
    wrapped = _wrap_strategy(strategy_class)
    # 第2步：加载数据并配置 Cerebro 引擎
    cerebro, df = setup_cerebro(wrapped, stock_code, start_date, end_date, **strategy_kwargs)

    if not quiet and label:
        # 打印回测基本信息：标的、时间范围、交易日数
        print(f"{label} | {stock_code} | {df.index[0].strftime('%Y-%m-%d')} ~ "
              f"{df.index[-1].strftime('%Y-%m-%d')} | {len(df)}个交易日")

    # 第3步：运行回测
    results = cerebro.run()
    strat = results[0]        # run() 返回列表，取第一个（只有一个策略）
    # 第4步：计算绩效指标
    m = _calc_metrics(cerebro, strat, df)

    if not quiet:
        # 第5步：输出核心绩效指标
        print(f"  总收益: {m['total_return']*100:+.2f}% | 年化: {m['annual_return']*100:+.2f}% | "
              f"最大回撤: {m['max_drawdown']*100:.2f}% | 夏普: {m['sharpe_ratio']:.2f} | "
              f"卡玛: {m['calmar_ratio']:.2f}")
        print(f"  交易: {m['total_trades']}次 | 胜率: {m['win_rate']*100:.1f}% | "
              f"盈亏比: {m['profit_loss_ratio']:.2f} | 利润因子: {m['profit_factor']:.2f} | "
              f"最大连亏: {m['max_consecutive_losses']}次")

    # 第6步：组装完整结果（绩效 + 原始数据 + 交易日志 + 净值日志）
    result = {**m, 'df': df, 'trades': strat._trade_log, 'nav': strat._nav_log}

    if plot:
        # 第7步：生成图表
        chart_name = label or strategy_class.__name__
        plot_backtest(result, stock_code, chart_name)

    return result


# ============================================================
# 可视化图表 -- 三合一：K线+买卖点、净值曲线、回撤曲线
# ============================================================

def plot_backtest(result, stock_code='', title=''):
    """
    绘制回测结果图表（三面板合一）

    图表布局：
      上图: K线收盘价走势 + 买入/卖出标记点
      中图: 策略净值曲线 vs 买入持有基准线
      下图: 回撤曲线（用红色填充显示亏损幅度）

    参数:
        result: run_and_report 的返回字典
        stock_code: 股票代码（用于标题）
        title: 图表标题
    """
    import matplotlib.pyplot as plt
    import matplotlib
    # 设置中文字体，SimHei（黑体）在 Windows 下可用
    # 如果 Linux/Mac 没有 SimHei，可改为 'WenQuanYi Micro Hei' 或 'Arial Unicode MS'
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    os.makedirs('outputs', exist_ok=True)  # 创建输出目录（如果不存在）

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
    nav_df['nav_pct'] = nav_df['nav'] / INITIAL_CASH   # 净值归一化（初始=1.0）
    nav_df['peak'] = nav_df['nav'].cummax()             # 计算历史最高净值（用于回撤计算）
    nav_df['drawdown'] = (nav_df['nav'] - nav_df['peak']) / nav_df['peak'] * 100  # 回撤百分比

    # ---- 买入持有基准 ----
    close_start = float(df['close'].iloc[0])
    benchmark = df['close'] / close_start  # 也归一化到 1.0，方便对比

    # ---- 分离买入点和卖出点 ----
    buy_dates = [t['date'] for t in trades if t['type'] == 'BUY']
    buy_prices = [t['price'] for t in trades if t['type'] == 'BUY']
    sell_dates = [t['date'] for t in trades if t['type'] == 'SELL']
    sell_prices = [t['price'] for t in trades if t['type'] == 'SELL']

    m = result
    # 三面板：3:2:1 高度比例，K线图最重要占最多空间
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                         gridspec_kw={'height_ratios': [3, 2, 1]})

    # ================================================================
    # 上图: K线(收盘价) + 买卖点
    # ================================================================
    ax1.plot(df.index, df['close'], 'gray', linewidth=1, alpha=0.8, label='收盘价')
    if buy_dates:
        # 买入点用红色上三角标记（尖峰向上 = 做多入场）
        ax1.scatter(buy_dates, buy_prices, color='#e74c3c', marker='^', s=80,
                    zorder=5, label=f'买入({len(buy_dates)}次)')
    if sell_dates:
        # 卖出点用绿色下三角标记（尖峰向下 = 平仓离场）
        ax1.scatter(sell_dates, sell_prices, color='#2ecc71', marker='v', s=80,
                    zorder=5, label=f'卖出({len(sell_dates)}次)')
    ax1.set_ylabel('价格')
    ax1.set_title(f'{title}  {stock_code}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 右侧文本框显示关键绩效指标
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

    # ================================================================
    # 中图: 净值曲线 vs 买入持有基准
    # ================================================================
    ax2.plot(nav_df.index, nav_df['nav_pct'], '#2980b9', linewidth=1.5, label='策略净值')
    ax2.plot(benchmark.index, benchmark, 'gray', linewidth=1, alpha=0.6, label='买入持有')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)  # 基准线 y=1.0
    ax2.set_ylabel('净值 (初始=1.0)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ================================================================
    # 下图: 回撤曲线
    # ================================================================
    ax3.fill_between(nav_df.index, nav_df['drawdown'], 0, color='#e74c3c', alpha=0.4)  # 红色半透明填充
    ax3.plot(nav_df.index, nav_df['drawdown'], '#c0392b', linewidth=0.8)               # 回撤线
    ax3.set_ylabel('回撤(%)')
    ax3.set_xlabel('日期')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图片，文件名用 label 替换特殊字符
    safe_name = title.replace(' ', '_').replace('/', '_')
    plot_file = os.path.join('outputs', f'{safe_name}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {plot_file}")
    plt.close()  # 关闭图形释放内存
