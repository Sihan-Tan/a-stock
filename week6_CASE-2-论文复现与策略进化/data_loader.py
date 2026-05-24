# -*- coding: utf-8 -*-
"""
数据加载与回测工具模块

本模块是整个量化回测系统的"数据管道"和"回测基础设施"。
它封装了从 MySQL 数据库读取行情数据、配置 Backtrader 引擎、
记录交易日志、计算绩效指标等一系列标准化操作。

核心功能:
  1. MLSignalData      - 自定义 Backtrader 数据类，支持 ML 预测概率作为额外数据线
  2. load_stock_data() - 从 MySQL trade_stock_daily 表加载单只股票日 K 线
  3. batch_load_daily() - 批量加载多只股票的日 K 线数据
  4. _wrap_strategy()  - 策略包装器，自动记录每笔交易和每日净值
  5. setup_cerebro()   - 统一配置 Backtrader 引擎（资金/手续费/分析器）
  6. run_and_report()  - 回测主入口：运行回测 + 打印绩效报告 + 可选绘图
  7. plot_backtest()   - 可视化回测结果（K线/买卖点/净值曲线/回撤曲线）

为什么封装这些功能？
  - 标准化回测流程，避免在多个策略中重复编写相同的 Cerebro 配置代码
  - 统一绩效计算口径，确保不同策略之间的对比是公平的
  - 自动记录交易日志，便于事后分析和归因
"""
import pandas as pd
import numpy as np
import backtrader as bt
import os
from db_config import execute_query, INITIAL_CASH, COMMISSION, POSITION_PCT


# ============================================================
# ML 信号专用 PandasData
# ============================================================

class MLSignalData(bt.feeds.PandasData):
    """
    扩展 Backtrader 的 PandasData，增加 ML 预测概率信号线

    为什么需要这个类？
      标准 PandasData 只包含 OHLCV 五条数据线。当我们用机器学习模型
      （XGBoost/MASTER Transformer）预测出每日的上涨概率后，需要将这个
      概率值作为额外的数据线传入策略。MLSignalData 正是在此场景下使用。

    属性:
        lines: 定义了 'ml_prob' 这条新数据线，策略中通过 self.data.ml_prob 访问
        params: 默认第 -1 列（最后一列）作为 ml_prob 的来源

    使用方式:
        data = MLSignalData(dataname=df_with_ml_prob)
        cerebro.adddata(data)
    """
    # 新增一条数据线，存储 ML 模型预测的上涨概率
    # Backtrader 的数据线（lines）是时间序列的核心抽象
    lines = ('ml_prob',)

    # 参数设置：告诉 Backtrader 从 DataFrame 的哪一列读取 ml_prob
    # -1 表示 DataFrame 的最后一列（约定将 ML 概率放在最后一列）
    params = (
        ('ml_prob', -1),
    )


# ============================================================
# 数据加载
# ============================================================

def load_stock_data(stock_code, start_date=None, end_date=None):
    """
    从 MySQL 加载单只股票的日 K 线数据

    参数:
        stock_code: 股票代码，格式如 '600519.SH'
            - 6 开头上交所（.SH），0/3 开头深交所（.SZ）
            - 注意：数据库中的代码格式需要与 trade_stock_daily 表一致
        start_date: 开始日期，格式 'YYYY-MM-DD'，None 表示不限
        end_date: 结束日期，格式 'YYYY-MM-DD'，None 表示不限

    返回:
        pandas DataFrame，索引为日期，列名为 open/high/low/close/volume
        数据已按日期升序排列，这是 Backtrader 对数据的基本要求

    为什么返回值需要按日期升序？
      Backtrader 的 feed 机制要求数据按时间从小到大排列，
      因为回测引擎是从左到右逐根 K 线推进的。
    """
    # 构建查询条件：使用参数化查询防止 SQL 注入
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

    # 统一列名为 Backtrader 默认的 OHLCV 命名
    # 这样可以直接传入 bt.feeds.PandasData(dataname=df) 使用
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    # 确保数值类型：数据库可能返回字符串类型，这里强制转换为 float
    # errors='coerce' 让无法转换的值变为 NaN，避免程序崩溃
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def batch_load_daily(start_date, end_date, min_bars=120):
    """
    批量加载所有股票的日 K 线（用于全市场截面分析）

    参数:
        start_date: 起始日期
        end_date: 结束日期
        min_bars: 最少交易日数量
            - 用于过滤上市时间过短或数据不完整的股票
            - 120 个交易日约等于半年，确保有足够的训练样本

    返回:
        dict {stock_code: DataFrame}，键为股票代码，值为该股票的 OHLCV DataFrame

    为什么返回值是 dict 而不是 concat 后的 DataFrame？
      不同股票的交易日期并不完全一致（停牌、涨停/跌停无交易），
      用 dict 结构保持每只股票的数据独立性，便于后续处理。
    """
    sql = """
        SELECT stock_code, trade_date, open_price, high_price, low_price,
               close_price, volume
        FROM trade_stock_daily
        WHERE trade_date >= %s AND trade_date <= %s
        ORDER BY stock_code, trade_date ASC
    """
    rows = execute_query(sql, [start_date, end_date])
    if not rows:
        return {}

    all_df = pd.DataFrame(rows)
    all_df['trade_date'] = pd.to_datetime(all_df['trade_date'])
    for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
        all_df[col] = pd.to_numeric(all_df[col], errors='coerce')

    result = {}
    for code, grp in all_df.groupby('stock_code'):
        if len(grp) < min_bars:
            continue  # 跳过数据不足的股票
        df = grp.set_index('trade_date')[['open_price', 'high_price', 'low_price',
                                           'close_price', 'volume']].copy()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        result[code] = df
    return result


def load_financial_data(report_date_min=None):
    """
    加载财务数据 (PE/ROE/毛利率等基本面指标)

    为什么需要财务数据？
      技术指标反映市场行为，财务数据反映公司基本面。
      许多量化策略会结合技术和基本面因子来提高预测稳定性。

    返回:
        DataFrame，列包含 stock_code, report_date, eps, roe, gross_margin, ...
        其中 eps(每股收益)和 roe(净资产收益率)是最常用的基本面指标
    """
    sql = """
        SELECT stock_code, report_date, eps, roe, gross_margin,
               debt_ratio, net_profit, revenue, total_assets
        FROM trade_stock_financial
        WHERE 1=1
    """
    params = []
    if report_date_min:
        sql += " AND report_date >= %s"
        params.append(report_date_min)
    sql += " ORDER BY stock_code, report_date DESC"

    rows = execute_query(sql, params)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['report_date'] = pd.to_datetime(df['report_date'])
    for col in ['eps', 'roe', 'gross_margin', 'debt_ratio', 'net_profit', 'revenue', 'total_assets']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def calc_buy_and_hold(stock_code, start_date, end_date):
    """计算区间买入持有收益率（作为策略的基准参照）"""
    try:
        df = load_stock_data(stock_code, start_date, end_date)
        if len(df) < 2:
            return None
        # 收益率 = 最后收盘价 / 最初收盘价 - 1
        return float(df['close'].iloc[-1] / df['close'].iloc[0] - 1)
    except Exception:
        return None


# ============================================================
# 策略包装器 - 自动记录交易和净值
# ============================================================

def _wrap_strategy(strategy_class):
    """
    包装策略类，自动记录交易日志和每日净值

    为什么用包装器而非在策略内部直接记录？
      1. 保持策略代码的纯净度：策略类只需关注买卖逻辑，不掺杂日志代码
      2. 统一的记录格式：所有策略使用相同的日志结构，便于后续比较分析
      3. AOP（面向切面编程）思想：通过装饰/包装在运行时动态增加横切关注点

    实现机制：
      在运行时创建一个继承自原始策略的新类，重写 notify_order 和 next 方法，
      在调用父类方法之前/之后插入日志记录逻辑。

    参数:
        strategy_class: 原始的 Backtrader 策略类

    返回:
        WrappedStrategy: 包装后的策略类（功能和原始类完全一致，只是多了日志功能）
    """
    class WrappedStrategy(strategy_class):
        def __init__(self):
            super().__init__()
            self._trade_log = []  # 交易记录列表，每条包含 {date, type, price, size}
            self._nav_log = []    # 每日净值记录列表，每条包含 {date, nav}

        def notify_order(self, order):
            """
            订单状态变化回调
            Backtrader 在订单状态发生任何变化时都会调用此方法。
            order.Completed 表示订单已完全成交。
            """
            if order.status == order.Completed:
                self._trade_log.append({
                    'date': self.data.datetime.date(0),  # 当前 K 线日期
                    'type': 'BUY' if order.isbuy() else 'SELL',
                    'price': round(order.executed.price, 2),
                    'size': abs(int(order.executed.size)),  # 成交股数（取绝对值为正数）
                })
            if hasattr(super(), 'notify_order'):
                super().notify_order(order)

        def next(self):
            """
            每根 K 线（每个交易日）调用一次
            记录当前账户净值到 _nav_log
            """
            self._nav_log.append({
                'date': self.data.datetime.date(0),
                'nav': self.broker.getvalue(),  # 当前账户总价值（现金 + 持仓市值）
            })
            super().next()

    # 保持类的元信息不变，让外界感觉不到包装的存在
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

    参数:
        strategy_class: Backtrader 策略类
        stock_code: 股票代码（当 df=None 时用于加载数据）
        start_date/end_date: 回测日期范围
        use_sizer: 是否自动添加仓位管理器（PercentSizer，按比例分配仓位）
        df: 可选，直接传入 DataFrame（优先于 stock_code+日期）
        data_class: 可选，自定义数据类（如 MLSignalData）
        **strategy_kwargs: 传递给策略类的参数

    返回:
        (cerebro, df) 元组，cerebro 是配置好的回测引擎，df 是使用的数据

    为什么把 cerebro 和 df 一起返回？
      回测结束后计算绩效指标需要 df（知道交易天数）和 cerebro（知道最终净值），
      两者都是必须的，一起返回避免调用方重复加载数据。
    """
    if df is None:
        df = load_stock_data(stock_code, start_date, end_date)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, **strategy_kwargs)

    feed_class = data_class or bt.feeds.PandasData
    cerebro.adddata(feed_class(dataname=df))

    # 设置初始资金和手续费（来自 .env 配置）
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

    if use_sizer:
        # PercentSizer：每次交易使用固定比例的资金
        # 比如 POSITION_PCT=95，就是用 95% 的可用资金买入
        cerebro.addsizer(bt.sizers.PercentSizer, percents=POSITION_PCT)

    # 添加内置分析器：
    # SharpeRatio  - 夏普比率（衡量风险调整后收益）
    # DrawDown     - 最大回撤（衡量策略的抗风险能力）
    # TradeAnalyzer - 交易分析（胜率/盈亏比/连续盈亏等）
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    return cerebro, df


# ============================================================
# 绩效计算
# ============================================================

def _calc_metrics(cerebro, strat, df):
    """
    从 Backtrader 运行结果中提取完整绩效指标

    计算以下指标：
      - 总收益率 / 年化收益率
      - 夏普比率 / 卡玛比率（Calmar Ratio）
      - 最大回撤 / 回撤持续时间
      - 交易统计（总交易次数/胜率/盈亏比/利润因子）
      - 最大连续亏损次数

    卡玛比率 = 年化收益率 / 最大回撤
    它衡量的是"每承受一单位回撤能获得多少收益"。
    相比夏普比率，卡玛比率对投资者更直观——最大回撤是投资者最关心的风险指标。

    参数:
        cerebro: 运行后的 Cerebro 实例
        strat: 运行后返回的策略实例（含分析器结果）
        df: 原始数据 DataFrame

    返回:
        dict，包含所有绩效指标
    """
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH

    # 年化收益率计算
    # 假设一年 252 个交易日（A股实际约 242-250 天）
    trading_days = len(df)
    years = trading_days / 252
    if years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = total_return

    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0

    # 手动计算最大回撤：遍历每日净值，跟踪历史峰值
    # 为什么不用 DrawDown 分析器的结果？
    # 因为 DrawDown 分析器的计算方式和我们的需求略有差异，
    # 手动计算让我们能同时获取回撤持续天数（max_dd_len）
    max_drawdown = 0.0
    max_dd_len = 0
    nav_log = getattr(strat, '_nav_log', [])
    if nav_log:
        navs = [x['nav'] for x in nav_log]
        peak = navs[0]
        dd_len = 0
        for v in navs:
            if v > peak:
                peak = v
                dd_len = 0
            else:
                dd_len += 1
                if peak > 0 and v > 0:
                    dd_pct = (peak - v) / peak
                    max_drawdown = max(max_drawdown, min(dd_pct, 1.0))
                max_dd_len = max(max_dd_len, dd_len)
    if not nav_log:
        dd = strat.analyzers.drawdown.get_analysis()
        bt_dd = dd.get('max', {}).get('drawdown', 0) / 100
        max_drawdown = min(bt_dd, 1.0)
        max_dd_len = dd.get('max', {}).get('len', 0)

    # 卡玛比率：年化收益 / 最大回撤
    # 卡玛比率 > 1 表示策略的风险调整后收益较好
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # 交易分析器提供了详细的交易统计
    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get('total', {}).get('total', 0)
    won_trades = ta.get('won', {}).get('total', 0)
    lost_trades = ta.get('lost', {}).get('total', 0)
    win_rate = won_trades / total_trades if total_trades > 0 else 0

    avg_win = ta.get('won', {}).get('pnl', {}).get('average', 0) or 0
    avg_loss = ta.get('lost', {}).get('pnl', {}).get('average', 0) or 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    gross_profit = ta.get('won', {}).get('pnl', {}).get('total', 0) or 0
    gross_loss = ta.get('lost', {}).get('pnl', {}).get('total', 0) or 0
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

    max_consecutive_losses = _calc_max_consecutive_losses(ta)

    # 期望值（Expected Value）：每笔交易平均能赚多少
    # EV = P(win) * avg_win + P(loss) * avg_loss
    # 正的期望值是量化策略长期盈利的根本保证
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
    从 TradeAnalyzer 结果中提取最大连续亏损次数

    连续亏损次数是衡量策略"抗揍"能力的重要指标：
    - 连续亏损 10 次，资金可能只剩 60%（取决于单次亏损幅度）
    - 连续亏损次数越小，策略的实际可操作性越强
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
    运行回测并打印绩效报告（回测的"一键启动"入口）

    这是整个回测系统的调用入口。它封装了从数据加载到结果输出的完整流程：
      包装策略 -> 配置引擎 -> 运行回测 -> 计算绩效 -> 打印报告 -> 可选绘图

    参数:
        strategy_class: 策略类
        stock_code: 股票代码
        start_date/end_date: 回测时间范围
        label: 回测标签（用于区分多次回测结果）
        plot: 是否生成可视化图表
        quiet: 是否静默模式（不打印输出）
        use_sizer: 是否使用仓位管理器
        df: 直接传入数据（替代从数据库加载）
        data_class: 自定义数据类
        **strategy_kwargs: 策略参数

    返回:
        dict，包含绩效指标、交易日志、净值曲线和数据
    """
    wrapped = _wrap_strategy(strategy_class)
    cerebro, df = setup_cerebro(wrapped, stock_code, start_date, end_date,
                                use_sizer=use_sizer, df=df, data_class=data_class,
                                **strategy_kwargs)

    if not quiet and label:
        print(f"{label} | {stock_code or ''} | {df.index[0].strftime('%Y-%m-%d')} ~ "
              f"{df.index[-1].strftime('%Y-%m-%d')} | {len(df)}个交易日")

    results = cerebro.run()
    strat = results[0]  # run() 返回列表，取第一个（只有一个策略）
    m = _calc_metrics(cerebro, strat, df)

    if not quiet:
        print(f"  总收益: {m['total_return']*100:+.2f}% | 年化: {m['annual_return']*100:+.2f}% | "
              f"最大回撤: {m['max_drawdown']*100:.2f}% | 夏普: {m['sharpe_ratio']:.2f} | "
              f"卡玛: {m['calmar_ratio']:.2f}")
        print(f"  交易: {m['total_trades']}次 | 胜率: {m['win_rate']*100:.1f}% | "
              f"盈亏比: {m['profit_loss_ratio']:.2f} | 利润因子: {m['profit_factor']:.2f} | "
              f"最大连亏: {m['max_consecutive_losses']}次")

    result = {**m, 'df': df, 'trades': strat._trade_log, 'nav': strat._nav_log}

    if plot:
        chart_name = label or strategy_class.__name__
        plot_backtest(result, stock_code or '', chart_name)

    return result


# ============================================================
# 可视化图表
# ============================================================

def plot_backtest(result, stock_code='', title=''):
    """
    绘制回测结果三合一图表

    图表排版（上中下三部分）：
      1. 上图：K 线 + 买卖点标记
         - 灰色线为收盘价
         - 红色三角为买入点（^）
         - 绿色三角为卖出点（v）
         - 右上角显示绩效摘要
      2. 中图：净值曲线 vs 买入持有基准线
         - 蓝色线为策略净值
         - 灰色线为被动买入持有收益
      3. 下图：回撤曲线
         - 红色填充区域直观展示回撤幅度

    参数:
        result: _calc_metrics 返回的字典（含 df, trades, nav）
        stock_code: 股票代码（用于图表标题）
        title: 图表标题
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']         # 设置中文字体
    matplotlib.rcParams['axes.unicode_minus'] = False            # 解决负号显示问题

    os.makedirs('outputs', exist_ok=True)                       # 创建输出目录

    df = result['df']
    trades = result.get('trades', [])
    nav_data = result.get('nav', [])

    if not nav_data:
        print("没有净值数据，跳过绘图")
        return

    nav_df = pd.DataFrame(nav_data)
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df.set_index('date', inplace=True)
    nav_df['nav_pct'] = nav_df['nav'] / INITIAL_CASH            # 归一化到初始资金 1.0
    nav_df['peak'] = nav_df['nav'].cummax()                     # 历史峰值（用于计算回撤）
    nav_df['drawdown'] = (nav_df['nav'] - nav_df['peak']) / nav_df['peak'] * 100

    # 买入持有基准：简单持有不动，收益完全跟随股价涨跌
    close_start = float(df['close'].iloc[0])
    benchmark = df['close'] / close_start

    buy_dates = [t['date'] for t in trades if t['type'] == 'BUY']
    buy_prices = [t['price'] for t in trades if t['type'] == 'BUY']
    sell_dates = [t['date'] for t in trades if t['type'] == 'SELL']
    sell_prices = [t['price'] for t in trades if t['type'] == 'SELL']

    m = result
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                         gridspec_kw={'height_ratios': [3, 2, 1]})

    # 上图：K线和买卖点
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

    # 中图：净值曲线
    ax2.plot(nav_df.index, nav_df['nav_pct'], '#2980b9', linewidth=1.5, label='策略净值')
    ax2.plot(benchmark.index, benchmark, 'gray', linewidth=1, alpha=0.6, label='买入持有')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
    ax2.set_ylabel('净值 (初始=1.0)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 下图：回撤曲线
    ax3.fill_between(nav_df.index, nav_df['drawdown'], 0, color='#e74c3c', alpha=0.4)
    ax3.plot(nav_df.index, nav_df['drawdown'], '#c0392b', linewidth=0.8)
    ax3.set_ylabel('回撤(%)')
    ax3.set_xlabel('日期')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    safe_name = title.replace(' ', '_').replace('/', '_')
    plot_file = os.path.join('outputs', f'{safe_name}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {plot_file}")
    plt.close()
