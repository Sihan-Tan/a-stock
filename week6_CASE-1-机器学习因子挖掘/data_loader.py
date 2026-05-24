# -*- coding: utf-8 -*-
"""
数据加载与回测工具模块

本模块是整个项目的"数据管道"核心，负责三层任务：
  层1 - 数据获取：从 MySQL 数据库读取日K线数据和财务数据
  层2 - 回测框架：统一配置 Backtrader 引擎（资金、手续费、仓位），自动记录交易日志
  层3 - 绩效评估：计算完整的绩效指标（夏普率、最大回撤、卡玛比等）并生成可视化图表

为什么需要这个模块？
  避免在每个策略脚本中重复编写数据库查询和回测配置代码。
  所有回测相关的标准化流程（如净值记录、绩效计算、绘图）都在这里一次性实现。

依赖关系:
  - db_config.py: 数据库连接配置和查询工具
  - feature_engine.py: 被 1-4 号脚本调用，本模块不直接依赖

Backtrader 核心概念:
  - Cerebro: 回测引擎大脑，调度数据流、策略、分析器
  - Data Feed: 数据源，将 DataFrame 喂给引擎
  - Strategy: 交易策略，包含买卖逻辑
  - Analyzer: 分析器，事后计算绩效指标
  - Sizer: 仓位管理器，决定每次交易多少股
"""
import pandas as pd
import numpy as np
import backtrader as bt
import os
from db_config import execute_query, INITIAL_CASH, COMMISSION, POSITION_PCT


# ============================================================
# ML 信号专用数据源
# ============================================================

class MLSignalData(bt.feeds.PandasData):
    """
    扩展自 backtrader 的标准 PandasData，增加一条 ML 预测概率线。

    为什么需要这个类？
      标准 PandasData 只支持 open/high/low/close/volume 等基础字段。
      当我们用机器学习模型预测出"上涨概率"后，需要把它作为一条额外的
      数据线传递给策略，让策略根据这个概率值来做交易决策。

    用法：
      data = MLSignalData(dataname=df)  # df 需包含 ml_prob 列

    扩展的线（lines）：
      - ml_prob: ML 模型输出的上涨概率（0~1），作为交易的辅助信号
    """
    # lines 定义新增的数据线，Backtrader 会自动创建 ml_prob 属性
    lines = ('ml_prob',)

    # params 定义默认值，(-1) 表示如果数据中不包含该列则自动忽略
    params = (
        ('ml_prob', -1),
    )


# ============================================================
# 数据加载函数
# ============================================================

def load_stock_data(stock_code, start_date=None, end_date=None):
    """
    从 MySQL 数据库加载单只股票的日K线数据。

    参数:
        stock_code: 股票代码，如 '600519.SH'。
                    注意 A 股代码需要带后缀：.SH 上海，.SZ 深圳
        start_date: 起始日期，格式 'YYYY-MM-DD'，None 表示不限起始
        end_date:   结束日期，格式 'YYYY-MM-DD'，None 表示不限结束

    返回:
        pandas DataFrame，索引为日期，列名为 open/high/low/close/volume

    数据表 trade_stock_daily 的字段：
      stock_code  - 股票代码
      trade_date  - 交易日
      open_price  - 开盘价
      high_price  - 最高价
      low_price   - 最低价
      close_price - 收盘价（最常用，计算收益率和标签的基础）
      volume      - 成交量

    为什么需要动态拼接 WHERE 条件？
      让调用者可以灵活筛选时间范围，而不需要写多个版本的 SQL 查询。
    """
    # 动态构建查询条件：将非空的参数加入条件列表
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
        # 抛出异常而非返回空 DataFrame：让调用者明确知道数据缺失
        raise ValueError(f"没有找到 {stock_code} 的数据，请检查数据库或先运行数据采集")

    # 将数据库查询结果（字典列表）转为 pandas DataFrame
    df = pd.DataFrame(rows)
    # 日期列转为 datetime 类型，方便按时间索引和绘图
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    # 将日期列设为索引 -- Backtrader 要求数据以日期为索引
    df.set_index('trade_date', inplace=True)
    # 统一列名为小写英文，方便后续代码处理（backtrader 要求的标准命名）
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    # 确保所有数值列都是 float 类型，避免因数据库返回字符串类型导致计算错误
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def batch_load_daily(start_date, end_date, min_bars=120):
    """
    批量加载所有股票的日K线数据。

    在实战中，我们通常需要同时分析数百只股票。这个函数一次性加载
    所有股票的数据，并自动过滤掉交易天数不足的股票。

    参数:
        start_date: 起始日期
        end_date: 结束日期
        min_bars: 最少交易日数量阈值（默认120，约半年交易天数）
                  如果某只股票的交易天数少于这个值，说明其数据不完整，
                  或可能是刚上市/长期停牌的股票，应排除以避免分析偏差

    返回:
        dict: {stock_code: DataFrame}，DataFrame 格式与 load_stock_data 一致

    优化策略：
      使用一条 SQL 查询获取所有股票的数据（而非每只股票查一次），
      在 Python 中通过 groupby 拆分，大幅减少数据库连接次数。
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
    # 批量转换数值列类型
    for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
        all_df[col] = pd.to_numeric(all_df[col], errors='coerce')

    # 按股票代码分组，每组是一整只股票的全部日K线
    result = {}
    for code, grp in all_df.groupby('stock_code'):
        if len(grp) < min_bars:
            # 交易天数不足的股票直接跳过，避免噪声数据影响模型
            continue
        # 选取需要的列，设置日期索引，统一列名
        df = grp.set_index('trade_date')[['open_price', 'high_price', 'low_price',
                                           'close_price', 'volume']].copy()
        df.columns = ['open', 'high', 'low', 'close', 'volume']
        result[code] = df
    return result


def load_financial_data(report_date_min=None):
    """
    加载财务数据（PE、ROE、毛利率、负债率等）。

    财务数据在量化选股中非常重要：
      - PE（市盈率）：衡量估值水平
      - ROE（净资产收益率）：衡量盈利能力，巴菲特最看重的指标之一
      - 毛利率：反映产品竞争力和定价权
      - 负债率：评估财务风险

    参数:
        report_date_min: 最早财报日期，用于过滤过旧的财务数据

    返回:
        DataFrame，列包含 stock_code, report_date, eps, roe, gross_margin, ...
        注意：财务数据是按报告期发布的（季报/半年报/年报），
        而日K线是每天一条，使用时需要用 reindex + ffill 对齐。
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
    # 确保数值列类型正确
    for col in ['eps', 'roe', 'gross_margin', 'debt_ratio', 'net_profit', 'revenue', 'total_assets']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def calc_buy_and_hold(stock_code, start_date, end_date):
    """
    计算区间买入持有收益率。

    这是一个简单的基准（benchmark），用于衡量"什么也不做，买入持有"
    的收益表现。任何主动策略都应该超越这个基准，否则不如被动投资。

    参数:
        stock_code: 股票代码
        start_date: 起始日期
        end_date: 结束日期

    返回:
        float: 区间收益率，如 0.25 表示 25%。数据不足时返回 None
    """
    try:
        df = load_stock_data(stock_code, start_date, end_date)
        if len(df) < 2:
            return None
        # 收益率 = 期末收盘价 / 期初收盘价 - 1
        return float(df['close'].iloc[-1] / df['close'].iloc[0] - 1)
    except Exception:
        return None


# ============================================================
# 策略包装器 - 自动记录交易和净值
# ============================================================

def _wrap_strategy(strategy_class):
    """
    包装策略类，自动记录交易日志和每日净值。

    原理（装饰器模式）：
      不修改原始策略类的代码，而是在运行时动态创建一个子类，
      在 notify_order 和 next 钩子中插入日志记录逻辑。

    为什么需要自动记录？
      1. 避免在每个策略中重复写日志代码
      2. 确保日志格式统一，后续绩效计算和绘图都能正确解析
      3. 减少策略编写者的心智负担

    参数:
        strategy_class: 原始的 Backtrader 策略类

    返回:
        WrappedStrategy: 包装后的子类，具有 _trade_log 和 _nav_log 属性
    """
    class WrappedStrategy(strategy_class):
        def __init__(self):
            super().__init__()
            self._trade_log = []  # 交易记录：每笔成交的时间、方向、价格、数量
            self._nav_log = []    # 净值记录：每天收盘后的账户总资产

        def notify_order(self, order):
            """
            Backtrader 回调函数：订单状态变化时自动调用。

            我们只记录成交的订单（order.Completed），因为撤单和挂单
            不影响实际资产变化。
            """
            if order.status == order.Completed:
                self._trade_log.append({
                    'date': self.data.datetime.date(0),  # 当前 bar 的日期
                    'type': 'BUY' if order.isbuy() else 'SELL',
                    'price': round(order.executed.price, 2),
                    'size': abs(int(order.executed.size)),  # 成交数量（取绝对值）
                })
            # 保留原始策略的 notify_order 行为（如果有的话）
            if hasattr(super(), 'notify_order'):
                super().notify_order(order)

        def next(self):
            """
            Backtrader 回调函数：每个 bar 执行一次。

            在当前 bar 结束时记录净值，然后执行策略逻辑。
            净值 = 现金 + 持仓市值
            """
            self._nav_log.append({
                'date': self.data.datetime.date(0),
                'nav': self.broker.getvalue(),  # 账户总资产
            })
            super().next()

    # 保持类名不变，避免混淆调试信息
    WrappedStrategy.__name__ = strategy_class.__name__
    WrappedStrategy.__qualname__ = strategy_class.__qualname__
    WrappedStrategy.__module__ = strategy_class.__module__
    return WrappedStrategy


# ============================================================
# Cerebro 配置与创建
# ============================================================

def setup_cerebro(strategy_class, stock_code=None, start_date=None, end_date=None,
                  use_sizer=True, df=None, data_class=None, **strategy_kwargs):
    """
    创建并配置好 Backtrader Cerebro 引擎。

    这是整个回测的"一站式配置入口"：
      1. 数据准备：自动加载股票数据或使用传入的 DataFrame
      2. 策略注入：添加策略类和参数
      3. 回测设置：初始资金、手续费、仓位管理
      4. 分析器注册：夏普比、回撤、交易分析

    参数:
        strategy_class: 策略类（未包装，setup_cerebro 内部会包装）
        stock_code: 股票代码（当 df=None 时用于加载数据）
        start_date: 起始日期
        end_date: 结束日期
        use_sizer: 是否使用仓位管理器（True=按比例下单，False=固定数量）
        df: 预加载的 DataFrame（非 None 时跳过数据库加载）
        data_class: 自定义数据源类（如 MLSignalData），None 则用标准 PandasData
        **strategy_kwargs: 传递给策略类的关键字参数

    返回:
        (cerebro, df): Cerebro 引擎实例和对应的 DataFrame
    """
    if df is None:
        df = load_stock_data(stock_code, start_date, end_date)

    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy_class, **strategy_kwargs)

    # 数据源：支持标准数据和 ML 扩展数据
    feed_class = data_class or bt.feeds.PandasData
    cerebro.adddata(feed_class(dataname=df))

    # 资金和费用设置
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)

    # PercentSizer: 每次交易使用账户固定比例的资金
    if use_sizer:
        cerebro.addsizer(bt.sizers.PercentSizer, percents=POSITION_PCT)

    # 注册三个关键分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.02)  # 夏普比（无风险利率2%）
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')                       # 回撤分析
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')                    # 交易统计

    return cerebro, df


# ============================================================
# 绩效指标计算
# ============================================================

def _calc_metrics(cerebro, strat, df):
    """
    从 Backtrader 运行结果中提取完整的绩效指标。

    计算的指标包括：
      - 收益类：总收益、年化收益
      - 风险类：最大回撤（比例和持续天数）
      - 风险调整收益：夏普比、卡玛比
      - 交易统计：交易次数、胜率、盈亏比、利润因子
      - 高级指标：最大连续亏损次数、期望值

    参数:
        cerebro: Cerebro 引擎（用于获取最终资产值）
        strat: 策略实例（运行结果），包含分析器数据
        df: 原始数据（用于计算交易天数）

    返回:
        dict: 包含以上所有指标的字典

    关键概念：
      - 夏普比：衡量每承担一单位风险能获得多少超额收益
      - 卡玛比：年化收益 / 最大回撤，衡量收益与最大回撤的平衡
      - 利润因子：总盈利 / 总亏损，大于 1 表示整体盈利
      - 盈亏比：平均盈利 / 平均亏损
    """
    # ---- 收益指标 ----
    final_value = cerebro.broker.getvalue()
    total_return = (final_value - INITIAL_CASH) / INITIAL_CASH

    # 年化收益：按 252 个交易日/年计算
    trading_days = len(df)
    years = trading_days / 252
    if years > 0 and total_return > -1:
        annual_return = (1 + total_return) ** (1 / years) - 1
    else:
        annual_return = total_return

    # ---- 风险指标 ----
    # 夏普比：来自 Backtrader 的分析器
    sharpe_ratio = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0

    # 最大回撤：从净值日志中逐日计算
    # 为什么要自己算而不是用 Backtrader 的 DrawDown 分析器？
    # 因为自己计算的回撤可以精确到日，且能与净值图匹配
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
        # 如果没有净值日志，回退到 Backtrader 的分析器
        dd = strat.analyzers.drawdown.get_analysis()
        bt_dd = dd.get('max', {}).get('drawdown', 0) / 100
        max_drawdown = min(bt_dd, 1.0)
        max_dd_len = dd.get('max', {}).get('len', 0)

    # 卡玛比：年化收益 / 最大回撤，越高说明收益质量越好
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # ---- 交易统计 ----
    ta = strat.analyzers.trades.get_analysis()
    total_trades = ta.get('total', {}).get('total', 0)
    won_trades = ta.get('won', {}).get('total', 0)
    lost_trades = ta.get('lost', {}).get('total', 0)
    win_rate = won_trades / total_trades if total_trades > 0 else 0

    avg_win = ta.get('won', {}).get('pnl', {}).get('average', 0) or 0
    avg_loss = ta.get('lost', {}).get('pnl', {}).get('average', 0) or 0
    # 盈亏比：平均盈利除以平均亏损的绝对值
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    gross_profit = ta.get('won', {}).get('pnl', {}).get('total', 0) or 0
    gross_loss = ta.get('lost', {}).get('pnl', {}).get('total', 0) or 0
    # 利润因子：总盈利 / 总亏损，> 2 表示策略比较稳健
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

    max_consecutive_losses = _calc_max_consecutive_losses(ta)
    # 期望值：每次交易的平均期望盈亏，> 0 说明策略有正期望
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
    从 TradeAnalyzer 提取最大连续亏损次数。

    为什么这个指标重要？
      连续亏损次数直接影响交易者的心理承受能力。
      如果一个策略经常出现 5 次以上的连续亏损，
      即使总体是盈利的，也很难在实际交易中坚持执行。
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
    一站式回测函数：运行回测并打印绩效报告。

    这是最高层级的回测接口，将以下步骤整合为一个调用：
      1. 包装策略类（自动记录）
      2. 配置 Cerebro 引擎
      3. 运行回测
      4. 计算绩效指标
      5. 打印报告
      6. （可选）绘制图表

    参数:
        strategy_class: 策略类
        stock_code: 股票代码
        start_date: 起始日期
        end_date: 结束日期
        label: 回测标签（用于打印和图表标题）
        plot: 是否绘制图表
        quiet: 是否静默模式（不打印日志）
        use_sizer: 是否使用仓位管理
        df: 预加载的数据
        data_class: 自定义数据源类
        **strategy_kwargs: 策略参数

    返回:
        dict: 包含绩效指标 + df + 交易日志 + 净值日志
    """
    wrapped = _wrap_strategy(strategy_class)
    cerebro, df = setup_cerebro(wrapped, stock_code, start_date, end_date,
                                use_sizer=use_sizer, df=df, data_class=data_class,
                                **strategy_kwargs)

    if not quiet and label:
        print(f"{label} | {stock_code or ''} | {df.index[0].strftime('%Y-%m-%d')} ~ "
              f"{df.index[-1].strftime('%Y-%m-%d')} | {len(df)}个交易日")

    results = cerebro.run()
    strat = results[0]
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
    绘制回测结果三合一图表。

    图表布局（从上到下）：
      上图：K线走势 + 买卖点标记 + 绩效指标面板
      中图：策略净值 vs 买入持有基准
      下图：回撤曲线

    参数:
        result: run_and_report 的返回结果字典
        stock_code: 股票代码（图标题用）
        title: 图表标题

    颜色规范：
      红色（#e74c3c）：买入信号
      绿色（#2ecc71）：卖出信号
      蓝色（#2980b9）：策略净值曲线
      灰色：基准线
    """
    import matplotlib.pyplot as plt
    import matplotlib
    # 中文字体配置，避免 matplotlib 默认不支持中文显示
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    os.makedirs('outputs', exist_ok=True)

    df = result['df']
    trades = result.get('trades', [])
    nav_data = result.get('nav', [])

    if not nav_data:
        print("没有净值数据，跳过绘图")
        return

    # ---- 计算净值曲线和回撤 ----
    nav_df = pd.DataFrame(nav_data)
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df.set_index('date', inplace=True)
    nav_df['nav_pct'] = nav_df['nav'] / INITIAL_CASH       # 净值归一化到1
    nav_df['peak'] = nav_df['nav'].cummax()                  # 累计最高净值
    nav_df['drawdown'] = (nav_df['nav'] - nav_df['peak']) / nav_df['peak'] * 100  # 回撤百分比

    # 买入持有基准：同期股价归一化
    close_start = float(df['close'].iloc[0])
    benchmark = df['close'] / close_start

    # 提取买卖点
    buy_dates = [t['date'] for t in trades if t['type'] == 'BUY']
    buy_prices = [t['price'] for t in trades if t['type'] == 'BUY']
    sell_dates = [t['date'] for t in trades if t['type'] == 'SELL']
    sell_prices = [t['price'] for t in trades if t['type'] == 'SELL']

    m = result
    # 创建三面板图：高度比例 3:2:1
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12),
                                         gridspec_kw={'height_ratios': [3, 2, 1]})

    # ---- 上图：K线 + 买卖点 ----
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

    # 绩效指标面板（右上角信息框）
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

    # ---- 中图：净值对比 ----
    ax2.plot(nav_df.index, nav_df['nav_pct'], '#2980b9', linewidth=1.5, label='策略净值')
    ax2.plot(benchmark.index, benchmark, 'gray', linewidth=1, alpha=0.6, label='买入持有')
    ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
    ax2.set_ylabel('净值 (初始=1.0)')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ---- 下图：回撤 ----
    ax3.fill_between(nav_df.index, nav_df['drawdown'], 0, color='#e74c3c', alpha=0.4)
    ax3.plot(nav_df.index, nav_df['drawdown'], '#c0392b', linewidth=0.8)
    ax3.set_ylabel('回撤(%)')
    ax3.set_xlabel('日期')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    safe_name = title.replace(' ', '_').replace('/', '_')
    plot_file = os.path.join('outputs', f'{safe_name}.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  图表已保存: {plot_file}")
    plt.close()
