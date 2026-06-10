# -*- coding: utf-8 -*-
"""
QuantStats 报告引擎

功能:
  1. nav_to_returns()          - 净值序列转收益率序列
  2. calc_quantstats_metrics() - 计算 QuantStats 全量绩效指标
  3. plot_returns_chart()      - 累计收益 + 回撤 + 月度热力图
  4. plot_strategy_comparison() - 多策略对比图
  5. generate_html_report()    - 生成 QuantStats HTML 报告
  6. generate_ic_report_section() - IC/RankIC 分析 HTML 片段
  7. generate_comprehensive_report() - 综合分析报告(多章节 HTML)
  8. backtrader_nav_to_series() - 从 Backtrader 回测结果提取净值
  9. trade_log_to_dataframe()  - 交易记录转 DataFrame
"""
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import quantstats as qs

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def _load_translation_map(yaml_path=None):
    """
    从 YAML 文件加载翻译映射, 按分类合并为一个有序字典。
    长字符串在前, 短字符串在后, 避免替换冲突。
    如果 YAML 不可用或文件不存在, 返回 None (由调用方使用内置默认值)。
    """
    if yaml_path is None:
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'translations.yaml')
    if not _HAS_YAML or not os.path.exists(yaml_path):
        return None

    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if not data or not isinstance(data, dict):
        return None

    merged = {}
    for section_name, section_dict in data.items():
        if isinstance(section_dict, dict):
            for en, zh in section_dict.items():
                merged[str(en)] = str(zh)

    # 按 key 长度降序排列, 确保长字符串先被替换
    sorted_map = dict(sorted(merged.items(), key=lambda x: len(x[0]), reverse=True))
    return sorted_map


# ============================================================
# 净值 / 收益率转换
# ============================================================

def nav_to_returns(nav_series):
    """
    净值序列 -> 日收益率序列

    参数:
        nav_series: Series, 索引为日期, 值为净值

    返回:
        Series, 日收益率
    """
    if isinstance(nav_series, list):
        nav_series = pd.Series(nav_series)
    returns = nav_series.pct_change().dropna()
    returns.index = pd.to_datetime(returns.index)
    return returns


def backtrader_nav_to_series(nav_log, initial_cash=1000000):
    """
    从 Backtrader 的 _nav_log 提取净值 Series

    参数:
        nav_log: list of dict, 每个 dict 含 'date' 和 'nav'
        initial_cash: 初始资金(用于归一化)

    返回:
        Series, 索引为日期, 值为归一化净值(初始=1.0)
    """
    if not nav_log:
        return pd.Series(dtype=float)

    df = pd.DataFrame(nav_log)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)
    nav_series = df['nav'] / initial_cash
    return nav_series


def trade_log_to_dataframe(trade_log):
    """
    交易记录列表 -> DataFrame

    参数:
        trade_log: list of dict, 含 date/type/price/size

    返回:
        DataFrame
    """
    if not trade_log:
        return pd.DataFrame(columns=['date', 'type', 'price', 'size'])
    df = pd.DataFrame(trade_log)
    df['date'] = pd.to_datetime(df['date'])
    return df


# ============================================================
# QuantStats 指标计算
# ============================================================

def calc_quantstats_metrics(returns, benchmark=None):
    """
    计算 QuantStats 全量绩效指标

    参数:
        returns: Series, 日收益率序列
        benchmark: Series, 基准日收益率(可选)

    返回:
        dict, 所有绩效指标
    """
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)
    # 确保是收益率序列(非累计净值)
    if returns.mean() > 1:
        returns = returns.pct_change().dropna()

    metrics = {}

    # -- 收益指标 --
    metrics['total_return'] = qs.stats.comp(returns)
    metrics['cagr'] = qs.stats.cagr(returns)
    metrics['best_day'] = returns.max()
    metrics['worst_day'] = returns.min()

    # -- 风险指标 --
    metrics['volatility'] = qs.stats.volatility(returns)
    metrics['max_drawdown'] = qs.stats.max_drawdown(returns)
    metrics['var_95'] = qs.stats.value_at_risk(returns)
    metrics['cvar_95'] = qs.stats.cvar(returns)

    # -- 风险调整收益 --
    def _safe(func, *args, default=0.0, **kwargs):
        try:
            val = func(*args, **kwargs)
            if val is None or (isinstance(val, float) and (np.isinf(val) or np.isnan(val))):
                return default
            return val
        except Exception:
            return default

    metrics['sharpe'] = _safe(qs.stats.sharpe, returns)
    metrics['sortino'] = _safe(qs.stats.sortino, returns)
    metrics['calmar'] = _safe(qs.stats.calmar, returns)
    metrics['omega'] = _safe(qs.stats.omega, returns, default=1.0)
    metrics['gain_to_pain'] = _safe(qs.stats.gain_to_pain_ratio, returns)

    # -- 分布特征 --
    metrics['skew'] = _safe(qs.stats.skew, returns)
    metrics['kurtosis'] = _safe(qs.stats.kurtosis, returns)

    # -- 胜率统计 --
    metrics['win_rate'] = _safe(qs.stats.win_rate, returns, default=0.5)
    metrics['avg_win'] = _safe(qs.stats.avg_win, returns)
    metrics['avg_loss'] = _safe(qs.stats.avg_loss, returns)
    metrics['profit_factor'] = _safe(qs.stats.profit_factor, returns, default=1.0)
    metrics['payoff_ratio'] = _safe(qs.stats.payoff_ratio, returns, default=1.0)

    # -- 连续统计 --
    metrics['consecutive_wins'] = _safe(qs.stats.consecutive_wins, returns, default=0)
    metrics['consecutive_losses'] = _safe(qs.stats.consecutive_losses, returns, default=0)

    # -- 基准相关(如果提供) --
    if benchmark is not None:
        benchmark = qs.utils.make_index(benchmark)
        benchmark.index = pd.to_datetime(benchmark.index)
        common_idx = returns.index.intersection(benchmark.index)
        if len(common_idx) > 20:
            r = returns.loc[common_idx]
            b = benchmark.loc[common_idx]
            metrics['information_ratio'] = qs.stats.information_ratio(r, b)
            metrics['alpha'] = r.mean() * 252 - b.mean() * 252
            cov = np.cov(r.values, b.values)
            if cov[1, 1] > 0:
                metrics['beta'] = cov[0, 1] / cov[1, 1]
            else:
                metrics['beta'] = 0.0
            metrics['tracking_error'] = (r - b).std() * np.sqrt(252)

    return metrics


def print_metrics_table(metrics, title='QuantStats 绩效指标'):
    """格式化打印指标表"""
    print(f'\n{"=" * 60}')
    print(f'  {title}')
    print(f'{"=" * 60}')

    label_map = {
        'total_return': ('总收益率', '{:.2%}'),
        'cagr': ('年化收益率(CAGR)', '{:.2%}'),
        'volatility': ('年化波动率', '{:.2%}'),
        'max_drawdown': ('最大回撤', '{:.2%}'),
        'sharpe': ('夏普比率', '{:.4f}'),
        'sortino': ('索提诺比率', '{:.4f}'),
        'calmar': ('卡玛比率', '{:.4f}'),
        'omega': ('Omega比率', '{:.4f}'),
        'gain_to_pain': ('盈亏比(Gain/Pain)', '{:.4f}'),
        'var_95': ('VaR(95%)', '{:.2%}'),
        'cvar_95': ('CVaR(95%)', '{:.2%}'),
        'skew': ('偏度(Skew)', '{:.4f}'),
        'kurtosis': ('峰度(Kurtosis)', '{:.4f}'),
        'win_rate': ('日胜率', '{:.2%}'),
        'avg_win': ('平均盈利日', '{:.4%}'),
        'avg_loss': ('平均亏损日', '{:.4%}'),
        'profit_factor': ('利润因子', '{:.4f}'),
        'payoff_ratio': ('赔付比率', '{:.4f}'),
        'best_day': ('最佳单日', '{:.2%}'),
        'worst_day': ('最差单日', '{:.2%}'),
        'consecutive_wins': ('最大连胜', '{:.0f}'),
        'consecutive_losses': ('最大连亏', '{:.0f}'),
        'alpha': ('Alpha(年化)', '{:.2%}'),
        'beta': ('Beta', '{:.4f}'),
        'information_ratio': ('信息比率', '{:.4f}'),
        'tracking_error': ('跟踪误差', '{:.2%}'),
    }

    for key, (label, fmt) in label_map.items():
        if key in metrics and metrics[key] is not None:
            try:
                val_str = fmt.format(float(metrics[key]))
            except (ValueError, TypeError):
                val_str = str(metrics[key])
            print(f'  {label:<22s}  {val_str:>12s}')

    print(f'{"=" * 60}')


_matplotlib_chinese_configured = False


def configure_matplotlib_chinese():
    """
    注册本机中文字体文件并写入 rcParams，避免仅写字体名时匹配失败（方框/乱码）。
    Windows 下优先 Microsoft YaHei (msyh.ttc)，其次黑体、宋体。
    """
    global _matplotlib_chinese_configured
    if _matplotlib_chinese_configured:
        return

    import platform
    import matplotlib
    from matplotlib import font_manager as fm

    font_paths = []
    if platform.system() == 'Windows':
        font_dir = os.path.join(os.environ.get('WINDIR', r'C:\Windows'), 'Fonts')
        for fname in ('msyh.ttc', 'msyhbd.ttc', 'simhei.ttf', 'simsun.ttc',
                      'STSONG.TTF', 'STHEITI.TTF'):
            fp = os.path.join(font_dir, fname)
            if os.path.isfile(fp):
                font_paths.append(fp)

    registered_name = None
    for fp in font_paths:
        try:
            fm.fontManager.addfont(fp)
            registered_name = fm.FontProperties(fname=fp).get_name()
            break
        except Exception:
            continue

    fallback = ['Microsoft YaHei', 'SimHei', 'SimSun', 'PingFang SC',
                'Arial Unicode MS', 'DejaVu Sans']
    if registered_name:
        matplotlib.rcParams['font.sans-serif'] = [registered_name] + fallback
    else:
        matplotlib.rcParams['font.sans-serif'] = fallback
    matplotlib.rcParams['axes.unicode_minus'] = False
    _matplotlib_chinese_configured = True


# ============================================================
# 可视化
# ============================================================

def plot_returns_chart(returns, benchmark=None, title='策略绩效分析',
                       save_dir='outputs'):
    """
    绘制综合绩效图: 累计收益 + 回撤 + 月度热力图

    参数:
        returns: Series, 日收益率
        benchmark: Series, 基准收益率(可选)
        title: 图表标题
        save_dir: 输出目录
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    configure_matplotlib_chinese()

    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                              gridspec_kw={'height_ratios': [3, 2, 3]})

    # -- 累计收益曲线 --
    cum_returns = (1 + returns).cumprod()
    axes[0].plot(cum_returns.index, cum_returns.values, '#2980b9',
                 linewidth=1.5, label='策略')
    if benchmark is not None:
        cum_bench = (1 + benchmark).cumprod()
        common = cum_returns.index.intersection(cum_bench.index)
        if len(common) > 0:
            axes[0].plot(common, cum_bench.loc[common].values, 'gray',
                         linewidth=1, alpha=0.7, label='基准')
    axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
    axes[0].set_ylabel('累计净值')
    axes[0].set_title(title, fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # -- 回撤水下图 --
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak
    axes[1].fill_between(drawdown.index, drawdown.values, 0,
                          color='#e74c3c', alpha=0.4)
    axes[1].plot(drawdown.index, drawdown.values, '#c0392b', linewidth=0.8)
    axes[1].set_ylabel('回撤')
    axes[1].grid(True, alpha=0.3)

    # -- 月度收益热力图 --
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'return': monthly.values
    })
    if not monthly_df.empty:
        pivot = monthly_df.pivot_table(values='return', index='year',
                                        columns='month', aggfunc='first')
        pivot.columns = [f'{m}月' for m in pivot.columns]
        im = axes[2].imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                            vmin=-0.1, vmax=0.1)
        axes[2].set_xticks(range(len(pivot.columns)))
        axes[2].set_xticklabels(pivot.columns, fontsize=9)
        axes[2].set_yticks(range(len(pivot.index)))
        axes[2].set_yticklabels(pivot.index, fontsize=9)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    axes[2].text(j, i, f'{val:.1%}', ha='center', va='center',
                                fontsize=8, color='black')
        axes[2].set_title('月度收益热力图', fontsize=12)
        plt.colorbar(im, ax=axes[2], shrink=0.6)

    plt.tight_layout()
    safe_title = title.replace(' ', '_').replace('/', '_')
    path = os.path.join(save_dir, f'{safe_title}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  图表已保存: {path}')
    return path


def plot_strategy_comparison(nav_dict, benchmark=None, title='多策略对比',
                              save_dir='outputs'):
    """
    多策略净值曲线对比图

    参数:
        nav_dict: dict {策略名: 净值Series}
        benchmark: Series, 基准净值(可选)
        title: 标题
        save_dir: 输出目录
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    configure_matplotlib_chinese()

    os.makedirs(save_dir, exist_ok=True)

    colors = ['#2980b9', '#e74c3c', '#27ae60', '#f39c12', '#8e44ad', '#1abc9c']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10),
                                     gridspec_kw={'height_ratios': [3, 1]})

    for idx, (name, nav) in enumerate(nav_dict.items()):
        color = colors[idx % len(colors)]
        ax1.plot(nav.index, nav.values, color=color, linewidth=1.5, label=name)

        # 计算回撤
        peak = nav.cummax()
        dd = (nav - peak) / peak
        ax2.plot(dd.index, dd.values, color=color, linewidth=0.8, label=name)

    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark.values, 'gray', linewidth=1,
                 alpha=0.6, linestyle='--', label='基准')

    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
    ax1.set_ylabel('累计净值')
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    ax2.set_ylabel('回撤')
    ax2.set_xlabel('日期')
    ax2.legend(loc='lower left', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_title = title.replace(' ', '_').replace('/', '_')
    path = os.path.join(save_dir, f'{safe_title}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  图表已保存: {path}')
    return path


# ============================================================
# QuantStats 图表中文化补丁
# ============================================================

def _patch_quantstats_chinese():
    """
    Monkey-patch QuantStats 的图表生成函数, 将英文标题替换为中文。
    同时设置 matplotlib 字体支持中文。
    """
    import matplotlib
    import matplotlib.dates as mdates

    configure_matplotlib_chinese()

    # 月份名中文化
    _month_zh = ['1月', '2月', '3月', '4月', '5月', '6月',
                 '7月', '8月', '9月', '10月', '11月', '12月']

    class _ZhMonthFormatter(mdates.DateFormatter):
        """将月份名从英文缩写替换为中文"""
        def __call__(self, x, pos=0):
            result = super().__call__(x, pos)
            month_map = {
                'Jan': '1月', 'Feb': '2月', 'Mar': '3月', 'Apr': '4月',
                'May': '5月', 'Jun': '6月', 'Jul': '7月', 'Aug': '8月',
                'Sep': '9月', 'Oct': '10月', 'Nov': '11月', 'Dec': '12月',
            }
            for en, zh in month_map.items():
                result = result.replace(en, zh)
            return result

    # 补丁 matplotlib 的 AutoDateFormatter, 让日期轴自动使用中文
    _orig_adf_init = mdates.AutoDateFormatter.__init__
    def _zh_adf_init(self, *args, **kwargs):
        _orig_adf_init(self, *args, **kwargs)
        # 覆盖默认的月格式
        if hasattr(self, 'scaled'):
            for key in list(self.scaled.keys()):
                fmt = self.scaled[key]
                if isinstance(fmt, str):
                    fmt = fmt.replace('%b', '%m月').replace('%B', '%m月')
                    self.scaled[key] = fmt
    mdates.AutoDateFormatter.__init__ = _zh_adf_init

    try:
        import quantstats._plotting.wrappers as _qsw
        import quantstats._plotting.core as _qsc
    except ImportError:
        return

    # 保存原始函数
    if hasattr(_qsw, '_patched_zh'):
        return
    _qsw._patched_zh = True

    # --- 补丁 wrappers 中的图表函数 ---

    # 1. snapshot (概览图)
    _orig_snapshot = _qsw.snapshot
    def _zh_snapshot(*args, **kwargs):
        if 'title' not in kwargs or kwargs['title'] == 'Portfolio Summary':
            kwargs['title'] = '策略概览'
        return _orig_snapshot(*args, **kwargs)
    _qsw.snapshot = _zh_snapshot

    # 2. returns (累计收益)
    _orig_returns = _qsw.returns
    def _zh_returns(*args, **kwargs):
        if kwargs.get('ylabel') == 'Cumulative Returns':
            kwargs['ylabel'] = '累计收益'
        return _orig_returns(*args, **kwargs)
    _qsw.returns = _zh_returns

    # 3. log_returns
    _orig_log_returns = _qsw.log_returns
    def _zh_log_returns(*args, **kwargs):
        if kwargs.get('ylabel') == 'Cumulative Returns':
            kwargs['ylabel'] = '累计收益'
        return _orig_log_returns(*args, **kwargs)
    _qsw.log_returns = _zh_log_returns

    # 4. daily_returns
    _orig_daily = _qsw.daily_returns
    def _zh_daily(*args, **kwargs):
        return _orig_daily(*args, **kwargs)
    _qsw.daily_returns = _zh_daily

    # 5. drawdown
    _orig_drawdown = _qsw.drawdown
    def _zh_drawdown(*args, **kwargs):
        if kwargs.get('ylabel') == 'Drawdown':
            kwargs['ylabel'] = '回撤'
        return _orig_drawdown(*args, **kwargs)
    _qsw.drawdown = _zh_drawdown

    # 6. rolling_volatility
    _orig_rv = _qsw.rolling_volatility
    def _zh_rv(*args, **kwargs):
        return _orig_rv(*args, **kwargs)
    _qsw.rolling_volatility = _zh_rv

    # 7. rolling_sharpe
    _orig_rs = _qsw.rolling_sharpe
    def _zh_rs(*args, **kwargs):
        return _orig_rs(*args, **kwargs)
    _qsw.rolling_sharpe = _zh_rs

    # 8. rolling_sortino
    _orig_rso = _qsw.rolling_sortino
    def _zh_rso(*args, **kwargs):
        return _orig_rso(*args, **kwargs)
    _qsw.rolling_sortino = _zh_rso

    # 9. rolling_beta
    _orig_rb = _qsw.rolling_beta
    def _zh_rb(*args, **kwargs):
        return _orig_rb(*args, **kwargs)
    _qsw.rolling_beta = _zh_rb

    # --- 补丁 core 中的 plot_timeseries 图表标题 ---
    _orig_plot_ts = _qsc.plot_timeseries
    _title_map = {
        'Cumulative Returns': '累计收益',
        'Returns': '收益',
        'Cumulative Returns vs Benchmark': '累计收益 vs 基准',
        'Returns vs Benchmark': '收益 vs 基准',
        'Cumulative Returns (Volatility Matched)': '累计收益(波动率匹配)',
        'Daily Active Returns': '每日主动收益',
        'Daily Returns': '每日收益',
        'EOY Returns': '年度收益',
        'Underwater Plot': '水下图(回撤)',
        'Drawdown': '回撤',
        'Rolling Beta to Benchmark': '滚动贝塔',
    }
    _ylabel_map = {
        'Returns': '收益',
        'Cumulative Returns': '累计收益',
        'Drawdown': '回撤',
        'Cumulative Return': '累计收益',
        'Daily Return': '每日收益',
    }

    def _zh_plot_ts(*args, **kwargs):
        if 'title' in kwargs:
            t = kwargs['title']
            for en, zh in _title_map.items():
                if en in t:
                    kwargs['title'] = t.replace(en, zh)
                    break
        if 'returns_label' in kwargs and kwargs['returns_label'] in _ylabel_map:
            kwargs['returns_label'] = _ylabel_map[kwargs['returns_label']]
        return _orig_plot_ts(*args, **kwargs)
    _qsc.plot_timeseries = _zh_plot_ts

    # 补丁 core.plot_returns_bars (年度柱状图)
    if hasattr(_qsc, 'plot_returns_bars'):
        _orig_bars = _qsc.plot_returns_bars
        def _zh_bars(*args, **kwargs):
            if 'title' in kwargs:
                t = kwargs['title']
                for en, zh in _title_map.items():
                    if en in t:
                        kwargs['title'] = t.replace(en, zh)
                        break
            return _orig_bars(*args, **kwargs)
        _qsc.plot_returns_bars = _zh_bars

    # --- 补丁 stats.monthly_returns 将列名改为中文 ---
    try:
        import quantstats.stats as _qss
        if not hasattr(_qss, '_patched_monthly'):
            _qss._patched_monthly = True
            _orig_monthly_returns = _qss.monthly_returns
            _month_col_map = {
                'JAN': '1月', 'FEB': '2月', 'MAR': '3月', 'APR': '4月',
                'MAY': '5月', 'JUN': '6月', 'JUL': '7月', 'AUG': '8月',
                'SEP': '9月', 'OCT': '10月', 'NOV': '11月', 'DEC': '12月',
                'EOY': '全年',
            }
            def _zh_monthly_returns(*args, **kwargs):
                result = _orig_monthly_returns(*args, **kwargs)
                result.columns = [_month_col_map.get(str(c), c) for c in result.columns]
                return result
            _qss.monthly_returns = _zh_monthly_returns
    except Exception:
        pass

    # --- 补丁 wrappers.monthly_heatmap 翻译标题 ---
    _orig_heatmap = _qsw.monthly_heatmap
    def _zh_heatmap(*args, **kwargs):
        if kwargs.get('returns_label') == 'Strategy':
            kwargs['returns_label'] = '策略'
        return _orig_heatmap(*args, **kwargs)
    _qsw.monthly_heatmap = _zh_heatmap


def _post_translate_chart_text(html_path):
    """
    后处理: 替换图表图片中无法通过 monkey-patch 修改的文本。
    主要处理 QuantStats HTML 中嵌入的 SVG 图表和剩余英文。
    """
    if not os.path.exists(html_path):
        return

    with open(html_path, 'r', encoding='utf-8-sig') as f:
        html = f.read()

    # 替换图表标题 (在 HTML 模板的标题区域)
    chart_title_map = {
        'Cumulative Returns vs Benchmark': '累计收益 vs 基准',
        'Cumulative Returns': '累计收益',
        'EOY Returns vs Benchmark': '年度收益 vs 基准',
        'EOY Returns': '年度收益',
        'Returns Distribution': '收益分布',
        'Monthly Returns': '月度收益',
        'Monthly Active Returns': '月度主动收益',
        'Daily Returns': '每日收益',
        'Daily Active Returns': '每日主动收益',
        'Rolling Volatility': '滚动波动率',
        'Rolling Sharpe': '滚动夏普比率',
        'Rolling Sortino': '滚动索提诺比率',
        'Rolling Beta': '滚动贝塔',
        'Underwater Plot': '水下图(回撤)',
        'Drawdowns Periods': '回撤区间',
        'Drawdown Periods': '回撤区间',
        'Worst 10 Drawdowns': '最差10次回撤',
        'Worst 5 Drawdowns': '最差5次回撤',
        'Portfolio Summary': '策略概览',
        'Log Returns': '对数收益',
        'Volatility Matched': '波动率匹配',
        'Active Returns': '主动收益',
        'Strategy': '策略',
    }

    for en, zh in chart_title_map.items():
        html = html.replace(en, zh)

    with open(html_path, 'w', encoding='utf-8-sig') as f:
        f.write(html)


# ============================================================
# QuantStats HTML 报告
# ============================================================

def generate_html_report(returns, benchmark=None, title='Strategy Performance Report',
                          output_path='outputs/quantstats_report.html'):
    """
    生成 QuantStats 原版英文 HTML 报告 (一键生成, 无中文补丁)

    QuantStats 内部渲染图表时不支持中文字体, 强行中文化会导致乱码。
    因此本函数生成纯英文报告, 中文版请使用 generate_chinese_report()。

    参数:
        returns: Series, 日收益率
        benchmark: Series 或 str, 基准(可选)
        title: 报告标题
        output_path: 输出路径
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)

    bench = benchmark
    if bench is not None and isinstance(bench, pd.Series):
        bench = bench.copy()
        bench.index = pd.to_datetime(bench.index)
        common = returns.index.intersection(bench.index)
        if len(common) > 0:
            returns = returns.loc[common]
            bench = bench.loc[common]
        else:
            bench = None

    try:
        qs.reports.html(returns, benchmark=bench, title=title,
                        output=output_path)
        print(f'  英文报告已生成: {output_path}')
    except Exception as e:
        print(f'  英文报告生成失败: {e}, 尝试无基准模式...')
        qs.reports.html(returns, title=title, output=output_path)
        print(f'  英文报告已生成(无基准): {output_path}')

    return output_path


def generate_chinese_report(returns, benchmark=None, title='策略绩效报告',
                            output_path='outputs/chinese_report.html'):
    """
    生成完全中文化的策略绩效分析报告 (自主渲染, 全中文)

    使用 QuantStats 函数计算指标数据,
    所有图表使用 matplotlib 自主渲染(中文字体),
    所有表格和标签均为中文, 不依赖 QuantStats HTML 模板。

    参数:
        returns: Series, 日收益率序列
        benchmark: Series, 基准日收益率(可选)
        title: str, 报告标题
        output_path: str, 输出文件路径

    返回:
        str, 输出文件路径
    """
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    configure_matplotlib_chinese()

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    returns = returns.copy()
    returns.index = pd.to_datetime(returns.index)

    bench = None
    if benchmark is not None and isinstance(benchmark, pd.Series):
        bench = benchmark.copy()
        bench.index = pd.to_datetime(bench.index)
        common = returns.index.intersection(bench.index)
        if len(common) > 0:
            bench = bench.loc[common]
        else:
            bench = None

    metrics = calc_quantstats_metrics(returns, benchmark=bench)

    def _fig_to_b64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('ascii')

    charts = {}

    # ---- 图1: 累计收益曲线 ----
    fig, ax = plt.subplots(figsize=(14, 5))
    cum = (1 + returns).cumprod()
    ax.plot(cum.index, cum.values, '#2980b9', linewidth=1.5, label='策略')
    if bench is not None:
        cum_b = (1 + bench).cumprod()
        ci = cum.index.intersection(cum_b.index)
        if len(ci) > 0:
            ax.plot(ci, cum_b.loc[ci].values, '#95a5a6', linewidth=1,
                    label='基准', alpha=0.7)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
    ax.set_title('累计收益曲线', fontsize=14, fontweight='bold')
    ax.set_ylabel('累计净值')
    ax.set_xlabel('日期')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    charts['cumulative'] = _fig_to_b64(fig)

    # ---- 图2: 回撤水下图 ----
    fig, ax = plt.subplots(figsize=(14, 4))
    peak = cum.cummax()
    dd = (cum - peak) / peak
    ax.fill_between(dd.index, dd.values, 0, color='#e74c3c', alpha=0.4)
    ax.plot(dd.index, dd.values, '#c0392b', linewidth=0.8)
    ax.set_title('水下图(回撤)', fontsize=14, fontweight='bold')
    ax.set_ylabel('回撤幅度')
    ax.set_xlabel('日期')
    ax.grid(True, alpha=0.3)
    charts['drawdown'] = _fig_to_b64(fig)

    # ---- 图3: 月度收益热力图 ----
    monthly = returns.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    m_df = pd.DataFrame({
        'year': monthly.index.year,
        'month': monthly.index.month,
        'ret': monthly.values,
    })
    if not m_df.empty and len(m_df) > 1:
        pivot = m_df.pivot_table(values='ret', index='year',
                                  columns='month', aggfunc='first')
        mn = {1: '1月', 2: '2月', 3: '3月', 4: '4月', 5: '5月', 6: '6月',
              7: '7月', 8: '8月', 9: '9月', 10: '10月', 11: '11月', 12: '12月'}
        pivot.columns = [mn.get(c, c) for c in pivot.columns]
        fig, ax = plt.subplots(figsize=(14, max(3, len(pivot) * 0.8 + 1)))
        im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto',
                       vmin=-0.1, vmax=0.1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=10)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.1%}', ha='center', va='center',
                            fontsize=9, color='black')
        ax.set_title('月度收益热力图', fontsize=14, fontweight='bold')
        fig.colorbar(im, ax=ax, shrink=0.6)
        charts['monthly_heatmap'] = _fig_to_b64(fig)

    # ---- 图4: 日收益分布 ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(returns.dropna().values, bins=50, color='#3498db', alpha=0.7,
            edgecolor='white')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=returns.mean(), color='green', linestyle='--', alpha=0.7,
               label=f'均值: {returns.mean():.4%}')
    ax.set_title('日收益分布', fontsize=14, fontweight='bold')
    ax.set_xlabel('日收益率')
    ax.set_ylabel('频次')
    ax.legend()
    ax.grid(True, alpha=0.3)
    charts['distribution'] = _fig_to_b64(fig)

    # ---- 图5: 滚动夏普 + 滚动波动率 ----
    if len(returns) > 60:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
        rs = returns.rolling(60).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )
        ax1.plot(rs.index, rs.values, '#2980b9', linewidth=1)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax1.set_title('滚动夏普比率(60日)', fontsize=13, fontweight='bold')
        ax1.set_ylabel('夏普比率')
        ax1.grid(True, alpha=0.3)
        rv = returns.rolling(20).std() * np.sqrt(252)
        ax2.plot(rv.index, rv.values, '#e74c3c', linewidth=1)
        ax2.set_title('滚动波动率(20日)', fontsize=13, fontweight='bold')
        ax2.set_ylabel('年化波动率')
        ax2.set_xlabel('日期')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        charts['rolling'] = _fig_to_b64(fig)

    # ---- 图6: 年度收益柱状图 ----
    yearly = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    if len(yearly) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        colors_bar = ['#27ae60' if v >= 0 else '#e74c3c' for v in yearly.values]
        bars = ax.bar(yearly.index.year, yearly.values * 100,
                      color=colors_bar, width=0.6)
        for bar, val in zip(bars, yearly.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.1%}', ha='center',
                    va='bottom' if val >= 0 else 'top', fontsize=10)
        ax.set_title('年度收益', fontsize=14, fontweight='bold')
        ax.set_ylabel('收益率 (%)')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3, axis='y')
        charts['yearly'] = _fig_to_b64(fig)

    # ---- 计算 Top 5 回撤区间 ----
    dd_periods = []
    in_dd = False
    start_d = None
    for date, val in dd.items():
        if val < 0 and not in_dd:
            start_d = date
            in_dd = True
        elif val >= 0 and in_dd:
            valley_idx = dd[start_d:date].idxmin()
            valley_val = dd[start_d:date].min()
            dd_periods.append({
                'start': start_d, 'valley': valley_idx, 'end': date,
                'drawdown': valley_val, 'days': (date - start_d).days,
            })
            in_dd = False
    if in_dd and start_d is not None:
        valley_idx = dd[start_d:].idxmin()
        valley_val = dd[start_d:].min()
        dd_periods.append({
            'start': start_d, 'valley': valley_idx, 'end': dd.index[-1],
            'drawdown': valley_val, 'days': (dd.index[-1] - start_d).days,
            'ongoing': True,
        })
    dd_periods.sort(key=lambda x: x['drawdown'])
    top_dd = dd_periods[:5]

    # ---- 月度收益表格 ----
    monthly_table_html = ''
    if not m_df.empty:
        pivot_raw = m_df.pivot_table(values='ret', index='year',
                                      columns='month', aggfunc='first')
        eoy = returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
        eoy_dict = {y.year: v for y, v in eoy.items()}
        monthly_table_html = '<table class="data-table"><tr><th>年份</th>'
        for m in range(1, 13):
            monthly_table_html += f'<th>{m}月</th>'
        monthly_table_html += '<th>全年</th></tr>'
        for year in sorted(pivot_raw.index):
            monthly_table_html += f'<tr><td style="font-weight:bold">{year}</td>'
            for m in range(1, 13):
                cell_val = None
                if m in pivot_raw.columns:
                    raw_val = pivot_raw.loc[year, m]
                    if not pd.isna(raw_val):
                        cell_val = raw_val
                if cell_val is not None:
                    c = '#27ae60' if cell_val >= 0 else '#e74c3c'
                    monthly_table_html += f'<td style="color:{c}">{cell_val:.1%}</td>'
                else:
                    monthly_table_html += '<td>-</td>'
            eoy_val = eoy_dict.get(year, None)
            if eoy_val is not None:
                ec = '#27ae60' if eoy_val >= 0 else '#e74c3c'
                monthly_table_html += (
                    f'<td style="font-weight:bold;color:{ec}">{eoy_val:.1%}</td>')
            else:
                monthly_table_html += '<td>-</td>'
            monthly_table_html += '</tr>'
        monthly_table_html += '</table>'

    # ---- 绩效指标表格 ----
    start_date = returns.index[0].strftime('%Y-%m-%d')
    end_date = returns.index[-1].strftime('%Y-%m-%d')
    trading_days = len(returns)

    metric_rows = [
        ('开始日期', start_date),
        ('结束日期', end_date),
        ('---', ''),
        ('累计收益', f'{metrics.get("total_return", 0):.2%}'),
        ('年化收益(CAGR)', f'{metrics.get("cagr", 0):.2%}'),
        ('年化波动率', f'{metrics.get("volatility", 0):.2%}'),
        ('---', ''),
        ('夏普比率', f'{metrics.get("sharpe", 0):.4f}'),
        ('索提诺比率', f'{metrics.get("sortino", 0):.4f}'),
        ('卡玛比率', f'{metrics.get("calmar", 0):.4f}'),
        ('欧米伽比率', f'{metrics.get("omega", 1):.4f}'),
        ('盈亏比(Gain/Pain)', f'{metrics.get("gain_to_pain", 0):.4f}'),
        ('---', ''),
        ('最大回撤', f'{metrics.get("max_drawdown", 0):.2%}'),
        ('VaR(95%)', f'{metrics.get("var_95", 0):.2%}'),
        ('CVaR(95%)', f'{metrics.get("cvar_95", 0):.2%}'),
        ('---', ''),
        ('最佳单日', f'{metrics.get("best_day", 0):.2%}'),
        ('最差单日', f'{metrics.get("worst_day", 0):.2%}'),
        ('日胜率', f'{metrics.get("win_rate", 0):.2%}'),
        ('平均盈利', f'{metrics.get("avg_win", 0):.4%}'),
        ('平均亏损', f'{metrics.get("avg_loss", 0):.4%}'),
        ('利润因子', f'{metrics.get("profit_factor", 0):.4f}'),
        ('赔付比率', f'{metrics.get("payoff_ratio", 0):.4f}'),
        ('最大连胜', f'{metrics.get("consecutive_wins", 0):.0f}'),
        ('最大连亏', f'{metrics.get("consecutive_losses", 0):.0f}'),
        ('---', ''),
        ('偏度(Skew)', f'{metrics.get("skew", 0):.4f}'),
        ('峰度(Kurtosis)', f'{metrics.get("kurtosis", 0):.4f}'),
    ]
    if bench is not None:
        metric_rows.append(('---', ''))
        if 'alpha' in metrics:
            metric_rows.append(('Alpha(年化超额)', f'{metrics["alpha"]:.2%}'))
        if 'beta' in metrics:
            metric_rows.append(('Beta(贝塔)', f'{metrics["beta"]:.4f}'))
        if 'information_ratio' in metrics:
            metric_rows.append(('信息比率', f'{metrics["information_ratio"]:.4f}'))
        if 'tracking_error' in metrics:
            metric_rows.append(('跟踪误差', f'{metrics["tracking_error"]:.2%}'))

    metrics_html = '<table class="metrics-table">'
    metrics_html += '<tr><th style="width:45%">指标</th><th style="width:55%">数值</th></tr>'
    for label, value in metric_rows:
        if label == '---':
            metrics_html += ('<tr><td colspan="2" style="border-bottom:2px solid '
                             '#eee;padding:2px;"></td></tr>')
        else:
            metrics_html += f'<tr><td>{label}</td><td>{value}</td></tr>'
    metrics_html += '</table>'

    # ---- 回撤区间表格 ----
    dd_table_html = ''
    if top_dd:
        dd_table_html = ('<table class="data-table"><tr><th>序号</th>'
                         '<th>开始日期</th><th>谷底日期</th><th>恢复日期</th>'
                         '<th>回撤幅度</th><th>持续天数</th></tr>')
        for i, d in enumerate(top_dd, 1):
            ongoing = d.get('ongoing', False)
            end_str = '未恢复' if ongoing else d['end'].strftime('%Y-%m-%d')
            dd_table_html += (
                f'<tr><td>{i}</td>'
                f'<td>{d["start"].strftime("%Y-%m-%d")}</td>'
                f'<td>{d["valley"].strftime("%Y-%m-%d")}</td>'
                f'<td>{end_str}</td>'
                f'<td style="color:#e74c3c;font-weight:bold">{d["drawdown"]:.2%}</td>'
                f'<td>{d["days"]}</td></tr>')
        dd_table_html += '</table>'

    # ---- 组装 HTML ----
    total_return = metrics.get('total_return', 0)
    cagr = metrics.get('cagr', 0)
    max_dd = metrics.get('max_drawdown', 0)
    sharpe = metrics.get('sharpe', 0)
    sortino = metrics.get('sortino', 0)
    calmar = metrics.get('calmar', 0)
    win_rate = metrics.get('win_rate', 0)
    volatility = metrics.get('volatility', 0)

    def _vc(v, reverse=False):
        if reverse:
            return '#27ae60' if v <= 0 else '#e74c3c'
        return '#27ae60' if v >= 0 else '#e74c3c'

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: "Microsoft YaHei","SimHei","PingFang SC",sans-serif;
         background: #f0f2f5; color: #2c3e50; }}
  .container {{ max-width: 1100px; margin: 0 auto; background: white;
               box-shadow: 0 4px 20px rgba(0,0,0,0.08); }}
  .header {{ background: linear-gradient(135deg, #1a2a3a 0%, #2980b9 100%);
            color: white; padding: 40px 50px; }}
  .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
  .header .sub {{ font-size: 14px; color: rgba(255,255,255,0.7); }}
  .metric-cards {{ display: flex; flex-wrap: wrap; gap: 12px;
                   padding: 24px 40px; background: #fafbfc;
                   border-bottom: 1px solid #eee; }}
  .card {{ flex: 1; min-width: 120px; max-width: 200px; background: white;
          border-radius: 8px; padding: 16px; text-align: center;
          box-shadow: 0 1px 4px rgba(0,0,0,0.06);
          border-top: 3px solid #3498db; }}
  .card .val {{ font-size: 22px; font-weight: bold; margin: 4px 0; }}
  .card .lbl {{ font-size: 11px; color: #7f8c8d; }}
  .section {{ padding: 30px 40px; border-bottom: 1px solid #f0f0f0; }}
  .section h2 {{ font-size: 18px; color: #2c3e50; margin-bottom: 16px;
                padding-bottom: 8px; border-bottom: 2px solid #3498db; }}
  .chart-box {{ text-align: center; margin: 16px 0; }}
  .chart-box img {{ max-width: 100%; border-radius: 6px;
                    border: 1px solid #e8e8e8; }}
  .metrics-table {{ width: 100%; border-collapse: collapse; }}
  .metrics-table th {{ background: #34495e; color: white; padding: 10px 16px;
                       text-align: left; font-size: 13px; }}
  .metrics-table td {{ padding: 8px 16px; border-bottom: 1px solid #f0f0f0;
                       font-size: 13px; }}
  .metrics-table tr:hover {{ background: #f8f9fa; }}
  .data-table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
  .data-table th {{ background: #34495e; color: white; padding: 10px 12px;
                    text-align: center; font-size: 12px; }}
  .data-table td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0;
                    text-align: center; font-size: 12px; }}
  .data-table tr:hover {{ background: #f8f9fa; }}
  .footer {{ text-align: center; padding: 20px; color: #95a5a6;
            font-size: 11px; background: #fafbfc; }}
  @media (max-width: 768px) {{
    .metric-cards {{ flex-direction: column; }}
    .section {{ padding: 20px; }}
    .header {{ padding: 24px; }}
  }}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>{title}</h1>
  <div class="sub">{start_date} ~ {end_date} | 共 {trading_days} 个交易日 | 报告生成: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}</div>
</div>

<div class="metric-cards">
  <div class="card">
    <div class="lbl">累计收益</div>
    <div class="val" style="color:{_vc(total_return)}">{total_return:.2%}</div>
  </div>
  <div class="card">
    <div class="lbl">年化收益(CAGR)</div>
    <div class="val" style="color:{_vc(cagr)}">{cagr:.2%}</div>
  </div>
  <div class="card">
    <div class="lbl">最大回撤</div>
    <div class="val" style="color:#e74c3c">{max_dd:.2%}</div>
  </div>
  <div class="card">
    <div class="lbl">夏普比率</div>
    <div class="val" style="color:{_vc(sharpe)}">{sharpe:.2f}</div>
  </div>
  <div class="card">
    <div class="lbl">索提诺比率</div>
    <div class="val" style="color:{_vc(sortino)}">{sortino:.2f}</div>
  </div>
  <div class="card">
    <div class="lbl">年化波动率</div>
    <div class="val">{volatility:.2%}</div>
  </div>
  <div class="card">
    <div class="lbl">日胜率</div>
    <div class="val">{win_rate:.1%}</div>
  </div>
  <div class="card">
    <div class="lbl">卡玛比率</div>
    <div class="val" style="color:{_vc(calmar)}">{calmar:.2f}</div>
  </div>
</div>

<div class="section">
  <h2>累计收益曲线</h2>
  <div class="chart-box">
    <img src="data:image/png;base64,{charts['cumulative']}" alt="累计收益">
  </div>
</div>

<div class="section">
  <h2>水下图(回撤)</h2>
  <div class="chart-box">
    <img src="data:image/png;base64,{charts['drawdown']}" alt="回撤">
  </div>
</div>
'''

    if 'yearly' in charts:
        html += f'''
<div class="section">
  <h2>年度收益</h2>
  <div class="chart-box">
    <img src="data:image/png;base64,{charts['yearly']}" alt="年度收益">
  </div>
</div>
'''

    if 'monthly_heatmap' in charts:
        html += f'''
<div class="section">
  <h2>月度收益热力图</h2>
  <div class="chart-box">
    <img src="data:image/png;base64,{charts['monthly_heatmap']}" alt="月度热力图">
  </div>
</div>
'''

    html += f'''
<div class="section">
  <h2>日收益分布</h2>
  <div class="chart-box">
    <img src="data:image/png;base64,{charts['distribution']}" alt="日收益分布">
  </div>
</div>
'''

    if 'rolling' in charts:
        html += f'''
<div class="section">
  <h2>滚动风险指标</h2>
  <div class="chart-box">
    <img src="data:image/png;base64,{charts['rolling']}" alt="滚动指标">
  </div>
</div>
'''

    html += f'''
<div class="section">
  <h2>绩效指标明细</h2>
  {metrics_html}
</div>
'''

    if monthly_table_html:
        html += f'''
<div class="section">
  <h2>月度收益明细</h2>
  {monthly_table_html}
</div>
'''

    if dd_table_html:
        html += f'''
<div class="section">
  <h2>Top {len(top_dd)} 回撤区间</h2>
  {dd_table_html}
</div>
'''

    html += '''
<div class="footer">
  <p>本报告基于 QuantStats 计算引擎自主渲染生成 | 仅供参考, 不构成投资建议</p>
</div>

</div>
</body>
</html>'''

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(html)

    print(f'  中文报告已生成: {output_path}')
    return output_path


def translate_quantstats_html(html_path):
    """
    将 QuantStats 生成的 HTML 报告中的英文标签翻译为中文。
    优先从 translations.yaml 加载翻译, 找不到时使用内置默认值。

    参数:
        html_path: str, HTML 文件路径
    """
    if not os.path.exists(html_path):
        return

    with open(html_path, 'r', encoding='utf-8-sig') as f:
        html = f.read()

    # 优先尝试从 YAML 加载
    translation_map = _load_translation_map()

    if translation_map is None:
        # YAML 不可用时使用内置默认值 (按长度降序排列)
        _builtin = {
            'Cumulative Returns vs Benchmark': '累计收益 vs 基准',
            'Expected Shortfall (cVaR)': '预期亏损(CVaR)',
            'EOY Returns vs Benchmark': '年度收益 vs 基准',
            'Top 5 Drawdown Periods': 'Top 5 回撤区间',
            'Strategy Visualization': '策略可视化分析',
            'Ulcer Performance Index': '溃疡绩效指数',
            'Max Consecutive Losses': '最大连亏',
            'Risk-Adjusted Return': '风险调整收益',
            'Daily Value-at-Risk': '日VaR风险价值',
            'Max Consecutive Wins': '最大连胜',
            'Outlier Loss Ratio': '异常亏损比率',
            'Outlier Win Ratio': '异常盈利比率',
            'Performance Metrics': '绩效指标',
            'Risk-Return Ratio': '风险收益比',
            'Common Sense Ratio': '常识比率',
            'Max DD Period Start': '最大回撤起始',
            'Cumulative Returns': '累计收益曲线',
            'Returns Distribution': '收益分布',
            'Worst 10 Drawdowns': '最差10次回撤',
            'Max DD Period End': '最大回撤结束',
            'Drawdowns Periods': '回撤区间',
            'Drawdown Periods': '回撤区间',
            'Rolling Volatility': '滚动波动率',
            'Worst 5 Drawdowns': '最差5次回撤',
            'Cumulative Return': '累计收益',
            'Monthly Returns': '月度收益热力图',
            'Longest DD Days': '最长回撤天数',
            'Information Ratio': '信息比率',
            'Recovery Factor': '恢复因子',
            'Rolling Sortino': '滚动索提诺比率',
            'Gain/Pain Ratio': '盈亏比率',
            'Gain/Pain (1M)': '盈亏比(近1月)',
            'Time in Market': '持仓时间占比',
            'Avg. Down Month': '平均下跌月份',
            'Serenity Index': '宁静指数',
            'Risk-Free Rate': '无风险利率',
            'Kelly Criterion': '凯利公式仓位',
            'Treynor Ratio': '特雷诺比率',
            'Volatility (ann.)': '年化波动率',
            'Expected Daily %%': '预期日收益%%',
            'Expected Daily %': '预期日收益%',
            'Expected Monthly %%': '预期月收益%%',
            'Expected Monthly %': '预期月收益%',
            'Expected Yearly %%': '预期年收益%%',
            'Expected Yearly %': '预期年收益%',
            'Rolling Sharpe': '滚动夏普比率',
            'Avg. Up Month': '平均上涨月份',
            'Rolling Beta': '滚动贝塔',
            'Underwater Plot': '水下图(回撤)',
            'Smart Sortino': '智能索提诺',
            'Profit Factor': '利润因子',
            'Daily Returns': '每日收益',
            'Start Period': '开始日期',
            'Max Drawdown': '最大回撤',
            'Sharpe Ratio': '夏普比率',
            'Sortino Ratio': '索提诺比率',
            'Calmar Ratio': '卡玛比率',
            'Omega Ratio': '欧米伽比率',
            'Payoff Ratio': '赔付比率',
            'Profit Ratio': '利润比率',
            'Win/Loss Ratio': '盈亏比',
            'EOY Returns': '年度收益',
            'End Period': '结束日期',
            'Smart Sharpe': '智能夏普',
            'Risk of Ruin': '破产风险',
            'Win Days %%': '日胜率%%',
            'Win Days %': '日胜率%',
            'Win Month %%': '月胜率%%',
            'Win Month %': '月胜率%',
            'Win Quarter %%': '季胜率%%',
            'Win Quarter %': '季胜率%',
            'Win Year %%': '年胜率%%',
            'Win Year %': '年胜率%',
            'Expected Daily': '预期日收益',
            'Expected Monthly': '预期月收益',
            'Expected Yearly': '预期年收益',
            'Max DD Date': '最大回撤日',
            'Ulcer Index': '溃疡指数',
            'Avg. Return': '平均收益',
            'Win Quarter': '季胜率',
            'Best Month': '最佳月份',
            'Worst Month': '最差月份',
            'Win Month': '月胜率',
            'Tail Ratio': '尾部比率',
            'Volatility': '波动率',
            'Best Year': '最佳年度',
            'Worst Year': '最差年度',
            'Best Day': '最佳单日',
            'Worst Day': '最差单日',
            'Avg. Win': '平均盈利',
            'Avg. Loss': '平均亏损',
            'Win Year': '年胜率',
            'CPC Index': 'CPC指数',
            'CAGR%%': '年化收益(CAGR)%%',
            'CAGR%': '年化收益(CAGR)%',
            '3Y (ann.)': '近3年(年化)',
            '5Y (ann.)': '近5年(年化)',
            '10Y (ann.)': '近10年(年化)',
            'Strategy': '策略',
            'Benchmark': '基准',
            'Kurtosis': '峰度',
            'Drawdown': '回撤',
            'Alpha': 'Alpha(超额)',
            'Beta': 'Beta(贝塔)',
            'R^2': 'R平方',
            'Started': '开始',
            'Recovered': '恢复',
            'Valley': '谷底',
            'Metric': '指标',
            'Sharpe': '夏普',
            'Sortino': '索提诺',
            'Calmar': '卡玛',
            'Omega': '欧米伽',
            'Skew': '偏度',
            'Days': '天数',
            'MTD': '本月至今',
            'QTD': '本季至今',
            'YTD': '本年至今',
            '3M': '近3月',
            '6M': '近6月',
            '1Y': '近1年',
            'All': '全部',
            'Cumulative': '累计',
            'Year': '年度',
            'Return': '收益',
        }
        translation_map = _builtin

    for en, zh in translation_map.items():
        html = html.replace(en, zh)

    with open(html_path, 'w', encoding='utf-8-sig') as f:
        f.write(html)

    print(f'  HTML报告已中文化: {html_path}')


# ============================================================
# IC/RankIC 分析报告片段
# ============================================================

def calc_ic_analysis(factor_values, return_values, dates, horizons=None):
    """
    计算 IC/RankIC 全套分析指标

    参数:
        factor_values: Series 或 array, 因子值
        return_values: Series 或 array, 实际收益率
        dates: 日期序列
        horizons: list of int, IC衰减分析的时间窗口

    返回:
        dict, 包含 ic/rank_ic/icir/monthly_ic 等
    """
    from scipy import stats as sp_stats

    df = pd.DataFrame({
        'date': pd.to_datetime(dates),
        'factor': factor_values,
        'return': return_values,
    }).dropna()

    if len(df) < 20:
        return {'ic': 0, 'rank_ic': 0, 'icir': 0}

    # Pearson IC
    ic = df['factor'].corr(df['return'], method='pearson')

    # Spearman RankIC
    rank_ic = df['factor'].corr(df['return'], method='spearman')

    # 月度 IC 序列
    df['month'] = df['date'].dt.to_period('M')
    monthly_ic = df.groupby('month').apply(
        lambda g: g['factor'].corr(g['return'], method='spearman')
        if len(g) > 5 else np.nan
    ).dropna()

    # ICIR = mean(IC) / std(IC)
    icir = monthly_ic.mean() / monthly_ic.std() if monthly_ic.std() > 0 else 0

    # IC 胜率
    ic_positive_rate = (monthly_ic > 0).mean() if len(monthly_ic) > 0 else 0

    result = {
        'ic': round(ic, 4),
        'rank_ic': round(rank_ic, 4),
        'icir': round(icir, 4),
        'monthly_ic_mean': round(monthly_ic.mean(), 4) if len(monthly_ic) > 0 else 0,
        'monthly_ic_std': round(monthly_ic.std(), 4) if len(monthly_ic) > 0 else 0,
        'ic_positive_rate': round(ic_positive_rate, 4),
        'monthly_ic_series': monthly_ic,
    }

    # IC 衰减分析
    if horizons:
        decay = {}
        for h in horizons:
            shifted_ret = df['return'].shift(-h)
            valid = pd.DataFrame({'f': df['factor'], 'r': shifted_ret}).dropna()
            if len(valid) > 20:
                decay[h] = round(valid['f'].corr(valid['r'], method='spearman'), 4)
        result['ic_decay'] = decay

    return result


def generate_ic_report_html(ic_results, factor_name='因子', save_path=None):
    """
    生成 IC 分析的 HTML 片段

    参数:
        ic_results: dict, 来自 calc_ic_analysis 的结果
        factor_name: 因子名称
        save_path: 保存路径(可选)

    返回:
        str, HTML 内容
    """
    html = f"""
    <div class="ic-report">
    <h3>{factor_name} - IC/RankIC 分析</h3>
    <table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse;">
      <tr><th>指标</th><th>数值</th><th>说明</th></tr>
      <tr><td>Pearson IC</td><td>{ic_results.get('ic', 0):.4f}</td>
          <td>因子值与收益率的线性相关</td></tr>
      <tr><td>Spearman RankIC</td><td>{ic_results.get('rank_ic', 0):.4f}</td>
          <td>因子排名与收益排名的相关(更稳健)</td></tr>
      <tr><td>ICIR</td><td>{ic_results.get('icir', 0):.4f}</td>
          <td>IC均值/IC标准差, 衡量IC稳定性</td></tr>
      <tr><td>IC均值</td><td>{ic_results.get('monthly_ic_mean', 0):.4f}</td>
          <td>月度IC的平均值</td></tr>
      <tr><td>IC标准差</td><td>{ic_results.get('monthly_ic_std', 0):.4f}</td>
          <td>月度IC的波动</td></tr>
      <tr><td>IC胜率</td><td>{ic_results.get('ic_positive_rate', 0):.2%}</td>
          <td>IC为正的月份占比</td></tr>
    </table>
    """

    if 'ic_decay' in ic_results:
        html += '<h4>IC 衰减分析</h4><table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">'
        html += '<tr><th>Horizon(天)</th><th>RankIC</th></tr>'
        for h, val in ic_results['ic_decay'].items():
            html += f'<tr><td>{h}</td><td>{val:.4f}</td></tr>'
        html += '</table>'

    html += '</div>'

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w', encoding='utf-8-sig') as f:
            f.write(html)
        print(f'  IC报告片段已保存: {save_path}')

    return html


# ============================================================
# 综合报告生成
# ============================================================

def generate_comprehensive_report(sections, title='综合绩效分析报告',
                                   output_path='outputs/comprehensive_report.html'):
    """
    生成综合分析报告(多章节拼装 HTML)

    参数:
        sections: list of dict, 每个 dict 含:
            - title: str, 章节标题
            - content: str, HTML 内容
            - charts: list of str, 图片路径(可选)
        title: 报告总标题
        output_path: 输出文件路径

    返回:
        str, 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>{title}</title>
<style>
body {{ font-family: 'Microsoft YaHei', Arial, sans-serif; margin: 40px; background: #f5f5f5; color: #333; }}
.container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 40px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #2980b9; border-left: 4px solid #3498db; padding-left: 12px; margin-top: 30px; }}
h3 {{ color: #34495e; }}
table {{ border-collapse: collapse; margin: 15px 0; width: 100%; }}
table th {{ background: #3498db; color: white; padding: 10px; text-align: left; }}
table td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
table tr:hover {{ background: #f8f9fa; }}
.metric-card {{ display: inline-block; background: #ecf0f1; padding: 15px 20px; margin: 5px; border-radius: 8px; min-width: 150px; text-align: center; }}
.metric-card .value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
.metric-card .label {{ font-size: 12px; color: #7f8c8d; margin-top: 5px; }}
.chart-img {{ max-width: 100%; margin: 15px 0; border: 1px solid #ddd; border-radius: 4px; }}
.section {{ margin-bottom: 40px; }}
.footer {{ text-align: center; color: #95a5a6; margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
<h1>{title}</h1>
<p style="color:#7f8c8d;">生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

    for idx, section in enumerate(sections, 1):
        html += f'<div class="section"><h2>第{idx}章 {section["title"]}</h2>'
        html += section.get('content', '')

        for chart_path in section.get('charts', []):
            if os.path.exists(chart_path):
                import base64
                with open(chart_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                html += f'<img class="chart-img" src="data:image/png;base64,{img_data}" />'

        html += '</div>'

    html += """
<div class="footer">
  <p>本报告由 QuantStats 绩效分析引擎自动生成</p>
</div>
</div>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8-sig') as f:
        f.write(html)

    print(f'  综合报告已生成: {output_path}')
    return output_path


def metrics_to_html_cards(metrics, card_keys=None):
    """将指标 dict 转为 HTML 卡片展示"""
    if card_keys is None:
        card_keys = ['total_return', 'cagr', 'max_drawdown', 'sharpe',
                     'sortino', 'calmar', 'win_rate', 'profit_factor']

    label_map = {
        'total_return': '总收益率',
        'cagr': '年化收益(CAGR)',
        'max_drawdown': '最大回撤',
        'sharpe': '夏普比率',
        'sortino': '索提诺比率',
        'calmar': '卡玛比率',
        'omega': 'Omega比率',
        'volatility': '年化波动率',
        'win_rate': '日胜率',
        'profit_factor': '利润因子',
        'var_95': 'VaR(95%)',
        'cvar_95': 'CVaR(95%)',
    }

    fmt_pct = {'total_return', 'cagr', 'max_drawdown', 'volatility',
               'win_rate', 'var_95', 'cvar_95'}

    html = '<div style="margin:15px 0;">'
    for key in card_keys:
        if key not in metrics or metrics[key] is None:
            continue
        val = float(metrics[key])
        label = label_map.get(key, key)
        if key in fmt_pct:
            val_str = f'{val:.2%}'
        else:
            val_str = f'{val:.4f}'
        html += f'<div class="metric-card"><div class="value">{val_str}</div><div class="label">{label}</div></div>'
    html += '</div>'
    return html


def metrics_to_html_table(metrics):
    """将指标 dict 转为 HTML 表格"""
    label_map = {
        'total_return': ('总收益率', True),
        'cagr': ('年化收益率(CAGR)', True),
        'volatility': ('年化波动率', True),
        'max_drawdown': ('最大回撤', True),
        'sharpe': ('夏普比率', False),
        'sortino': ('索提诺比率', False),
        'calmar': ('卡玛比率', False),
        'omega': ('Omega比率', False),
        'gain_to_pain': ('盈亏比', False),
        'var_95': ('VaR(95%)', True),
        'cvar_95': ('CVaR(95%)', True),
        'skew': ('偏度', False),
        'kurtosis': ('峰度', False),
        'win_rate': ('日胜率', True),
        'profit_factor': ('利润因子', False),
        'best_day': ('最佳单日', True),
        'worst_day': ('最差单日', True),
        'alpha': ('Alpha', True),
        'beta': ('Beta', False),
        'information_ratio': ('信息比率', False),
        'tracking_error': ('跟踪误差', True),
    }

    html = '<table><tr><th>指标</th><th>数值</th></tr>'
    for key, (label, is_pct) in label_map.items():
        if key not in metrics or metrics[key] is None:
            continue
        val = float(metrics[key])
        val_str = f'{val:.2%}' if is_pct else f'{val:.4f}'
        html += f'<tr><td>{label}</td><td>{val_str}</td></tr>'
    html += '</table>'
    return html
