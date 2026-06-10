# -*- coding: utf-8 -*-
"""
QuantStats 绩效分析 -- 专业级量化绩效分析工具

核心内容:
  1. QuantStats 全量指标体系 (30+ 指标)
  2. 从 Backtrader 回测结果导入交易单, 自动绩效分析
  3. 可视化: 累计收益/回撤水下图/月度热力图/滚动夏普
  4. 基准对比: vs 沪深300 (Alpha/Beta/信息比率/跟踪误差)
  5. 生成独立 HTML 报告 (可直接发给客户/领导)
  6. Backtrader 内置指标 vs QuantStats 指标对比

运行: python 1-QuantStats绩效分析.py
"""
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

import backtrader as bt
import quantstats as qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (load_stock_data, run_and_report, MLSignalData,
                          INITIAL_CASH, COMMISSION, POSITION_PCT)
from report_engine import (nav_to_returns, backtrader_nav_to_series,
                            trade_log_to_dataframe, calc_quantstats_metrics,
                            print_metrics_table, plot_returns_chart,
                            plot_strategy_comparison, generate_html_report,
                            generate_chinese_report)


# ============================================================
# 双均线策略 (从 L5 复用)
# ============================================================

class DoubleMAStrategy(bt.Strategy):
    """双均线金叉/死叉策略"""
    params = (('fast', 10), ('slow', 30))

    def __init__(self):
        self.ma_fast = bt.indicators.SMA(self.data.close, period=self.p.fast)
        self.ma_slow = bt.indicators.SMA(self.data.close, period=self.p.slow)
        self.crossover = bt.indicators.CrossOver(self.ma_fast, self.ma_slow)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()


# ============================================================
# MACD策略
# ============================================================

class MACDStrategy(bt.Strategy):
    """MACD金叉死叉策略"""
    params = (('fast', 12), ('slow', 26), ('signal', 9))

    def __init__(self):
        self.macd = bt.indicators.MACD(
            self.data.close,
            period_me1=self.p.fast,
            period_me2=self.p.slow,
            period_signal=self.p.signal,
        )
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()


# ============================================================
# RSI策略
# ============================================================

class RSIStrategy(bt.Strategy):
    """RSI超买超卖策略"""
    params = (('period', 14), ('upper', 70), ('lower', 30))

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)

    def next(self):
        if not self.position:
            if self.rsi[0] < self.p.lower:
                self.buy()
        elif self.rsi[0] > self.p.upper:
            self.close()


OUTPUT_DIR = 'outputs/1-QuantStats绩效分析'


def generate_index_html(output_dir, all_metrics=None, comparison_summary=None,
                        best_strategy=None):
    """
    在输出目录中生成 index.html, 聚合所有报告/图表/数据文件。
    包含背景说明、策略指标、图表嵌入、LLM分析结论。
    """
    import base64

    html_files = []
    png_files = []
    csv_files = []

    for f in sorted(os.listdir(output_dir)):
        full = os.path.join(output_dir, f)
        if not os.path.isfile(full) or f == 'index.html':
            continue
        if f.endswith('.html'):
            html_files.append(f)
        elif f.endswith('.png'):
            png_files.append(f)
        elif f.endswith('.csv'):
            csv_files.append(f)

    def _embed_png(filename):
        """将PNG文件转为base64 img标签"""
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            return ''
        with open(path, 'rb') as img:
            b64 = base64.b64encode(img.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border:1px solid #ddd;border-radius:6px;">'

    # 策略指标表格
    metrics_table = ''
    if all_metrics:
        metrics_table = '<table style="width:100%;border-collapse:collapse;margin:12px 0;">'
        metrics_table += ('<tr style="background:#2980b9;color:white;">'
                          '<th style="padding:10px;text-align:left;">策略</th>'
                          '<th style="padding:10px;">总收益</th>'
                          '<th style="padding:10px;">年化收益</th>'
                          '<th style="padding:10px;">最大回撤</th>'
                          '<th style="padding:10px;">夏普比率</th>'
                          '<th style="padding:10px;">索提诺</th>'
                          '<th style="padding:10px;">日胜率</th></tr>')
        for name, m in all_metrics.items():
            is_best = (name == best_strategy)
            row_style = 'background:#e8f5e9;font-weight:bold;' if is_best else ''
            badge = ' [最优]' if is_best else ''
            metrics_table += (
                f'<tr style="{row_style}">'
                f'<td style="padding:8px;border-bottom:1px solid #eee;">{name}{badge}</td>'
                f'<td style="padding:8px;text-align:center;border-bottom:1px solid #eee;">{m.get("total_return",0):.2%}</td>'
                f'<td style="padding:8px;text-align:center;border-bottom:1px solid #eee;">{m.get("cagr",0):.2%}</td>'
                f'<td style="padding:8px;text-align:center;border-bottom:1px solid #eee;">{m.get("max_drawdown",0):.2%}</td>'
                f'<td style="padding:8px;text-align:center;border-bottom:1px solid #eee;">{m.get("sharpe",0):.4f}</td>'
                f'<td style="padding:8px;text-align:center;border-bottom:1px solid #eee;">{m.get("sortino",0):.4f}</td>'
                f'<td style="padding:8px;text-align:center;border-bottom:1px solid #eee;">{m.get("win_rate",0):.2%}</td>'
                f'</tr>')
        metrics_table += '</table>'

    # LLM结论
    conclusion_html = ''
    if comparison_summary:
        try:
            from llm_engine import summary_to_html
            conclusion_html = summary_to_html(comparison_summary, title='AI 分析结论')
        except ImportError:
            conclusion_html = f'<div style="background:#f8f9fa;padding:16px;border-radius:8px;border-left:4px solid #2980b9;margin:16px 0;"><h3 style="color:#2980b9;margin-top:0;">AI 分析结论</h3><pre style="white-space:pre-wrap;font-size:13px;">{comparison_summary}</pre><p style="font-size:11px;color:#95a5a6;">* 以上分析由 AI 大模型自动生成, 仅供参考</p></div>'

    html_links = '\n'.join(
        f'<li><a href="{f}" target="_blank">{f.replace(".html","")}</a></li>'
        for f in html_files
    )
    csv_links = '\n'.join(
        f'<li><a href="{f}" download>{f}</a></li>'
        for f in csv_files
    )

    # 找到特定图表用于嵌入
    perf_chart = ''
    rolling_chart = ''
    compare_chart = ''
    for pf in png_files:
        if 'QuantStats绩效分析' in pf or '绩效分析' in pf:
            perf_chart = _embed_png(pf)
        elif '滚动分析' in pf:
            rolling_chart = _embed_png(pf)
        elif '策略对比' in pf or '对比' in pf:
            compare_chart = _embed_png(pf)

    # 如果没匹配到特定名称, 按顺序分配
    remaining_charts = []
    for pf in png_files:
        img_html = _embed_png(pf)
        title_text = pf.replace('.png', '').replace('_', ' ')
        if img_html not in [perf_chart, rolling_chart, compare_chart]:
            remaining_charts.append(f'<div style="margin:12px 0;"><h4 style="color:#34495e;">{title_text}</h4>{img_html}</div>')

    index_html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>QuantStats 绩效分析 - 分析报告</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: "Microsoft YaHei","SimHei",sans-serif;
         background:#f5f7fa; color:#333; padding:30px; }}
  .container {{ max-width:1200px; margin:0 auto; }}
  h1 {{ text-align:center; color:#2c3e50; margin-bottom:8px; font-size:28px; }}
  .subtitle {{ text-align:center; color:#7f8c8d; margin-bottom:30px; font-size:14px; }}
  .section {{ background:#fff; border-radius:10px; padding:24px;
              margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,0.08); }}
  .section h2 {{ color:#2980b9; border-bottom:2px solid #2980b9;
                 padding-bottom:8px; margin-bottom:16px; font-size:20px; }}
  ul {{ list-style:none; padding:0; }}
  ul li {{ padding:8px 12px; border-bottom:1px solid #eee; }}
  ul li:last-child {{ border-bottom:none; }}
  ul li a {{ color:#2980b9; text-decoration:none; font-size:15px; }}
  ul li a:hover {{ color:#e74c3c; text-decoration:underline; }}
  .stats {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:20px; }}
  .stat-box {{ background:#ecf0f1; border-radius:8px; padding:16px 24px;
               flex:1; min-width:150px; text-align:center; }}
  .stat-box .num {{ font-size:28px; font-weight:bold; color:#2980b9; }}
  .stat-box .label {{ font-size:13px; color:#7f8c8d; margin-top:4px; }}
  table th {{ text-align:center; }}
  pre {{ font-family: Consolas, monospace; }}
</style>
</head>
<body>
<div class="container">
  <h1>QuantStats 绩效分析报告</h1>
  <p class="subtitle">Backtrader 多策略回测 + QuantStats 全量绩效分析</p>

  <div class="section">
    <h2>分析背景</h2>
    <p style="line-height:1.8;font-size:14px;">
      本报告使用 <strong>QuantStats</strong> 对多个量化策略进行了全面的绩效分析。
      分析标的为 <strong>贵州茅台(600519.SH)</strong>, 回测区间为
      <strong>2024-01-01 ~ 2025-12-31</strong>。
    </p>
    <p style="line-height:1.8;font-size:14px;margin-top:8px;">
      共测试了 <strong>3个策略</strong>: 双均线(金叉死叉)、MACD(信号线交叉)、RSI(超买超卖)。
      通过 30+ 绩效指标的横向对比, 找出风险调整后收益最优的策略。
    </p>
  </div>

  <div class="stats">
    <div class="stat-box">
      <div class="num">{len(html_files)}</div>
      <div class="label">HTML 报告</div>
    </div>
    <div class="stat-box">
      <div class="num">{len(png_files)}</div>
      <div class="label">分析图表</div>
    </div>
    <div class="stat-box">
      <div class="num">{len(csv_files)}</div>
      <div class="label">数据文件</div>
    </div>
  </div>

  <div class="section">
    <h2>策略绩效对比</h2>
    {metrics_table if metrics_table else '<p>暂无指标数据</p>'}
    {compare_chart}
  </div>

  <div class="section">
    <h2>绩效分析图表</h2>
    {f'<div style="margin:12px 0;"><h4 style="color:#34495e;">策略绩效分析</h4>{perf_chart}</div>' if perf_chart else ''}
    {f'<div style="margin:12px 0;"><h4 style="color:#34495e;">滚动风险指标</h4>{rolling_chart}</div>' if rolling_chart else ''}
    {''.join(remaining_charts)}
  </div>

  {f'<div class="section"><h2>分析结论</h2>{conclusion_html}</div>' if conclusion_html else ''}

  <div class="section">
    <h2>QuantStats 详细报告 (点击查看)</h2>
    <ul>
      {html_links if html_links else '<li>暂无HTML报告</li>'}
    </ul>
  </div>

  <div class="section">
    <h2>数据文件 (点击下载)</h2>
    <ul>
      {csv_links if csv_links else '<li>暂无数据文件</li>'}
    </ul>
  </div>
</div>
</body>
</html>'''

    index_path = os.path.join(output_dir, 'index.html')
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_html)
    print(f'  索引页已生成: {index_path}')
    return index_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    stock_code = '600519.SH'
    start_date = '2024-01-01'
    end_date = '2025-12-31'

    # ============================================================
    # 场景导入
    # ============================================================
    print('=' * 70)
    print('场景: 量化投资经理的一天')
    print('=' * 70)
    print("""
    背景:
      你是某私募基金的量化投资经理, 管理着一个以A股为主的量化组合。
      上周, 你用 Backtrader 开发了3个策略(双均线/MACD/RSI),
      分别在贵州茅台上做了回测。

      今天领导要你提交一份专业的绩效分析报告:
      - 每个策略的核心指标(夏普/索提诺/最大回撤等)
      - 与沪深300基准的对比(Alpha/Beta/信息比率)
      - 哪个策略最适合当前市场?
      - 一份可以直接发给客户的HTML报告

      用 QuantStats, 这些需求可以在5分钟内完成。
    """)

    # ============================================================
    # 第一部分: QuantStats 基础 -- 从 Backtrader 导入交易单
    # ============================================================
    print('=' * 70)
    print('第一部分: QuantStats 绩效分析入门')
    print('=' * 70)

    print(f'\n[1] 运行 Backtrader 回测: 双均线策略  {stock_code}')
    result = run_and_report(
        DoubleMAStrategy, stock_code, start_date, end_date,
        label='双均线策略', plot=True
    )

    # -- 提取净值序列 --
    nav_series = backtrader_nav_to_series(result['nav'], INITIAL_CASH)
    print(f'    净值序列: {len(nav_series)} 个交易日')
    print(f'    起始净值: {nav_series.iloc[0]:.4f}')
    print(f'    最终净值: {nav_series.iloc[-1]:.4f}')

    # -- 转为日收益率 (QuantStats 的核心输入) --
    returns = nav_to_returns(nav_series)
    print(f'    收益率序列: {len(returns)} 天')

    # -- 导出交易记录 --
    trade_df = trade_log_to_dataframe(result['trades'])
    print(f'\n[2] 交易记录导出: {len(trade_df)} 笔交易')
    if not trade_df.empty:
        trade_csv = f'{OUTPUT_DIR}/双均线策略_交易记录.csv'
        trade_df.to_csv(trade_csv, index=False, encoding='utf-8-sig')
        print(f'    已保存: {trade_csv}')
        print(trade_df.to_string(index=False))

    # ============================================================
    # 第二部分: QuantStats 30+ 指标全览
    # ============================================================
    print(f'\n{"=" * 70}')
    print('第二部分: QuantStats 全量指标体系')
    print(f'{"=" * 70}')

    # 加载基准: 沪深300
    benchmark_returns = None
    try:
        bench_df = load_stock_data('000300.SH', start_date, end_date)
        if len(bench_df) > 20:
            bench_nav = bench_df['close'] / bench_df['close'].iloc[0]
            benchmark_returns = bench_nav.pct_change().dropna()
            benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
            print(f'\n  基准: 沪深300 ({len(bench_df)} 个交易日)')
    except Exception as e:
        print(f'  [提示] 沪深300数据加载失败({e}), 跳过基准对比')

    # -- 计算 QuantStats 全量指标 --
    metrics = calc_quantstats_metrics(returns, benchmark=benchmark_returns)
    print_metrics_table(metrics, '双均线策略 - QuantStats 全量指标')

    # ============================================================
    # 第三部分: Backtrader 内置 vs QuantStats 对比
    # ============================================================
    print(f'\n{"=" * 70}')
    print('第三部分: Backtrader 内置指标 vs QuantStats 对比')
    print(f'{"=" * 70}')

    print(f'\n  {"指标":<20s}  {"Backtrader":>14s}  {"QuantStats":>14s}  {"差异说明"}')
    print(f'  {"-"*20}  {"-"*14}  {"-"*14}  {"-"*20}')

    bt_total = result['total_return']
    qs_total = metrics['total_return']
    print(f'  {"总收益率":<18s}  {bt_total:>13.2%}  {qs_total:>13.2%}  '
          f'{"一致" if abs(bt_total - qs_total) < 0.01 else "计算方式微差"}')

    bt_sharpe = result['sharpe_ratio']
    qs_sharpe = metrics['sharpe']
    print(f'  {"夏普比率":<18s}  {bt_sharpe:>14.4f}  {qs_sharpe:>14.4f}  '
          f'BT用年化/QuantStats用日频*sqrt(252)')

    bt_dd = result['max_drawdown']
    qs_dd = abs(metrics['max_drawdown'])
    print(f'  {"最大回撤":<18s}  {bt_dd:>13.2%}  {qs_dd:>13.2%}  '
          f'{"一致" if abs(bt_dd - qs_dd) < 0.01 else "净值精度差异"}')

    print(f'\n  [QuantStats 独有指标]')
    exclusive = ['sortino', 'calmar', 'omega', 'var_95', 'cvar_95',
                 'skew', 'kurtosis', 'gain_to_pain']
    label_map = {
        'sortino': '索提诺比率', 'calmar': '卡玛比率', 'omega': 'Omega比率',
        'var_95': 'VaR(95%)', 'cvar_95': 'CVaR(95%)',
        'skew': '偏度', 'kurtosis': '峰度', 'gain_to_pain': '盈亏比',
    }
    for key in exclusive:
        if key in metrics and metrics[key] is not None:
            val = metrics[key]
            name = label_map.get(key, key)
            if key in ('var_95', 'cvar_95'):
                print(f'    {name}: {val:.2%}')
            else:
                print(f'    {name}: {val:.4f}')

    # ============================================================
    # 第四部分: 可视化 -- 专业图表
    # ============================================================
    print(f'\n{"=" * 70}')
    print('第四部分: 专业可视化')
    print(f'{"=" * 70}')

    # -- 综合绩效图 --
    plot_returns_chart(returns, benchmark=benchmark_returns,
                       title='双均线策略_QuantStats绩效分析',
                       save_dir=OUTPUT_DIR)

    # -- 滚动夏普比率 --
    print('\n[滚动分析]')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))

    rolling_sharpe = returns.rolling(60).apply(
        lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
    )
    ax1.plot(rolling_sharpe.index, rolling_sharpe.values, '#2980b9', linewidth=1)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    ax1.set_ylabel('滚动夏普(60日)')
    ax1.set_title('滚动风险指标', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    rolling_vol = returns.rolling(20).std() * np.sqrt(252)
    ax2.plot(rolling_vol.index, rolling_vol.values, '#e74c3c', linewidth=1)
    ax2.set_ylabel('滚动波动率(20日)')
    ax2.set_xlabel('日期')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/双均线策略_滚动分析.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  图表已保存: {OUTPUT_DIR}/双均线策略_滚动分析.png')

    # ============================================================
    # 第五部分: 多策略对比
    # ============================================================
    print(f'\n{"=" * 70}')
    print('第五部分: 多策略 QuantStats 对比')
    print(f'{"=" * 70}')

    strategies = {
        '双均线': (DoubleMAStrategy, {}),
        'MACD': (MACDStrategy, {}),
        'RSI': (RSIStrategy, {}),
    }

    nav_dict = {}
    all_metrics = {}

    for name, (strat_cls, kwargs) in strategies.items():
        print(f'\n  运行: {name} 策略...')
        r = run_and_report(strat_cls, stock_code, start_date, end_date,
                           label=name, quiet=True, **kwargs)
        nav = backtrader_nav_to_series(r['nav'], INITIAL_CASH)
        ret = nav_to_returns(nav)
        m = calc_quantstats_metrics(ret, benchmark=benchmark_returns)
        nav_dict[name] = nav
        all_metrics[name] = m

    # -- 对比表 --
    print(f'\n  {"策略":<10s} {"总收益":>10s} {"年化":>10s} {"最大回撤":>10s} '
          f'{"夏普":>8s} {"索提诺":>8s} {"卡玛":>8s} {"日胜率":>8s}')
    print(f'  {"-"*10} {"-"*10} {"-"*10} {"-"*10} '
          f'{"-"*8} {"-"*8} {"-"*8} {"-"*8}')
    for name, m in all_metrics.items():
        print(f'  {name:<10s} {m["total_return"]:>9.2%} {m["cagr"]:>9.2%} '
              f'{m["max_drawdown"]:>9.2%} {m["sharpe"]:>8.4f} '
              f'{m["sortino"]:>8.4f} {m["calmar"]:>8.4f} '
              f'{m["win_rate"]:>7.2%}')

    # -- 多策略对比图 --
    plot_strategy_comparison(nav_dict, title='三策略对比_QuantStats',
                             save_dir=OUTPUT_DIR)

    # ============================================================
    # 第六部分: 生成 HTML 报告
    # ============================================================
    print(f'\n{"=" * 70}')
    print('第六部分: 生成 QuantStats HTML 报告')
    print(f'{"=" * 70}')

    # -- 策略报告 (英文版: QuantStats原版, 中文版: 自主渲染) --
    for name in strategies:
        nav = nav_dict[name]
        ret = nav_to_returns(nav)

        en_path = f'{OUTPUT_DIR}/QuantStats报告_{name}_英文.html'
        generate_html_report(ret, benchmark=benchmark_returns,
                              title=f'{name} Strategy - QuantStats Report',
                              output_path=en_path)

        zh_path = f'{OUTPUT_DIR}/QuantStats报告_{name}.html'
        generate_chinese_report(ret, benchmark=benchmark_returns,
                                 title=f'{name}策略 - 绩效分析报告',
                                 output_path=zh_path)

    # ============================================================
    # 第七部分: 导出交易单 + 净值数据
    # ============================================================
    print(f'\n{"=" * 70}')
    print('第七部分: 导出数据文件')
    print(f'{"=" * 70}')

    # -- 导出每日净值 --
    for name in strategies:
        nav = nav_dict[name]
        nav_df = pd.DataFrame({'date': nav.index, 'nav': nav.values})
        csv_path = f'{OUTPUT_DIR}/{name}策略_每日净值.csv'
        nav_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f'  {name} 净值已导出: {csv_path}')

    # -- 从 CSV 导入并重新分析(演示 QuantStats 导入外部数据) --
    print(f'\n[演示] 从 CSV 文件导入净值, 重新进行 QuantStats 分析:')
    reload_df = pd.read_csv(f'{OUTPUT_DIR}/双均线策略_每日净值.csv')
    reload_df['date'] = pd.to_datetime(reload_df['date'])
    reload_df.set_index('date', inplace=True)
    reload_returns = reload_df['nav'].pct_change().dropna()
    reload_metrics = calc_quantstats_metrics(reload_returns)
    print(f'  从CSV重新计算 -- 夏普: {reload_metrics["sharpe"]:.4f}  '
          f'总收益: {reload_metrics["total_return"]:.2%}  '
          f'最大回撤: {reload_metrics["max_drawdown"]:.2%}')

    # ============================================================
    # 第八部分: LLM 智能分析
    # ============================================================
    print(f'\n{"=" * 70}')
    print('第八部分: LLM 智能分析(qwen-max)')
    print(f'{"=" * 70}')

    comparison_summary = None
    best_strategy = None
    try:
        from llm_engine import generate_comparison_summary

        # 找出综合最优策略 (按夏普比率排序)
        best_strategy = max(all_metrics, key=lambda k: all_metrics[k].get('sharpe', 0))
        print(f'\n  夏普最优策略: {best_strategy}')

        comparison_summary = generate_comparison_summary(all_metrics)
        if comparison_summary:
            print(f'\n  ---- AI 分析结论 ----')
            print(f'  {comparison_summary}')
    except ImportError:
        print('  [提示] llm_engine 未找到, 跳过LLM分析')
    except Exception as e:
        print(f'  [提示] LLM分析失败: {e}')

    # ============================================================
    # 课程小结
    # ============================================================
    print(f'\n{"=" * 70}')
    print('课程小结')
    print(f'{"=" * 70}')
    print("""
    QuantStats 核心能力:
    1. 一行代码计算 30+ 绩效指标 (Sharpe/Sortino/Calmar/VaR/CVaR...)
    2. 从 Backtrader 无缝导入净值和交易记录
    3. 一键生成专业中文 HTML 报告 (翻译配置见 translations.yaml)
    4. 支持基准对比 (Alpha/Beta/信息比率/跟踪误差)
    5. 内置丰富可视化 (累计收益/回撤/月度热力图/滚动指标)
    6. 可从 CSV/DataFrame 导入任意外部数据源进行分析
    7. 集成 LLM 大模型, 自动生成专业分析结论

    实际应用场景:
    - [策略回测] 用 Backtrader 开发策略后, 自动生成绩效报告
    - [策略PK]   多策略横向比较, 按夏普/卡玛等指标选择最优
    - [客户报告] 生成专业 HTML 报告(已中文化), 直接发给客户/领导
    - [风控监控] 用滚动夏普/波动率实时监控策略健康度
    - [实盘复盘] 导出历史交易, 用 QuantStats 分析实盘绩效

    回到我们的场景:
      作为量化投资经理, 你已经用 QuantStats 完成了:
      1. 3个策略的绩效评估 -> 找到最优策略
      2. 与沪深300的基准对比 -> 量化超额收益
      3. 一键生成HTML报告 -> 直接发给领导
      4. AI生成分析结论 -> 报告更加专业
      5. 导出CSV文件 -> 便于后续跟踪和复盘

      下一步:
      脚本2 -> SVD因子挖掘, 发现市场隐含驱动因子
      脚本3 -> 实盘交易记录绩效分析
      脚本4 -> 增强版报告(集成SVD市场状态诊断)
    """)

    # ============================================================
    # 生成 index.html 汇总页
    # ============================================================
    print(f'\n{"=" * 70}')
    print('生成汇总索引页')
    print(f'{"=" * 70}')
    generate_index_html(OUTPUT_DIR, all_metrics=all_metrics,
                        comparison_summary=comparison_summary,
                        best_strategy=best_strategy)


if __name__ == '__main__':
    main()
