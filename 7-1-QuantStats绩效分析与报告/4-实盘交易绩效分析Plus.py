# -*- coding: utf-8 -*-
"""
实盘交易绩效分析 Plus -- 集成 SVD 市场状态诊断的增强版报告

在基础版(脚本3)的基础上, 增加:
  1. SVD 市场状态诊断: 判断交易期间处于齐涨齐跌/板块分化/个股行情
  2. 因子集中度变化趋势: 用滚动SVD追踪市场驱动因子的时变性
  3. 投资建议: 根据当前市场状态给出策略选择参考
  4. 报告中嵌入 SVD 诊断结果图表

核心思路:
  对交易涉及的股票进行 SVD 分解, 观察第一因子(市场因子)的方差占比:
    > 50%  -> 齐涨齐跌, beta主导, 做指数增强更有效
    35%-50% -> 板块分化, 行业轮动机会
    < 35%  -> 个股行情, alpha机会多, 选股策略更有效

运行: python 4-实盘交易绩效分析Plus.py
"""
import os
import sys
import base64
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from report_engine import (nav_to_returns, calc_quantstats_metrics,
                            print_metrics_table, plot_returns_chart,
                            generate_html_report, generate_chinese_report,
                            generate_comprehensive_report,
                            metrics_to_html_cards, metrics_to_html_table,
                            configure_matplotlib_chinese)

configure_matplotlib_chinese()
from data_loader import load_stock_data

# 复用脚本3的CSV解析和分析函数
from importlib import import_module
_script3 = import_module('3-实盘交易绩效分析')

code_to_standard = _script3.code_to_standard
load_broker_csv = _script3.load_broker_csv
analyze_by_stock = _script3.analyze_by_stock
build_portfolio_nav = _script3.build_portfolio_nav
analyze_costs = _script3.analyze_costs
plot_stock_pnl = _script3.plot_stock_pnl
plot_cost_breakdown = _script3.plot_cost_breakdown

OUTPUT_DIR = 'outputs/4-实盘交易Plus'


# ============================================================
# SVD 市场状态诊断
# ============================================================

def diagnose_market_regime(stock_codes, start_date, end_date, window=120, step=20):
    """
    对指定股票进行滚动 SVD, 诊断交易期间的市场状态

    参数:
        stock_codes: list, 标准股票代码列表 (如 ['600519.SH', '002432.SZ'])
        start_date: str, 开始日期
        end_date: str, 结束日期
        window: int, 滚动窗口大小(交易日)
        step: int, 滚动步长

    返回:
        dict: 诊断结果, 包含:
          - rolling_df: DataFrame, 滚动SVD结果
          - current_state: str, 当前市场状态
          - current_f1_ratio: float, 当前第一因子方差占比
          - advice: str, 投资建议
          - chart_path: str, SVD趋势图路径
    """
    # 构建收益率矩阵
    returns_dict = {}
    for code in stock_codes:
        try:
            df = load_stock_data(code, start_date, end_date)
            if len(df) > 20:
                ret = df['close'].pct_change().dropna()
                returns_dict[code] = ret
        except Exception:
            pass

    if len(returns_dict) < 3:
        print(f'  [SVD诊断] 有效股票不足3只 (当前{len(returns_dict)}只), 跳过SVD分析')
        return None

    returns_df = pd.DataFrame(returns_dict).dropna()
    if len(returns_df) < window + step:
        print(f'  [SVD诊断] 数据不足 (需要{window+step}天, 实际{len(returns_df)}天), 跳过')
        return None

    # 滚动 SVD
    T = returns_df.shape[0]
    rolling_results = []

    for start in range(0, T - window, step):
        end_idx = start + window
        window_data = returns_df.iloc[start:end_idx]
        R_w = window_data.values.T
        R_w = R_w - R_w.mean(axis=1, keepdims=True)

        _, sigma_w, _ = np.linalg.svd(R_w, full_matrices=False)
        total_var = np.sum(sigma_w ** 2)
        if total_var == 0:
            continue
        top1_ratio = sigma_w[0] ** 2 / total_var
        top3_ratio = np.sum(sigma_w[:min(3, len(sigma_w))] ** 2) / total_var

        mid_date = window_data.index[window // 2]
        rolling_results.append({
            'date': mid_date,
            'top1_var': top1_ratio,
            'top3_var': top3_ratio,
        })

    if not rolling_results:
        return None

    roll_df = pd.DataFrame(rolling_results).set_index('date')

    # 当前状态判断 (取最近3个窗口的平均)
    recent_f1 = roll_df['top1_var'].iloc[-min(3, len(roll_df)):].mean()

    if recent_f1 > 0.50:
        state = '齐涨齐跌'
        advice = ('当前市场齐涨齐跌特征明显, beta因子主导。'
                  '建议: 指数增强策略更有效, 个股选择的alpha空间有限。'
                  '可考虑: 增大仓位跟随大盘趋势, 减少个股博弈。')
    elif recent_f1 > 0.35:
        state = '板块分化'
        advice = ('当前市场处于板块分化阶段, 行业轮动特征显著。'
                  '建议: 行业配置是关键, 选对板块比选对个股更重要。'
                  '可考虑: 关注行业动量因子, 超配强势板块。')
    else:
        state = '个股行情'
        advice = ('当前市场个股分化明显, alpha机会丰富。'
                  '建议: 选股策略更有效, 多因子模型价值凸显。'
                  '可考虑: 精选个股, 降低对大盘方向的依赖。')

    # 绘制SVD趋势图
    chart_path = f'{OUTPUT_DIR}/SVD市场状态诊断.png'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(roll_df.index, roll_df['top1_var'] * 100,
                     alpha=0.3, color='steelblue', label='Factor 1 方差占比')
    ax.fill_between(roll_df.index, roll_df['top3_var'] * 100,
                     alpha=0.15, color='darkorange', label='Top 3 方差占比')
    ax.plot(roll_df.index, roll_df['top1_var'] * 100,
            color='steelblue', linewidth=2)

    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (齐涨齐跌线)')
    ax.axhline(y=35, color='green', linestyle='--', alpha=0.5, label='35% (个股行情线)')

    # 标注区域
    ax.axhspan(50, 100, alpha=0.05, color='red')
    ax.axhspan(0, 35, alpha=0.05, color='green')

    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('方差解释比 (%)', fontsize=12)
    ax.set_title('滚动SVD: 市场因子集中度变化', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(roll_df['top3_var'].max() * 110, 60))

    plt.tight_layout()
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  图表已保存: {chart_path}')

    return {
        'rolling_df': roll_df,
        'current_state': state,
        'current_f1_ratio': recent_f1,
        'advice': advice,
        'chart_path': chart_path,
        'stock_count': len(returns_dict),
        'data_days': len(returns_df),
    }


def svd_section_to_html(svd_result):
    """将SVD诊断结果转为HTML报告板块"""
    if svd_result is None:
        return '<p>SVD市场诊断: 数据不足, 跳过</p>'

    state = svd_result['current_state']
    f1_ratio = svd_result['current_f1_ratio']
    advice = svd_result['advice']
    stock_count = svd_result['stock_count']
    data_days = svd_result['data_days']

    # 状态颜色
    if state == '齐涨齐跌':
        color = '#e74c3c'
    elif state == '板块分化':
        color = '#f39c12'
    else:
        color = '#27ae60'

    html = f'''
    <div style="background:#f8f9fa;border-radius:10px;padding:20px;margin:16px 0;">
      <h3 style="margin-top:0;">SVD 市场状态诊断</h3>
      <p style="font-size:13px;color:#7f8c8d;">
        基于 {stock_count} 只交易标的 {data_days} 个交易日的收益率矩阵,
        通过滚动SVD分解分析市场因子结构。
      </p>
      <div style="display:flex;gap:20px;margin:16px 0;">
        <div style="flex:1;background:white;border-radius:8px;padding:16px;text-align:center;
                    border:2px solid {color};">
          <div style="font-size:14px;color:#7f8c8d;">当前市场状态</div>
          <div style="font-size:28px;font-weight:bold;color:{color};margin:8px 0;">{state}</div>
          <div style="font-size:13px;color:#95a5a6;">Factor1 方差占比: {f1_ratio:.1%}</div>
        </div>
        <div style="flex:2;background:white;border-radius:8px;padding:16px;">
          <div style="font-size:14px;font-weight:bold;color:#2c3e50;margin-bottom:8px;">
            投资建议
          </div>
          <div style="font-size:13px;color:#34495e;line-height:1.6;">
            {advice}
          </div>
        </div>
      </div>
      <div style="margin-top:12px;">
        <p style="font-size:12px;color:#95a5a6;">
          解读: Factor1方差占比反映市场"一致性"程度 --
          &gt;50% 表示大部分股票同涨同跌(beta主导),
          &lt;35% 表示个股走势分化(alpha机会多)。
        </p>
      </div>
    </div>'''

    # 滚动SVD数据表
    roll_df = svd_result['rolling_df']
    html += '<h3>滚动SVD因子集中度</h3>'
    html += '<table><tr><th>时间</th><th>Factor1占比</th><th>Top3占比</th><th>市场状态</th></tr>'
    for idx, row in roll_df.iterrows():
        date_str = idx.strftime('%Y-%m')
        f1 = row['top1_var']
        f3 = row['top3_var']
        if f1 > 0.50:
            s = '齐涨齐跌'
            sc = '#e74c3c'
        elif f1 > 0.35:
            s = '板块分化'
            sc = '#f39c12'
        else:
            s = '个股行情'
            sc = '#27ae60'
        html += f'<tr><td>{date_str}</td><td>{f1:.1%}</td><td>{f3:.1%}</td>'
        html += f'<td style="color:{sc};font-weight:bold">{s}</td></tr>'
    html += '</table>'

    return html


# ============================================================
# 增强版综合报告
# ============================================================

def generate_plus_report(trades_df, stock_df, nav_series, metrics,
                          cost_info, charts, svd_result, output_path,
                          llm_sections=None):
    """组装增强版综合HTML报告 (包含SVD市场诊断 + LLM分析)"""
    sections = []

    # -- Ch1: 交易概览 --
    ch1 = '<h3>账户交易概览</h3>'
    total_stocks = len(trades_df['标准代码'].unique())
    total_trades = len(trades_df)
    buys = len(trades_df[trades_df['买卖方向'] == '买入'])
    sells = len(trades_df[trades_df['买卖方向'] == '卖出'])
    date_range = (f'{trades_df["成交日期"].min().strftime("%Y-%m-%d")} ~ '
                  f'{trades_df["成交日期"].max().strftime("%Y-%m-%d")}')

    ch1 += f'<p>交易区间: <strong>{date_range}</strong></p>'
    ch1 += f'<p>涉及标的: <strong>{total_stocks}</strong> 只 | '
    ch1 += f'总成交: <strong>{total_trades}</strong> 笔 '
    ch1 += f'(买入 {buys} 笔, 卖出 {sells} 笔)</p>'

    if not nav_series.empty:
        ch1 += '<h3>绩效指标</h3>'
        ch1 += metrics_to_html_cards(metrics)

    sections.append({
        'title': '交易概览与绩效指标',
        'content': ch1,
        'charts': [c for c in [charts.get('nav_chart')] if c],
    })

    # -- Ch2: SVD 市场状态诊断 (Plus独有) --
    ch2 = svd_section_to_html(svd_result)
    svd_charts = []
    if svd_result and svd_result.get('chart_path'):
        svd_charts = [svd_result['chart_path']]
    sections.append({
        'title': 'SVD 市场状态诊断',
        'content': ch2,
        'charts': svd_charts,
    })

    # -- Ch3: 个股盈亏分析 --
    ch3 = '<h3>个股交易明细</h3>'
    ch3 += '<table><tr><th>证券名称</th><th>买入金额</th><th>卖出金额</th>'
    ch3 += '<th>未平仓</th><th>已实现盈亏</th><th>总成本</th></tr>'
    for _, row in stock_df.iterrows():
        pnl_str = f'{row["已实现盈亏"]:+,.0f}' if pd.notna(row['已实现盈亏']) else '持仓中'
        pnl_color = '#27ae60' if pd.notna(row['已实现盈亏']) and row['已实现盈亏'] >= 0 else '#e74c3c'
        ch3 += f'<tr><td>{row["证券名称"]}</td>'
        ch3 += f'<td>{row["买入金额"]:,.0f}</td><td>{row["卖出金额"]:,.0f}</td>'
        ch3 += f'<td>{row["未平仓"]}</td>'
        ch3 += f'<td style="color:{pnl_color};font-weight:bold">{pnl_str}</td>'
        ch3 += f'<td>{row["总成本"]:,.0f}</td></tr>'
    ch3 += '</table>'

    sections.append({
        'title': '个股盈亏分析',
        'content': ch3,
        'charts': [c for c in [charts.get('pnl_chart')] if c],
    })

    # -- Ch4: 成本分析 --
    ch4 = '<h3>交易成本构成</h3>'
    ch4 += '<table><tr><th>项目</th><th>金额</th><th>占比</th></tr>'
    for key, label in [('commission', '佣金'), ('stamp_tax', '印花税'),
                        ('transfer_fee', '过户费')]:
        val = cost_info.get(key, 0)
        pct = val / cost_info['total_cost'] * 100 if cost_info['total_cost'] > 0 else 0
        ch4 += f'<tr><td>{label}</td><td>{val:,.0f} 元</td><td>{pct:.1f}%</td></tr>'
    ch4 += f'<tr style="font-weight:bold"><td>合计</td>'
    ch4 += f'<td>{cost_info["total_cost"]:,.0f} 元</td><td>100%</td></tr>'
    ch4 += '</table>'
    ch4 += f'<p>总成交额: {cost_info["total_turnover"]:,.0f} 元, '
    ch4 += f'综合费率: {cost_info["cost_ratio"]:.3f}%</p>'

    sections.append({
        'title': '交易成本分析',
        'content': ch4,
        'charts': [c for c in [charts.get('cost_chart')] if c],
    })

    # -- Ch5: 全量指标 --
    if metrics:
        ch5 = '<h3>QuantStats 全量绩效指标</h3>'
        ch5 += metrics_to_html_table(metrics)
        sections.append({
            'title': '全量绩效指标',
            'content': ch5,
            'charts': [],
        })

    # -- Ch6: AI 分析结论 --
    if llm_sections:
        ch6 = '\n'.join(llm_sections)
        sections.append({
            'title': 'AI 智能分析',
            'content': ch6,
            'charts': [],
        })

    report_path = generate_comprehensive_report(
        sections,
        title='实盘交易绩效分析报告 (Plus增强版)',
        output_path=output_path
    )
    return report_path


# ============================================================
# 主流程
# ============================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_files = [
        '历史成交_cy_260101-260325.csv',
    ]

    print(f'\n  待加载文件:')
    for f in csv_files:
        print(f'    - {f}')

    trades_df = load_broker_csv(csv_files)
    if trades_df.empty:
        print('\n  没有加载到有效的成交记录, 程序退出')
        return

    print(f'\n  数据示例 (前5条):')
    for _, row in trades_df.head(5).iterrows():
        date_str = row['成交日期'].strftime('%Y-%m-%d')
        print(f'    {date_str}  {row["买卖方向"]}  {row["证券名称"]:8s}  '
              f'{int(row["成交数量"]):>6}股  {row["成交价格"]:>10.2f}元  '
              f'金额:{row["成交金额"]:>12,.0f}')

    stock_df = analyze_by_stock(trades_df)

    print(f'\n  {"证券名称":8s}  {"买入金额":>12s}  {"卖出金额":>12s}  '
          f'{"未平仓":>6s}  {"已实现盈亏":>12s}  {"总成本":>8s}')
    print(f'  {"-"*8}  {"-"*12}  {"-"*12}  {"-"*6}  {"-"*12}  {"-"*8}')

    total_realized = 0
    for _, row in stock_df.iterrows():
        pnl_str = f'{row["已实现盈亏"]:>+12,.0f}' if pd.notna(row['已实现盈亏']) else '    持仓中   '
        if pd.notna(row['已实现盈亏']):
            total_realized += row['已实现盈亏']
        print(f'  {row["证券名称"]:8s}  {row["买入金额"]:>12,.0f}  {row["卖出金额"]:>12,.0f}  '
              f'{row["未平仓"]:>6}  {pnl_str}  {row["总成本"]:>8,.0f}')

    print(f'\n  已实现盈亏合计: {total_realized:+,.0f} 元')

    pnl_chart = plot_stock_pnl(stock_df, save_path=f'{OUTPUT_DIR}/个股盈亏分析.png')

    cost_info = analyze_costs(trades_df)

    print(f'\n  总成交额:   {cost_info["total_turnover"]:>14,.0f} 元')
    print(f'  佣金:       {cost_info["commission"]:>14,.0f} 元')
    print(f'  印花税:     {cost_info["stamp_tax"]:>14,.0f} 元')
    print(f'  过户费:     {cost_info["transfer_fee"]:>14,.0f} 元')
    print(f'  总成本:     {cost_info["total_cost"]:>14,.0f} 元')
    print(f'  综合费率:   {cost_info["cost_ratio"]:>14.3f} %')

    cost_chart = plot_cost_breakdown(cost_info, save_path=f'{OUTPUT_DIR}/交易成本分析.png')

    first_buy = trades_df[trades_df['买卖方向'] == '买入'].iloc[0]
    estimated_cash = first_buy['成交金额'] * 3
    estimated_cash = max(estimated_cash, 200000)
    estimated_cash = round(estimated_cash / 10000) * 10000

    print(f'\n  估算初始资金: {estimated_cash:,.0f} 元')

    nav_series, daily_df = build_portfolio_nav(trades_df, initial_cash=estimated_cash)

    nav_chart = None
    metrics = {}

    if not nav_series.empty and len(nav_series) > 5:
        print(f'\n  净值序列: {len(nav_series)} 个交易日')
        print(f'  起始净值: {nav_series.iloc[0]:.4f}')
        print(f'  最终净值: {nav_series.iloc[-1]:.4f}')
        print(f'  组合收益: {(nav_series.iloc[-1] / nav_series.iloc[0] - 1) * 100:+.2f}%')

        returns = nav_to_returns(nav_series)

        if len(returns) > 10:
            metrics = calc_quantstats_metrics(returns)
            print_metrics_table(metrics, '实盘交易 - QuantStats 绩效分析')

            nav_chart = plot_returns_chart(returns, title='实盘交易Plus_绩效分析',
                                           save_dir=OUTPUT_DIR)

            nav_csv = f'{OUTPUT_DIR}/实盘交易_每日净值.csv'
            nav_export = pd.DataFrame({
                'date': nav_series.index.strftime('%Y-%m-%d'),
                'nav': nav_series.values,
                'cash': daily_df['cash'].values if 'cash' in daily_df.columns else 0,
                'market_value': daily_df['market_value'].values if 'market_value' in daily_df.columns else 0,
            })
            nav_export.to_csv(nav_csv, index=False, encoding='utf-8-sig')
            print(f'  每日净值已导出: {nav_csv}')

            qs_en_report = f'{OUTPUT_DIR}/实盘交易_QuantStats报告_英文.html'
            generate_html_report(returns,
                                  title='Real Trading Plus - QuantStats Report',
                                  output_path=qs_en_report)

            qs_zh_report = f'{OUTPUT_DIR}/实盘交易_QuantStats报告.html'
            generate_chinese_report(returns,
                                     title='实盘交易Plus - 绩效分析报告',
                                     output_path=qs_zh_report)

    stock_codes = trades_df['标准代码'].unique().tolist()
    trade_start = trades_df['成交日期'].min()
    trade_end = trades_df['成交日期'].max()
    end_date = trade_end.strftime('%Y-%m-%d')

    # SVD需要足够长的时间窗口, 将分析起始日期往前延伸6个月
    # 即使交易只有近几天, SVD分析也需要回溯更长时间才能发现市场规律
    svd_start = (trade_start - pd.DateOffset(months=6)).strftime('%Y-%m-%d')

    # 补充多行业代表性ETF和龙头股, 覆盖主要板块
    # 这样可以更全面地捕捉市场因子结构
    supplement_codes = [
        # 宽基指数ETF
        '510300.SH',   # 沪深300ETF
        '510050.SH',   # 上证50ETF
        '510500.SH',   # 中证500ETF
        # 行业龙头
        '600519.SH',   # 贵州茅台 (消费)
        '000001.SZ',   # 平安银行 (金融)
        '601318.SH',   # 中国平安 (保险)
        '600036.SH',   # 招商银行 (银行)
        '002475.SZ',   # 立讯精密 (电子)
        '300750.SZ',   # 宁德时代 (新能源)
        '600276.SH',   # 恒瑞医药 (医药)
        '601012.SH',   # 隆基绿能 (光伏)
        '002594.SZ',   # 比亚迪 (汽车)
        '601888.SH',   # 中国中免 (旅游)
        '000858.SZ',   # 五粮液 (白酒)
        '002415.SZ',   # 海康威视 (科技)
    ]
    for sc in supplement_codes:
        if sc not in stock_codes:
            stock_codes.append(sc)

    svd_result = diagnose_market_regime(
        stock_codes, svd_start, end_date,
        window=60, step=10
    )

    llm_sections = []
    try:
        from llm_engine import (generate_trade_summary, generate_svd_summary,
                                summary_to_html)

        # 交易分析
        nav_info = None
        if metrics:
            nav_info = {
                'total_return': f'{metrics.get("total_return", 0):.2%}',
                'sharpe': f'{metrics.get("sharpe", 0):.4f}',
                'max_drawdown': f'{metrics.get("max_drawdown", 0):.2%}',
            }
        trade_conclusion = generate_trade_summary(stock_df, cost_info, nav_info)
        if trade_conclusion:
            llm_sections.append(summary_to_html(trade_conclusion, title='AI 交易分析'))

        # SVD分析
        if svd_result:
            svd_conclusion = generate_svd_summary(svd_result)
            if svd_conclusion:
                llm_sections.append(summary_to_html(svd_conclusion, title='AI 市场状态分析'))

    except ImportError:
        print('  [提示] llm_engine 未找到, 跳过LLM分析')
    except Exception as e:
        print(f'  [提示] LLM分析: {e}')

    charts = {
        'nav_chart': nav_chart,
        'pnl_chart': pnl_chart,
        'cost_chart': cost_chart,
    }

    report_path = generate_plus_report(
        trades_df, stock_df, nav_series, metrics,
        cost_info, charts, svd_result,
        output_path=f'{OUTPUT_DIR}/实盘交易Plus分析报告.html',
        llm_sections=llm_sections
    )

    output_files = [
        ('Plus综合报告', f'{OUTPUT_DIR}/实盘交易Plus分析报告.html'),
        ('QuantStats报告', f'{OUTPUT_DIR}/实盘交易_QuantStats报告.html'),
        ('每日净值CSV', f'{OUTPUT_DIR}/实盘交易_每日净值.csv'),
        ('SVD诊断图', f'{OUTPUT_DIR}/SVD市场状态诊断.png'),
        ('个股盈亏图', pnl_chart),
        ('交易成本图', cost_chart),
    ]
    for desc, path in output_files:
        if path and os.path.exists(path):
            print(f'  [OK] {desc}: {path}')
        else:
            print(f'  [--] {desc}: 未生成')


if __name__ == '__main__':
    main()
