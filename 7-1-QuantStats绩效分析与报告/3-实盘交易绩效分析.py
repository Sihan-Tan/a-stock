# -*- coding: utf-8 -*-
"""
实盘交易绩效分析 -- 从券商导出 CSV 生成 QuantStats 绩效报告

使用场景:
  你在券商PC客户端(如金阳光)导出了自己的历史成交记录 CSV,
  想看看自己的实盘交易绩效到底怎么样。

  本脚本支持:
    1. 解析券商导出的历史成交 CSV (光大证券金阳光格式)
    2. 合并多个 CSV 文件 (券商限制每次最多查90天)
    3. 按个股拆解盈亏明细 (哪只股票赚了/亏了)
    4. 构建组合每日净值曲线 (用实际市场价格逐日估值)
    5. 生成 QuantStats 全量绩效指标 + 中文 HTML 报告
    6. 成本分析: 佣金/印花税/过户费 总计花了多少
    7. miniQMT 实盘交易接口参考

导出方法:
  PC版金阳光 -> 顶部导航 "账户" -> 普通交易 -> 历史成交
  -> 选择起始/终止日期 (间隔不超过90天)
  -> 导出 CSV

运行: python 3-实盘交易绩效分析.py
"""
import os
import sys
import time
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

OUTPUT_DIR = 'outputs/3-实盘交易分析'


# ============================================================
# 券商 CSV 解析
# ============================================================

def code_to_standard(raw_code):
    """
    将券商导出的证券代码转为标准格式 (如 002432 -> 002432.SZ)

    规则:
      - 6xxxxx -> xxxxxx.SH (上交所)
      - 0xxxxx / 3xxxxx -> xxxxxx.SZ (深交所)
      - 5xxxxx -> xxxxxx.SH (上交所ETF)
      - 其他(如港股 00467) -> 返回 None (暂不支持)
    """
    code = raw_code.strip()
    if len(code) == 6:
        if code.startswith(('6', '5')):
            return code + '.SH'
        elif code.startswith(('0', '3')):
            return code + '.SZ'
    return None


def load_broker_csv(csv_paths):
    """
    加载并合并券商导出的历史成交 CSV 文件

    参数:
        csv_paths: str 或 list, CSV文件路径(支持多个文件合并)

    返回:
        DataFrame, 清洗后的成交记录
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    all_dfs = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f'  [跳过] 文件不存在: {path}')
            continue

        # utf-8-sig 优先：券商/Excel 导出的 UTF-8 常带 BOM；再试国标码
        encodings = ['utf-8-sig', 'utf-8', 'gb18030', 'gbk']
        df = None
        for enc in encodings:
            try:
                df = pd.read_csv(path, encoding=enc, dtype={'证券代码': str})
                break
            except (UnicodeDecodeError, Exception):
                continue

        if df is None:
            print(f'  [跳过] 无法读取: {path}')
            continue

        print(f'  读取: {path} ({len(df)} 条记录)')
        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    merged = pd.concat(all_dfs, ignore_index=True)

    # 清洗: 去除列名和值中的前后空格
    merged.columns = [c.strip() for c in merged.columns]
    for col in merged.columns:
        if merged[col].dtype == object:
            merged[col] = merged[col].astype(str).str.strip()

    merged['证券代码'] = merged['证券代码'].astype(str).str.strip()
    merged['成交日期'] = pd.to_datetime(merged['成交日期'].astype(str).str.strip(), errors='coerce')
    merged['买卖方向'] = merged['买卖方向'].astype(str).str.strip()

    for col in ['成交数量', '成交价格', '成交金额', '佣金', '印花税', '过户费', '结算费']:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors='coerce').fillna(0)

    # 转为标准代码, 过滤不支持的证券(如港股)
    merged['标准代码'] = merged['证券代码'].apply(code_to_standard)
    hk_count = merged['标准代码'].isna().sum()
    if hk_count > 0:
        hk_names = merged[merged['标准代码'].isna()]['证券名称'].unique()
        print(f'  [过滤] {hk_count} 条非A股记录 (港股等): {", ".join(hk_names)}')
        merged = merged[merged['标准代码'].notna()].copy()

    # 去重(按成交编号)
    if '成交编号' in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=['成交编号'])
        dup_count = before - len(merged)
        if dup_count > 0:
            print(f'  [去重] 移除 {dup_count} 条重复记录')

    merged = merged.sort_values('成交日期').reset_index(drop=True)

    print(f'  合并后: {len(merged)} 条A股成交记录')
    print(f'  时间范围: {merged["成交日期"].min().strftime("%Y-%m-%d")} '
          f'~ {merged["成交日期"].max().strftime("%Y-%m-%d")}')

    return merged


# ============================================================
# 个股盈亏分析
# ============================================================

def analyze_by_stock(trades_df):
    """按个股拆解交易明细和盈亏"""
    results = []

    for code in trades_df['标准代码'].unique():
        stock_trades = trades_df[trades_df['标准代码'] == code].copy()
        name = stock_trades['证券名称'].iloc[0]

        buys = stock_trades[stock_trades['买卖方向'] == '买入']
        sells = stock_trades[stock_trades['买卖方向'] == '卖出']

        buy_amount = buys['成交金额'].sum()
        sell_amount = sells['成交金额'].sum()
        buy_volume = buys['成交数量'].sum()
        sell_volume = sells['成交数量'].sum()

        commission = stock_trades['佣金'].sum()
        stamp_tax = stock_trades['印花税'].sum() if '印花税' in stock_trades.columns else 0
        transfer_fee = stock_trades['过户费'].sum() if '过户费' in stock_trades.columns else 0
        total_cost = commission + stamp_tax + transfer_fee

        remaining = buy_volume - sell_volume
        realized_pnl = sell_amount - buy_amount - total_cost if remaining <= 0 else None

        avg_buy = buy_amount / buy_volume if buy_volume > 0 else 0
        avg_sell = sell_amount / sell_volume if sell_volume > 0 else 0

        first_date = stock_trades['成交日期'].min()
        last_date = stock_trades['成交日期'].max()
        trade_days = (last_date - first_date).days

        results.append({
            '证券代码': code, '证券名称': name,
            '买入次数': len(buys), '卖出次数': len(sells),
            '买入金额': buy_amount, '卖出金额': sell_amount,
            '买入量': int(buy_volume), '卖出量': int(sell_volume),
            '未平仓': int(remaining),
            '均买价': avg_buy, '均卖价': avg_sell,
            '佣金': commission, '印花税': stamp_tax,
            '过户费': transfer_fee, '总成本': total_cost,
            '已实现盈亏': realized_pnl,
            '首次交易': first_date, '末次交易': last_date,
            '交易天数': trade_days,
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values('买入金额', ascending=False)

    return result_df


# ============================================================
# 构建组合每日净值
# ============================================================

def build_portfolio_nav(trades_df, initial_cash=1000000):
    """
    根据实际成交记录和市场行情, 构建每日组合净值

    流程:
      1. 按日期遍历所有成交记录, 更新现金和持仓
      2. 每个交易日, 用数据库中的收盘价对持仓做 mark-to-market
      3. NAV = 现金 + 所有持仓的市值
    """
    if trades_df.empty:
        return pd.Series(dtype=float), pd.DataFrame()

    start_date = trades_df['成交日期'].min().strftime('%Y-%m-%d')
    end_date = trades_df['成交日期'].max().strftime('%Y-%m-%d')

    stock_codes = trades_df['标准代码'].unique().tolist()
    close_prices = {}

    print(f'\n  加载行情数据 ({len(stock_codes)} 只股票)...')
    for code in stock_codes:
        try:
            df = load_stock_data(code, start_date, end_date)
            close_prices[code] = df['close']
            print(f'    {code}: {len(df)} 个交易日')
        except Exception as e:
            print(f'    {code}: 行情加载失败 ({e}), 使用成交价估值')

    all_dates = set()
    for code, prices in close_prices.items():
        all_dates.update(prices.index.tolist())
    for d in trades_df['成交日期'].unique():
        all_dates.add(pd.Timestamp(d))
    all_dates = sorted(all_dates)
    if not all_dates:
        return pd.Series(dtype=float), pd.DataFrame()

    cash = initial_cash
    holdings = {}
    daily_records = []
    trades_by_date = trades_df.groupby(trades_df['成交日期'].dt.date)

    for date in all_dates:
        date_key = date.date() if hasattr(date, 'date') else date

        if date_key in trades_by_date.groups:
            day_trades = trades_by_date.get_group(date_key)
            for _, trade in day_trades.iterrows():
                code = trade['标准代码']
                price = float(trade['成交价格'])
                volume = int(trade['成交数量'])
                amount = float(trade['成交金额'])
                direction = trade['买卖方向']
                cost = (float(trade.get('佣金', 0)) +
                        float(trade.get('印花税', 0)) +
                        float(trade.get('过户费', 0)))

                if code not in holdings:
                    holdings[code] = {
                        'volume': 0, 'last_price': price,
                        'name': trade['证券名称'],
                    }

                if direction == '买入':
                    cash -= (amount + cost)
                    holdings[code]['volume'] += volume
                    holdings[code]['last_price'] = price
                elif direction == '卖出':
                    cash += (amount - cost)
                    holdings[code]['volume'] = max(0, holdings[code]['volume'] - volume)
                    holdings[code]['last_price'] = price

        total_market_value = 0
        for code, pos in list(holdings.items()):
            if pos['volume'] <= 0:
                continue
            if code in close_prices:
                ts = close_prices[code]
                valid = ts[ts.index <= date]
                if len(valid) > 0:
                    pos['last_price'] = float(valid.iloc[-1])
            mv = pos['volume'] * pos['last_price']
            total_market_value += mv

        nav = cash + total_market_value
        daily_records.append({
            'date': date, 'cash': cash,
            'market_value': total_market_value, 'nav': nav,
            'holding_count': sum(1 for p in holdings.values() if p['volume'] > 0),
        })

    daily_df = pd.DataFrame(daily_records)
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df.set_index('date', inplace=True)

    nav_series = daily_df['nav'] / initial_cash
    nav_series.name = 'nav'
    return nav_series, daily_df


# ============================================================
# 成本分析
# ============================================================

def analyze_costs(trades_df):
    """分析交易成本构成"""
    commission = trades_df['佣金'].sum()
    stamp_tax = trades_df['印花税'].sum() if '印花税' in trades_df.columns else 0
    transfer_fee = trades_df['过户费'].sum() if '过户费' in trades_df.columns else 0
    settlement_fee = trades_df['结算费'].sum() if '结算费' in trades_df.columns else 0
    total_cost = commission + stamp_tax + transfer_fee + settlement_fee
    total_turnover = trades_df['成交金额'].sum()
    cost_ratio = total_cost / total_turnover * 100 if total_turnover > 0 else 0

    return {
        'commission': commission, 'stamp_tax': stamp_tax,
        'transfer_fee': transfer_fee, 'settlement_fee': settlement_fee,
        'total_cost': total_cost, 'total_turnover': total_turnover,
        'cost_ratio': cost_ratio,
    }


# ============================================================
# 可视化
# ============================================================

def plot_stock_pnl(stock_df, save_path=None):
    """绘制个股盈亏柱状图"""
    if save_path is None:
        save_path = f'{OUTPUT_DIR}/个股盈亏分析.png'
    closed = stock_df[stock_df['已实现盈亏'].notna()].copy()
    if closed.empty:
        print('  [跳过] 没有已平仓的交易')
        return save_path

    closed = closed.sort_values('已实现盈亏')
    names = closed['证券名称'].tolist()
    pnl = closed['已实现盈亏'].tolist()
    colors = ['#27ae60' if v >= 0 else '#e74c3c' for v in pnl]

    fig, ax = plt.subplots(figsize=(12, max(4, len(names) * 0.5)))
    bars = ax.barh(range(len(names)), pnl, color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel('已实现盈亏 (元)', fontsize=12)
    ax.set_title('个股盈亏分析', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
    ax.grid(True, axis='x', alpha=0.3)

    for bar, val in zip(bars, pnl):
        x_pos = bar.get_width()
        offset = max(abs(max(pnl, default=0) - min(pnl, default=0)) * 0.02, 100)
        ha = 'left' if val >= 0 else 'right'
        x_text = x_pos + offset if val >= 0 else x_pos - offset
        ax.text(x_text, bar.get_y() + bar.get_height() / 2,
                f'{val:+,.0f}', va='center', ha=ha, fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  图表已保存: {save_path}')
    return save_path


def plot_cost_breakdown(cost_info, save_path=None):
    """绘制交易成本饼图"""
    if save_path is None:
        save_path = f'{OUTPUT_DIR}/交易成本分析.png'
    labels = []
    values = []
    colors_pie = ['#3498db', '#e74c3c', '#f39c12', '#95a5a6']

    for key, label in [('commission', '佣金'), ('stamp_tax', '印花税'),
                        ('transfer_fee', '过户费'), ('settlement_fee', '结算费')]:
        val = cost_info.get(key, 0)
        if val > 0:
            labels.append(f'{label}\n{val:,.0f}元')
            values.append(val)

    if not values:
        return save_path

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.pie(values, labels=labels, autopct='%1.1f%%',
            colors=colors_pie[:len(values)], startangle=90,
            textprops={'fontsize': 11})
    ax1.set_title(f'交易成本构成 (合计: {cost_info["total_cost"]:,.0f}元)',
                   fontsize=13, fontweight='bold')

    info_text = (
        f'总成交额:  {cost_info["total_turnover"]:>14,.0f} 元\n'
        f'总成本:    {cost_info["total_cost"]:>14,.0f} 元\n'
        f'费率:      {cost_info["cost_ratio"]:>14.3f} %\n'
        f'---\n'
        f'佣金:      {cost_info["commission"]:>14,.0f} 元\n'
        f'印花税:    {cost_info["stamp_tax"]:>14,.0f} 元\n'
        f'过户费:    {cost_info["transfer_fee"]:>14,.0f} 元'
    )
    ax2.text(0.1, 0.5, info_text, transform=ax2.transAxes,
             fontsize=13, verticalalignment='center',
             fontfamily='Consolas',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dce4ec'))
    ax2.axis('off')
    ax2.set_title('成本明细', fontsize=13, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  图表已保存: {save_path}')
    return save_path


# ============================================================
# 生成综合HTML报告
# ============================================================

def generate_trade_analysis_report(trades_df, stock_df, nav_series, metrics,
                                    cost_info, charts, output_path):
    """组装综合HTML报告"""
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

    # -- Ch2: 个股盈亏分析 --
    ch2 = '<h3>个股交易明细</h3>'
    ch2 += '<table><tr><th>证券名称</th><th>买入金额</th><th>卖出金额</th>'
    ch2 += '<th>未平仓</th><th>已实现盈亏</th><th>总成本</th></tr>'
    for _, row in stock_df.iterrows():
        pnl_str = f'{row["已实现盈亏"]:+,.0f}' if pd.notna(row['已实现盈亏']) else '持仓中'
        pnl_color = '#27ae60' if pd.notna(row['已实现盈亏']) and row['已实现盈亏'] >= 0 else '#e74c3c'
        ch2 += f'<tr><td>{row["证券名称"]}</td>'
        ch2 += f'<td>{row["买入金额"]:,.0f}</td>'
        ch2 += f'<td>{row["卖出金额"]:,.0f}</td>'
        ch2 += f'<td>{row["未平仓"]}</td>'
        ch2 += f'<td style="color:{pnl_color};font-weight:bold">{pnl_str}</td>'
        ch2 += f'<td>{row["总成本"]:,.0f}</td></tr>'
    ch2 += '</table>'

    sections.append({
        'title': '个股盈亏分析',
        'content': ch2,
        'charts': [c for c in [charts.get('pnl_chart')] if c],
    })

    # -- Ch3: 成本分析 --
    ch3 = '<h3>交易成本构成</h3>'
    ch3 += '<table><tr><th>项目</th><th>金额</th><th>占比</th></tr>'
    for key, label in [('commission', '佣金'), ('stamp_tax', '印花税'),
                        ('transfer_fee', '过户费')]:
        val = cost_info.get(key, 0)
        pct = val / cost_info['total_cost'] * 100 if cost_info['total_cost'] > 0 else 0
        ch3 += f'<tr><td>{label}</td><td>{val:,.0f} 元</td><td>{pct:.1f}%</td></tr>'
    ch3 += f'<tr style="font-weight:bold"><td>合计</td>'
    ch3 += f'<td>{cost_info["total_cost"]:,.0f} 元</td><td>100%</td></tr>'
    ch3 += '</table>'
    ch3 += f'<p>总成交额: {cost_info["total_turnover"]:,.0f} 元, '
    ch3 += f'综合费率: {cost_info["cost_ratio"]:.3f}%</p>'

    sections.append({
        'title': '交易成本分析',
        'content': ch3,
        'charts': [c for c in [charts.get('cost_chart')] if c],
    })

    # -- Ch4: 全量指标 --
    if metrics:
        ch4 = '<h3>QuantStats 全量绩效指标</h3>'
        ch4 += metrics_to_html_table(metrics)
        sections.append({
            'title': '全量绩效指标',
            'content': ch4,
            'charts': [],
        })

    report_path = generate_comprehensive_report(
        sections,
        title='实盘交易绩效分析报告',
        output_path=output_path
    )
    return report_path


# ============================================================
# miniQMT 实盘交易接口 (参考)
# ============================================================

class MiniQMTTrader:
    """
    miniQMT 实盘交易封装

    使用前需要:
      1. 安装 xtquant: pip install xtquant
      2. 打开 QMT 客户端(极简模式)
      3. 配置 QMT_PATH 和 ACCOUNT_ID
    """

    def __init__(self, qmt_path, account_id, session_id=None):
        self.qmt_path = qmt_path
        self.account_id = account_id
        self.session_id = session_id or int(time.time())
        self.trader = None
        self.account = None
        self.connected = False

    def connect(self):
        """连接 miniQMT"""
        try:
            from xtquant import xttrader
            from xtquant.xttrader import XtQuantTrader
            from xtquant.xttype import StockAccount

            class _Callback(xttrader.XtQuantTraderCallback):
                def on_disconnected(self):
                    print('[miniQMT] 连接断开')

                def on_stock_order(self, order):
                    print(f'[miniQMT] 委托回报: {order.stock_code} '
                          f'编号:{order.order_id} 状态:{order.order_status}')

                def on_stock_trade(self, trade):
                    print(f'[miniQMT] 成交回报: {trade.stock_code} '
                          f'数量:{trade.traded_volume} 价格:{trade.traded_price}')

                def on_order_error(self, order_error):
                    print(f'[miniQMT] 委托失败: {order_error.error_msg}')

            self.trader = XtQuantTrader(self.qmt_path, self.session_id)
            self.trader.register_callback(_Callback())
            self.trader.start()

            for i in range(3):
                result = self.trader.connect()
                if result == 0:
                    break
                print(f'[miniQMT] 连接重试 {i+1}/3...')
                time.sleep(1)

            if result != 0:
                raise Exception(f'连接失败, 错误码: {result}. 请确保QMT客户端已启动')

            self.account = StockAccount(self.account_id)
            sub_result = self.trader.subscribe(self.account)
            if sub_result != 0:
                raise Exception(f'订阅失败, 错误码: {sub_result}')

            self.connected = True
            print(f'[miniQMT] 连接成功, 账户: {self.account_id}')
            return True

        except ImportError:
            print('[miniQMT] xtquant 未安装. 请运行: pip install xtquant')
            return False
        except Exception as e:
            print(f'[miniQMT] 连接异常: {e}')
            return False

    def buy(self, stock_code, volume, price=0):
        """买入股票"""
        if not self.connected:
            print('[miniQMT] 未连接, 请先调用 connect()')
            return None

        from xtquant import xtconstant
        order_type = xtconstant.LATEST_PRICE if price == 0 else xtconstant.FIX_PRICE
        order_price = price if price > 0 else 0

        order_id = self.trader.order_stock(
            self.account, stock_code,
            xtconstant.STOCK_BUY, volume,
            order_type, order_price
        )
        print(f'[miniQMT] 买入委托: {stock_code} {volume}股 '
              f'价格:{"市价" if price == 0 else price} 编号:{order_id}')
        return order_id

    def sell(self, stock_code, volume, price=0):
        """卖出股票"""
        if not self.connected:
            return None

        from xtquant import xtconstant
        order_type = xtconstant.LATEST_PRICE if price == 0 else xtconstant.FIX_PRICE
        order_price = price if price > 0 else 0

        order_id = self.trader.order_stock(
            self.account, stock_code,
            xtconstant.STOCK_SELL, volume,
            order_type, order_price
        )
        print(f'[miniQMT] 卖出委托: {stock_code} {volume}股 编号:{order_id}')
        return order_id

    def query_positions(self):
        """查询持仓"""
        if not self.connected:
            return []
        positions = self.trader.query_stock_positions(self.account)
        return [{'stock_code': p.stock_code, 'volume': p.volume,
                 'can_use': p.can_use_volume, 'cost_price': p.open_price,
                 'market_value': p.market_value} for p in positions]

    def query_asset(self):
        """查询账户资产"""
        if not self.connected:
            return {}
        asset = self.trader.query_stock_asset(self.account)
        return {'total_asset': asset.total_asset, 'cash': asset.cash,
                'market_value': asset.market_value}

    def query_today_orders(self):
        """查询当日委托"""
        if not self.connected:
            return []
        orders = self.trader.query_stock_orders(self.account)
        if not orders:
            return []
        return [{'stock_code': o.stock_code, 'order_id': o.order_id,
                 'order_volume': o.order_volume, 'price': o.price,
                 'traded_volume': o.traded_volume, 'order_status': o.order_status,
                 'order_time': o.order_time} for o in orders]

    def query_today_trades(self):
        """查询当日成交"""
        if not self.connected:
            return []
        trades = self.trader.query_stock_trades(self.account)
        if not trades:
            return []
        return [{'stock_code': t.stock_code, 'traded_id': t.traded_id,
                 'traded_time': t.traded_time, 'traded_price': t.traded_price,
                 'traded_volume': t.traded_volume, 'traded_amount': t.traded_amount,
                 'order_id': t.order_id} for t in trades]

    def disconnect(self):
        """断开连接"""
        if self.trader:
            self.trader.stop()
            self.connected = False
            print('[miniQMT] 已断开连接')


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

    pnl_chart = plot_stock_pnl(stock_df)

    cost_info = analyze_costs(trades_df)

    print(f'\n  总成交额:   {cost_info["total_turnover"]:>14,.0f} 元')
    print(f'  佣金:       {cost_info["commission"]:>14,.0f} 元')
    print(f'  印花税:     {cost_info["stamp_tax"]:>14,.0f} 元')
    print(f'  过户费:     {cost_info["transfer_fee"]:>14,.0f} 元')
    print(f'  总成本:     {cost_info["total_cost"]:>14,.0f} 元')
    print(f'  综合费率:   {cost_info["cost_ratio"]:>14.3f} %')

    cost_chart = plot_cost_breakdown(cost_info)

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

            nav_chart = plot_returns_chart(returns, title='实盘交易_绩效分析',
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
                                  title='Real Trading - QuantStats Report',
                                  output_path=qs_en_report)

            qs_zh_report = f'{OUTPUT_DIR}/实盘交易_QuantStats报告.html'
            generate_chinese_report(returns,
                                     title='实盘交易 - 绩效分析报告',
                                     output_path=qs_zh_report)
        else:
            print(f'  [提示] 净值数据点不足 ({len(returns)}天), 跳过QuantStats指标')
    else:
        print(f'  [提示] 无法构建有效净值曲线 (行情数据不足), 跳过绩效分析')

    charts = {
        'nav_chart': nav_chart,
        'pnl_chart': pnl_chart,
        'cost_chart': cost_chart,
    }

    report_path = generate_trade_analysis_report(
        trades_df, stock_df, nav_series, metrics,
        cost_info, charts,
        output_path=f'{OUTPUT_DIR}/实盘交易分析报告.html'
    )

    export_cols = ['证券名称', '标准代码', '成交日期', '买卖方向',
                   '成交数量', '成交价格', '成交金额', '佣金', '印花税', '过户费']
    export_cols = [c for c in export_cols if c in trades_df.columns]
    export_df = trades_df[export_cols].copy()
    export_df['成交日期'] = export_df['成交日期'].dt.strftime('%Y-%m-%d')
    export_csv = f'{OUTPUT_DIR}/清洗后_交易记录.csv'
    export_df.to_csv(export_csv, index=False, encoding='utf-8-sig')
    print(f'  已导出: {export_csv}')


if __name__ == '__main__':
    main()
