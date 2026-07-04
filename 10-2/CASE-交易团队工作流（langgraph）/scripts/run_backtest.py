# -*- coding: utf-8 -*-
# vendored: copy from 20-团队架构设计/CASE-AI量化助手（nanobot）/skills/strategy-backtest/scripts/run_backtest.py
# 团队工作流（langgraph）的 Zoe 节点通过 subprocess 调用此脚本，让本项目自包含
"""
策略回测脚本

支持 MACD 和双均线策略，基于 xtdata 获取历史数据，模拟交易并统计绩效。
输出 JSON 格式结果，供 Agent 解读后展示给用户。

用法:
    python run_backtest.py --code 513100.SH --strategy macd --count 250
    python run_backtest.py --code 513100.SH --strategy double_ma --start 20240101
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
from xtquant import xtdata


def fetch_kline(stock_code, period='1d', start_date='', count=250):
    """通过 xtdata 获取历史K线数据"""
    xtdata.connect()
    xtdata.download_history_data(
        stock_code, period=period,
        start_time=start_date if start_date else '20200101',
        end_time='', incrementally=True
    )
    data = xtdata.get_market_data_ex(
        field_list=['open', 'high', 'low', 'close', 'volume'],
        stock_list=[stock_code],
        period=period,
        start_time=start_date,
        end_time='',
        count=count
    )
    if not data or stock_code not in data:
        return None
    df = data[stock_code]
    if df is None or len(df) == 0:
        return None
    df.index = pd.to_datetime(df.index)
    df['date'] = df.index.strftime('%Y-%m-%d')
    return df


def calc_macd(df, fast=12, slow=26, signal=9):
    """计算 MACD 指标"""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['dif'] = ema_fast - ema_slow
    df['dea'] = df['dif'].ewm(span=signal, adjust=False).mean()
    df['macd_hist'] = 2 * (df['dif'] - df['dea'])
    return df


def calc_double_ma(df, short_window=5, long_window=20):
    """计算双均线"""
    df['ma_short'] = df['close'].rolling(window=short_window).mean()
    df['ma_long'] = df['close'].rolling(window=long_window).mean()
    return df


def generate_macd_signals(df):
    """基于 MACD 生成交易信号"""
    df = calc_macd(df)
    signals = []
    position = 0

    for i in range(1, len(df)):
        prev_dif = df['dif'].iloc[i - 1]
        prev_dea = df['dea'].iloc[i - 1]
        curr_dif = df['dif'].iloc[i]
        curr_dea = df['dea'].iloc[i]

        if prev_dif <= prev_dea and curr_dif > curr_dea and position == 0:
            signals.append({
                'date': df['date'].iloc[i],
                'action': 'buy',
                'price': float(df['close'].iloc[i]),
                'reason': 'MACD金叉'
            })
            position = 1

        elif prev_dif >= prev_dea and curr_dif < curr_dea and position == 1:
            signals.append({
                'date': df['date'].iloc[i],
                'action': 'sell',
                'price': float(df['close'].iloc[i]),
                'reason': 'MACD死叉'
            })
            position = 0

    return signals, df


def generate_double_ma_signals(df, short_window=5, long_window=20):
    """基于双均线生成交易信号"""
    df = calc_double_ma(df, short_window, long_window)
    df = df.dropna()
    signals = []
    position = 0

    for i in range(1, len(df)):
        prev_short = df['ma_short'].iloc[i - 1]
        prev_long = df['ma_long'].iloc[i - 1]
        curr_short = df['ma_short'].iloc[i]
        curr_long = df['ma_long'].iloc[i]

        if prev_short <= prev_long and curr_short > curr_long and position == 0:
            signals.append({
                'date': df['date'].iloc[i],
                'action': 'buy',
                'price': float(df['close'].iloc[i]),
                'reason': f'MA{short_window}上穿MA{long_window}'
            })
            position = 1

        elif prev_short >= prev_long and curr_short < curr_long and position == 1:
            signals.append({
                'date': df['date'].iloc[i],
                'action': 'sell',
                'price': float(df['close'].iloc[i]),
                'reason': f'MA{short_window}下穿MA{long_window}'
            })
            position = 0

    return signals, df


def compute_stats(signals, df):
    """根据交易信号计算绩效统计"""
    trades = []
    buy_price = None

    for sig in signals:
        if sig['action'] == 'buy':
            buy_price = sig['price']
        elif sig['action'] == 'sell' and buy_price is not None:
            ret = (sig['price'] - buy_price) / buy_price
            trades.append({
                'buy_date': None,
                'sell_date': sig['date'],
                'buy_price': round(buy_price, 3),
                'sell_price': round(sig['price'], 3),
                'return_pct': round(ret * 100, 2),
            })
            buy_price = None

    # 回填买入日期
    buy_idx = 0
    for i, sig in enumerate(signals):
        if sig['action'] == 'buy' and buy_idx < len(trades):
            trades[buy_idx]['buy_date'] = sig['date']
            buy_idx += 1

    if not trades:
        return {
            'trade_count': 0,
            'win_count': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'trades': [],
        }

    returns = [t['return_pct'] for t in trades]
    win_count = sum(1 for r in returns if r > 0)

    cumulative = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in returns:
        cumulative *= (1 + r / 100)
        peak = max(peak, cumulative)
        dd = (peak - cumulative) / peak
        max_dd = max(max_dd, dd)

    total_return = (cumulative - 1) * 100

    return {
        'trade_count': len(trades),
        'win_count': win_count,
        'win_rate': round(win_count / len(trades) * 100, 1),
        'total_return': round(total_return, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'trades': trades[-10:],
    }


def detect_latest_signal(df, strategy):
    """检测最新一根K线的信号状态"""
    if len(df) < 2:
        return 'none'

    if strategy == 'macd':
        curr_dif = df['dif'].iloc[-1]
        curr_dea = df['dea'].iloc[-1]
        prev_dif = df['dif'].iloc[-2]
        prev_dea = df['dea'].iloc[-2]

        if prev_dif <= prev_dea and curr_dif > curr_dea:
            return 'golden_cross'
        elif prev_dif >= prev_dea and curr_dif < curr_dea:
            return 'death_cross'
        elif curr_dif > curr_dea:
            return 'bullish'
        else:
            return 'bearish'

    elif strategy == 'double_ma':
        if 'ma_short' not in df.columns:
            return 'none'
        curr_short = df['ma_short'].iloc[-1]
        curr_long = df['ma_long'].iloc[-1]
        prev_short = df['ma_short'].iloc[-2]
        prev_long = df['ma_long'].iloc[-2]

        if prev_short <= prev_long and curr_short > curr_long:
            return 'golden_cross'
        elif prev_short >= prev_long and curr_short < curr_long:
            return 'death_cross'
        elif curr_short > curr_long:
            return 'bullish'
        else:
            return 'bearish'

    return 'none'


def main():
    parser = argparse.ArgumentParser(description='策略回测')
    parser.add_argument('--code', required=True, help='股票代码，如 513100.SH')
    parser.add_argument('--strategy', default='macd',
                        choices=['macd', 'double_ma'], help='策略类型')
    parser.add_argument('--start', default='', help='开始日期 YYYYMMDD')
    parser.add_argument('--count', type=int, default=250, help='K线条数')
    args = parser.parse_args()

    df = fetch_kline(args.code, period='1d', start_date=args.start,
                     count=args.count)
    if df is None:
        result = {"error": f"未获取到 {args.code} 的历史数据"}
        print(json.dumps(result, ensure_ascii=False))
        return

    if args.strategy == 'macd':
        signals, df = generate_macd_signals(df)
        strategy_name = 'MACD(12,26,9)'
    else:
        signals, df = generate_double_ma_signals(df)
        strategy_name = '双均线(MA5,MA20)'

    stats = compute_stats(signals, df)
    latest_signal = detect_latest_signal(df, args.strategy)

    latest_close = float(df['close'].iloc[-1])
    latest_date = df['date'].iloc[-1]

    result = {
        'stock_code': args.code,
        'strategy': strategy_name,
        'data_range': f"{df['date'].iloc[0]} ~ {latest_date}",
        'data_count': len(df),
        'latest_close': round(latest_close, 3),
        'latest_date': latest_date,
        'latest_signal': latest_signal,
        'trade_count': stats['trade_count'],
        'win_count': stats['win_count'],
        'win_rate': stats['win_rate'],
        'total_return': stats['total_return'],
        'max_drawdown': stats['max_drawdown'],
        'recent_trades': stats['trades'],
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
