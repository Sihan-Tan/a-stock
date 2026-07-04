# -*- coding: utf-8 -*-
"""
数据加载模块 (RL版)

功能:
  - 从MySQL读取K线数据 (trade_stock_daily)
  - 提供分钟级模拟数据生成（用于拆单和高频场景）
"""
import pandas as pd
import numpy as np
from db_config import execute_query


def load_stock_data(stock_code, start_date=None, end_date=None):
    """
    从MySQL加载日K线数据

    参数:
        stock_code: 股票代码，如 '510050.SH'
        start_date: 开始日期，如 '2022-01-01'
        end_date:   结束日期，如 '2025-12-31'

    返回:
        pandas DataFrame，索引为日期，列为 open/high/low/close/volume
    """
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
        raise ValueError(f"没有找到 {stock_code} 的数据，请检查数据库")

    df = pd.DataFrame(rows)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def generate_intraday_data(daily_row, num_steps=48, seed=None):
    """
    基于日线数据模拟一天的分钟级价格序列（用于拆单环境）

    将一个交易日拆成 num_steps 个时段，用几何布朗运动在
    open -> close 之间插值，同时满足 high/low 约束

    参数:
        daily_row: 包含 open/high/low/close/volume 的 Series
        num_steps: 每天切分的时段数（默认48，对应5分钟）
        seed: 随机种子

    返回:
        dict: prices(价格序列), volumes(成交量序列), vwap(当日VWAP)
    """
    if seed is not None:
        np.random.seed(seed)

    o, h, l, c = daily_row['open'], daily_row['high'], daily_row['low'], daily_row['close']
    total_vol = daily_row['volume']

    drift = (c / o) ** (1.0 / num_steps) - 1
    vol = (h - l) / o / np.sqrt(num_steps) * 0.5

    prices = [o]
    for i in range(num_steps):
        shock = np.random.normal(0, 1)
        new_price = prices[-1] * (1 + drift + vol * shock)
        new_price = np.clip(new_price, l * 0.999, h * 1.001)
        prices.append(new_price)

    prices[-1] = c
    prices = np.array(prices)

    # 成交量分布：U型（开盘收盘量大，盘中量小）
    x = np.linspace(0, 1, num_steps)
    vol_weights = 1.5 * (x - 0.5) ** 2 + 0.3
    vol_weights = vol_weights / vol_weights.sum()
    volumes = total_vol * vol_weights

    vwap = np.sum(prices[1:] * volumes) / np.sum(volumes)

    return {
        'prices': prices,
        'volumes': volumes,
        'vwap': vwap,
    }


def load_multi_stock_data(stock_codes, start_date=None, end_date=None):
    """
    批量加载多只股票数据

    返回:
        dict: {stock_code: DataFrame}
    """
    result = {}
    for code in stock_codes:
        try:
            result[code] = load_stock_data(code, start_date, end_date)
        except ValueError:
            print(f"  跳过 {code}: 无数据")
    return result
