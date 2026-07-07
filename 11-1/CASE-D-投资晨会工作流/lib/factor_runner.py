# -*- coding: utf-8 -*-
# 21-CASE-D: 多因子选股运行器 (从 CASE-C 复制核心逻辑)
"""
FactorRunner -- 多因子计算 + 预处理 + 合成, 给 graph.py 用

复制源:
    - CASE-C/factor_lib.py     calc_factors_for_one
    - CASE-C/preprocessor.py   winsorize_mad, zscore, industry_neutralize, preprocess_factors
    - CASE-C/stock_pool.py     filter_tradable

数据来源差异:
    CASE-C 原版 calc_factors_batch 走 xtdata.get_market_data_ex
    本 lib   load_kline_from_db   走 trade_stock_daily (CASE-A 落库的 K 线)
    -- 晨会工作流走 DB 更稳定, 不依赖 QMT 在线
    -- 兜底: 如果 DB 中数据不足 130 日, 会兜底走 xtdata 在线拉

主接口:
    filter_tradable(stock_codes) -> List[str]
    calc_factors_batch(stock_codes) -> DataFrame[index=code, columns=因子名]
    preprocess_factors(df, industry_map, neutralize=True) -> DataFrame
"""
from __future__ import annotations
import math
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db_config import execute_query


# ============================================================
# 个股 K 线加载 (从 trade_stock_daily 读)
# ============================================================

def load_kline_from_db(stock_code: str, lookback_days: int = 200) -> pd.DataFrame:
    """从 trade_stock_daily 加载单股最近 N 个交易日的 K 线"""
    sql = """
        SELECT trade_date, open_price, high_price, low_price, close_price,
               volume, amount
        FROM trade_stock_daily
        WHERE stock_code = %s
        ORDER BY trade_date DESC
        LIMIT %s
    """
    rows = execute_query(sql, (stock_code, lookback_days))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows[::-1])    # 倒序 -> 升序
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df.set_index("trade_date", inplace=True)
    df.rename(columns={
        "open_price": "open", "high_price": "high",
        "low_price":  "low",  "close_price": "close",
    }, inplace=True)
    for col in ["open", "high", "low", "close", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    return df


# ============================================================
# 因子计算 (复制自 CASE-C/factor_lib.py)
# ============================================================

def _safe_pct_change(prices: pd.Series, periods: int) -> float:
    if len(prices) <= periods:
        return np.nan
    p_now  = prices.iloc[-1]
    p_then = prices.iloc[-1 - periods]
    if p_then <= 0:
        return np.nan
    return p_now / p_then - 1.0


def calc_factors_for_one(df: pd.DataFrame, total_share: float = 0) -> Dict[str, float]:
    """
    给定单股日 K, 算 10 个因子 (与 CASE-C 完全一致)
    """
    if df is None or len(df) < 130:
        return {}

    close  = df["close"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else None
    amount = df["amount"].astype(float) if "amount" in df.columns else None

    returns = close.pct_change().dropna()
    if len(returns) < 100:
        return {}

    f = {}
    f["MOM_1M"] = _safe_pct_change(close, 21)
    f["MOM_3M"] = _safe_pct_change(close, 63)
    f["MOM_6M"] = _safe_pct_change(close, 126)

    rev_5d = _safe_pct_change(close, 5)
    f["REV_5D"] = -rev_5d if not np.isnan(rev_5d) else np.nan

    vol_20 = returns.tail(20).std() * math.sqrt(250)
    vol_60 = returns.tail(60).std() * math.sqrt(250)
    f["VOL_20"] = -vol_20 if not np.isnan(vol_20) else np.nan
    f["VOL_60"] = -vol_60 if not np.isnan(vol_60) else np.nan

    if amount is not None and len(amount) >= 20:
        liq_20 = amount.tail(20).mean()
        f["LIQ_20"] = -math.log(max(liq_20, 1.0))
    else:
        f["LIQ_20"] = np.nan

    if volume is not None and len(volume) >= 20:
        if total_share > 0:
            turn_20 = (volume.tail(20).mean() / total_share) * 100
        else:
            long_vol = volume.tail(60).mean()
            turn_20 = volume.tail(20).mean() / long_vol if long_vol > 0 else np.nan
        f["TURN_20"] = -turn_20 if not np.isnan(turn_20) else np.nan
    else:
        f["TURN_20"] = np.nan

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi_val = rsi.iloc[-1] if len(rsi) > 0 else np.nan
    f["RSI_14"] = (rsi_val - 50) if not np.isnan(rsi_val) else np.nan

    ma20 = close.rolling(20).mean().iloc[-1]
    bias_20 = (close.iloc[-1] - ma20) / ma20 if ma20 > 0 else np.nan
    f["BIAS_20"] = -bias_20 if not np.isnan(bias_20) else np.nan

    return f


def calc_factors_batch(stock_codes: List[str], lookback_days: int = 200) -> pd.DataFrame:
    """
    批量算因子矩阵 -- 从 trade_stock_daily 读

    返回: DataFrame, index=股票代码, columns=因子名
    """
    rows = {}
    for code in stock_codes:
        df = load_kline_from_db(code, lookback_days=lookback_days)
        if df.empty:
            continue
        f = calc_factors_for_one(df)
        if f:
            rows[code] = f

    df_result = pd.DataFrame.from_dict(rows, orient="index")
    print(f"  [FACTOR] {len(df_result)} 只有效, "
          f"{len(stock_codes) - len(df_result)} 只数据不足被剔除")
    return df_result


# ============================================================
# 预处理: Winsorize + Z-score + 行业中性化 (复制自 CASE-C)
# ============================================================

def winsorize_mad(series: pd.Series, n: float = 3.0) -> pd.Series:
    s = series.copy()
    median = s.median()
    mad = (s - median).abs().median()
    if mad == 0 or np.isnan(mad):
        return s
    upper = median + n * 1.4826 * mad
    lower = median - n * 1.4826 * mad
    return s.clip(lower=lower, upper=upper)


def zscore(series: pd.Series) -> pd.Series:
    s = series.copy()
    mean = s.mean()
    std  = s.std(ddof=1)
    if std == 0 or np.isnan(std):
        return s * 0.0
    return (s - mean) / std


def industry_neutralize(factor_series: pd.Series, industry_map: dict) -> pd.Series:
    df = pd.DataFrame({
        "factor":   factor_series,
        "industry": pd.Series(industry_map),
    })
    df = df.dropna(subset=["industry"])
    return df.groupby("industry")["factor"].transform(zscore)


def preprocess_factors(factor_df: pd.DataFrame,
                        industry_map: Optional[dict] = None,
                        winsorize_n: float = 3.0,
                        neutralize: bool = True) -> pd.DataFrame:
    """三件套: 去极值 + Z-score + 行业中性化"""
    result = pd.DataFrame(index=factor_df.index)
    for col in factor_df.columns:
        s = factor_df[col].dropna()
        if len(s) == 0:
            result[col] = factor_df[col]
            continue
        s_w = winsorize_mad(s, n=winsorize_n)
        s_z = zscore(s_w)
        if neutralize and industry_map:
            s_z = industry_neutralize(s_z, industry_map)
            s_z = zscore(s_z)
        result[col] = s_z
    return result


# ============================================================
# 可交易过滤 (复制自 CASE-C/stock_pool.py)
# ============================================================

def filter_tradable(stock_codes: List[str], min_listed_days: int = 250) -> List[str]:
    """
    过滤掉 ST / 退市 / 上市不足 1 年的股票

    优先用 trade_stock_status 中的 list_date 字段;
    没有 list_date 时回退到调 xtdata.get_instrument_detail
    """
    if not stock_codes:
        return []

    placeholders = ",".join(["%s"] * len(stock_codes))
    rows = execute_query(
        f"SELECT stock_code, stock_name, list_date FROM trade_stock_status "
        f"WHERE stock_code IN ({placeholders})",
        stock_codes)
    info = {r["stock_code"]: r for r in rows}
    today = date.today()

    keep = []
    for code in stock_codes:
        meta = info.get(code)
        if not meta:
            continue
        name = (meta.get("stock_name") or "")
        if "ST" in name.upper() or "退" in name:
            continue
        listed = meta.get("list_date")
        if listed:
            try:
                ld = listed if isinstance(listed, date) else date.fromisoformat(str(listed))
                if (today - ld).days < min_listed_days:
                    continue
            except Exception:
                pass
        keep.append(code)

    return keep
