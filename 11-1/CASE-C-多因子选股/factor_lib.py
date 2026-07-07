# -*- coding: utf-8 -*-
# 21-CASE-C 多因子: 因子库
"""
FactorLib -- 因子库 (技术面 + 流动性, 都从 K 线计算)

为什么本项目主要做技术面因子?
    - 财务因子 (PE/PB/ROE) 需要财报数据, xtdata 财务接口较弱, 数据不稳定
    - 技术面因子能从 K 线直接算, 数据稳定 + 全市场可计算
    - 教学重点是"因子合成 -&gt; 中性化 -&gt; 分层回测"的 pipeline
    - 学员学完可以自己接 tushare / wind 等加上财务因子

10 个核心因子 (覆盖 5 大类):

    动量类 (3 个):
        - MOM_1M  -- 过去 21 日累计收益率
        - MOM_3M  -- 过去 63 日累计收益率
        - MOM_6M  -- 过去 126 日累计收益率

    反转类 (1 个):
        - REV_5D  -- 过去 5 日收益率 (短期反转, 取负号)

    波动率类 (2 个):
        - VOL_20  -- 过去 20 日年化波动率 (波动率因子, 取负号 -- 低波好)
        - VOL_60  -- 过去 60 日年化波动率

    流动性类 (2 个):
        - LIQ_20  -- 过去 20 日日均成交额对数 (取负号 -- 流动性好但被关注度高)
        - TURN_20 -- 过去 20 日日均换手率 (取负号 -- 低换手好)

    技术指标类 (2 个):
        - RSI_14  -- 14 日 RSI
        - BIAS_20 -- 20 日乖离率 ((close - MA20) / MA20, 取负号 -- 超涨反转)

每个因子都是"越大越好"的方向 (有 take_negative 参数自动取负)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional
import numpy as np
import pandas as pd


def _safe_pct_change(prices: pd.Series, periods: int) -> float:
    """用 prices.iloc[-1] 比 prices.iloc[-1-periods] 的 pct change"""
    if len(prices) <= periods:
        return np.nan
    p_now = prices.iloc[-1]
    p_then = prices.iloc[-1 - periods]
    if p_then <= 0:
        return np.nan
    return p_now / p_then - 1.0


def calc_factors_for_one(df: pd.DataFrame, total_share: float = 0) -> Dict[str, float]:
    """
    给定单只股票的日 K 线 DataFrame (含 close, volume, amount),
    返回 10 个因子值的字典 (键 = 因子名, 值 = float / nan)

    df 必须按时间升序排列, index 是日期, 列含 close / volume / amount
    """
    if df is None or len(df) < 130:   # 6 个月 = 126 日 + 几天 buffer
        return {}

    close = df["close"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df.columns else None
    amount = df["amount"].astype(float) if "amount" in df.columns else None

    # 日收益率序列
    returns = close.pct_change().dropna()
    if len(returns) < 100:
        return {}

    factors = {}

    # ---- 动量 ----
    factors["MOM_1M"] = _safe_pct_change(close, 21)
    factors["MOM_3M"] = _safe_pct_change(close, 63)
    factors["MOM_6M"] = _safe_pct_change(close, 126)

    # ---- 反转 (短期, 取负号 -- 短期上涨过多易回调) ----
    rev_5d = _safe_pct_change(close, 5)
    factors["REV_5D"] = -rev_5d if not np.isnan(rev_5d) else np.nan

    # ---- 波动率 (年化, 取负号 -- 低波动好) ----
    vol_20 = returns.tail(20).std() * math.sqrt(250)
    vol_60 = returns.tail(60).std() * math.sqrt(250)
    factors["VOL_20"] = -vol_20 if not np.isnan(vol_20) else np.nan
    factors["VOL_60"] = -vol_60 if not np.isnan(vol_60) else np.nan

    # ---- 流动性 ----
    if amount is not None and len(amount) >= 20:
        liq_20 = amount.tail(20).mean()
        # 取对数后取负号 (流动性好的票被过度关注 -- 反向 alpha)
        # 对极小值兜底
        factors["LIQ_20"] = -math.log(max(liq_20, 1.0))
    else:
        factors["LIQ_20"] = np.nan

    # ---- 换手率 (volume / float_share, 这里用 volume 简化) ----
    if volume is not None and len(volume) >= 20:
        if total_share > 0:
            turn_20 = (volume.tail(20).mean() / total_share) * 100  # 百分数
        else:
            # 没有股本数据, 用 volume / 长期均量 作为相对换手代理
            long_vol = volume.tail(60).mean()
            turn_20 = volume.tail(20).mean() / long_vol if long_vol > 0 else np.nan
        factors["TURN_20"] = -turn_20 if not np.isnan(turn_20) else np.nan
    else:
        factors["TURN_20"] = np.nan

    # ---- RSI 14 ----
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi_val = rsi.iloc[-1] if len(rsi) > 0 else np.nan
    # RSI 50 中性, 偏离 50 越远表示动能越强 -- 取 (RSI - 50) 作为强弱信号
    factors["RSI_14"] = (rsi_val - 50) if not np.isnan(rsi_val) else np.nan

    # ---- BIAS 20 (乖离率) ----
    ma20 = close.rolling(20).mean().iloc[-1]
    bias_20 = (close.iloc[-1] - ma20) / ma20 if ma20 > 0 else np.nan
    # 乖离过大易回调 -- 取负号
    factors["BIAS_20"] = -bias_20 if not np.isnan(bias_20) else np.nan

    return factors


# ============================================================
# 批量计算: 给定股票列表 + 截止日期, 返回因子矩阵
# ============================================================

def calc_factors_batch(stock_codes: List[str], end_date: str = "",
                       lookback_days: int = 200) -> pd.DataFrame:
    """
    批量算因子矩阵

    参数:
        stock_codes:  股票代码列表
        end_date:     截止日期 yyyymmdd, 空 = 最新
        lookback_days: 拉多少日 K 线 (至少 130 日才能算 6M 动量)

    返回:
        DataFrame, index=股票代码, columns=因子名
    """
    from xtquant import xtdata
    xtdata.connect()

    # 下载历史数据 (会触发增量下载)
    end_date_str = end_date if end_date else ""
    start_date_str = ""

    print(f"[FACTOR] 开始下载 {len(stock_codes)} 只股票的日 K 线 ...")
    for i, code in enumerate(stock_codes):
        try:
            xtdata.download_history_data(code, period="1d",
                                         start_time="20230101",
                                         end_time=end_date_str,
                                         incrementally=True)
        except Exception as e:
            print(f"  [WARN] {code} 下载失败: {e}")
        if (i + 1) % 50 == 0:
            print(f"  ... 进度 {i+1}/{len(stock_codes)}")

    print(f"[FACTOR] 拉数据 ...")
    data = xtdata.get_market_data_ex(
        field_list=["close", "volume", "amount"],
        stock_list=stock_codes, period="1d",
        start_time=start_date_str, end_time=end_date_str,
        count=lookback_days,
        dividend_type="back",   # 用后复权, 让因子免受除权干扰
    )

    print(f"[FACTOR] 计算 10 个因子 ...")
    rows = {}
    for code in stock_codes:
        df = data.get(code)
        if df is None or len(df) < 130:
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        f = calc_factors_for_one(df)
        if f:
            rows[code] = f

    df_result = pd.DataFrame.from_dict(rows, orient="index")
    print(f"[FACTOR] 完成: {len(df_result)} 只有效, "
          f"{len(stock_codes) - len(df_result)} 只数据不足被剔除")
    return df_result


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="因子库 demo")
    parser.add_argument("--codes", default="600519.SH,000001.SZ,002594.SZ,300750.SZ,513100.SH",
                        help="测试股票, 逗号分隔")
    args = parser.parse_args()

    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    df = calc_factors_batch(codes)

    print(f"\n{'='*70}")
    print(f"  因子矩阵 (raw, 未标准化)")
    print(f"{'='*70}\n")
    print(df.round(4).to_string())

    print(f"\n[简单解读]")
    print(f"  - MOM_*  : 越正动能越强")
    print(f"  - REV_5D : 越正越超跌 (反转机会)")
    print(f"  - VOL_*  : 越接近 0 波动越低 (取了负号)")
    print(f"  - LIQ_20 : 越接近 0 流动性越好的反向 (取了负号)")
    print(f"  - RSI_14 : 越正动能越强 (50 中性)")
    print(f"  - BIAS_20: 越正越超跌 (20 日乖离取了负号)")


if __name__ == "__main__":
    main()
