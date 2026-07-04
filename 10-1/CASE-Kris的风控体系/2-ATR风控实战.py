# -*- coding: utf-8 -*-
"""
CASE: ATR 风控实战

核心命题: 让每笔交易承担恒定的风险

ATR 在风控中的两次出现:
    [事前] ATR 仓位:  建议仓位 = (总资产 * 1%) / ATR * 价格
                     -- 高波动 ATR 大 -> 仓位小; 低波动 ATR 小 -> 仓位大
    [事中] ATR 止损:  止损价 = 入场价 - 2 * ATR
                     -- 比固定百分比止损更智能, 自动适配市场波动

本脚本演示三件事:
    1) 用真实数据 (默认 510050.SH) 计算 ATR
    2) 对比 "高波动期" vs "低波动期" 的建议仓位差异
    3) 实战对比: ATR止损 vs 固定5%止损 在同一段行情下的表现

调用:
    python 2-ATR风控实战.py                  # 默认 510050.SH 上证50ETF
    python 2-ATR风控实战.py 600519           # 茅台 (自动补 .SH)
    python 2-ATR风控实战.py 002594           # 比亚迪 (自动补 .SZ)
    python 2-ATR风控实战.py 600519.SH 2022-01-01 2025-12-31
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from data_loader import load_stock_data
from importlib import import_module

risk_engine = import_module("1-风控引擎")
RiskManager = risk_engine.RiskManager
Order = risk_engine.Order
Decision = risk_engine.Decision


def normalize_stock_code(code: str) -> str:
    """
    归一化股票代码: 没有后缀时按首位数字补 .SH / .SZ
        6 开头  -> .SH (上交所)
        0/3开头 -> .SZ (深交所)
        5/9开头 -> .SH (基金/B股)
    """
    if '.' in code:
        return code.upper()
    if code.startswith('6') or code.startswith('5') or code.startswith('9'):
        return f"{code}.SH"
    if code.startswith('0') or code.startswith('3'):
        return f"{code}.SZ"
    return code.upper()


# ============================================================
# ATR 计算 (海龟标准 Wilder 平滑)
# ============================================================

def calc_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    ATR = period 日真实波幅的 Wilder 移动平均

    True Range = max(
        当日 high - 当日 low,
        |当日 high - 前日 close|,
        |当日 low  - 前日 close|,
    )
    """
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    atr = np.zeros(n)
    atr[:period] = np.nan
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(atr, index=df.index, name='atr')


# ============================================================
# Demo 1: 同一只票, 高/低波动期的仓位差异
# ============================================================

def demo_atr_position_sizing(df: pd.DataFrame, stock_code: str,
                              total_asset: float = 1_000_000):
    """
    在 120 日窗口里, 找出 ATR 最低和最高的两天, 对比建议仓位
    """
    print("\n" + "=" * 70)
    print(f"  [Demo 1] ATR 仓位: 高波动期 vs 低波动期 -- {stock_code}")
    print("=" * 70)

    atr = calc_atr(df, period=20)
    df = df.copy()
    df['atr'] = atr
    valid = df.dropna(subset=['atr']).iloc[-120:]   # 最近 120 个交易日

    low_vol_day = valid['atr'].idxmin()
    high_vol_day = valid['atr'].idxmax()

    kris = RiskManager()
    kris.start_day(total_asset)
    kris.macro.update_vix(18.0)

    rows = []
    for label, day in [("低波动期", low_vol_day), ("高波动期", high_vol_day)]:
        atr_val = float(df.loc[day, 'atr'])
        price = float(df.loc[day, 'close'])
        suggested = (total_asset * 0.01) / atr_val * price
        atr_stop = price - 2 * atr_val

        rows.append({
            '场景': label,
            '日期': day.strftime('%Y-%m-%d'),
            '收盘价': f"{price:.3f}",
            'ATR(20)': f"{atr_val:.4f}",
            '建议仓位(元)': f"{suggested:,.0f}",
            'ATR止损价': f"{atr_stop:.3f}",
        })

        # 用 Kris 跑一遍实际审批
        portfolio = {
            'total_asset': total_asset,
            'prices': {stock_code: price},
            'atr': {stock_code: atr_val},
        }
        # 故意给一个固定 10 万的订单, 看高波动时是否触发 WARN
        order = Order(stock_code, 'buy', 100_000, price)
        d = kris.approve(order, portfolio, {'news_text': ''})
        rows[-1]['Kris决策(10万订单)'] = d.decision.value

    print(pd.DataFrame(rows).to_string(index=False))

    print("\n  解读:")
    print("    - 低波动期 ATR 小 -> 建议仓位大 -> 10万通过 (风险小, 多买无妨)")
    print("    - 高波动期 ATR 大 -> 建议仓位小 -> 同样的10万订单可能触发 WARN")
    print("    - 整体效果: 不论市场怎么变, 每笔交易承担的风险 (1% 账户) 恒定")

    return rows, low_vol_day, high_vol_day


# ============================================================
# Demo 2: ATR 止损 vs 固定 5% 止损
# ============================================================

def demo_atr_stop_vs_fixed(df: pd.DataFrame, entry_idx: int = -120,
                            initial_cash: float = 100_000):
    """
    在第 entry_idx 天入场, 持有到结束:
      A. ATR 止损:  入场价 - 2 * ATR (动态)
      B. 固定 5%:   入场价 * 0.95   (固定不变)
    比较两种止损方式触发的时点和最终持有金额。
    """
    print("\n" + "=" * 70)
    print("  [Demo 2] ATR 止损 vs 固定 5% 止损")
    print("=" * 70)

    atr = calc_atr(df, period=20)
    df = df.copy()
    df['atr'] = atr
    df = df.dropna(subset=['atr'])

    if len(df) <= abs(entry_idx) + 5:
        entry_idx = -min(60, len(df) - 5)

    entry_row = df.iloc[entry_idx]
    entry_date = entry_row.name
    entry_price = float(entry_row['close'])
    entry_atr = float(entry_row['atr'])
    shares = int(initial_cash / entry_price / 100) * 100

    # 止损价
    atr_stop_price = entry_price - 2 * entry_atr        # 海龟 2N 止损
    fixed_stop_price = entry_price * 0.95               # 固定 5%

    print(f"  入场日期:     {entry_date.strftime('%Y-%m-%d')}")
    print(f"  入场价:       {entry_price:.3f}")
    print(f"  ATR(20):      {entry_atr:.4f}")
    print(f"  ATR 止损价:   {atr_stop_price:.3f} (= {entry_price:.3f} - 2 * {entry_atr:.4f})")
    print(f"  固定 5% 止损: {fixed_stop_price:.3f}")
    print(f"  入场股数:     {shares:,}")

    after = df.iloc[entry_idx + 1:]
    atr_stop_day = None
    atr_stop_price_actual = None
    fixed_stop_day = None
    fixed_stop_price_actual = None

    for date, row in after.iterrows():
        low = float(row['low'])
        if atr_stop_day is None and low <= atr_stop_price:
            atr_stop_day = date
            atr_stop_price_actual = atr_stop_price
        if fixed_stop_day is None and low <= fixed_stop_price:
            fixed_stop_day = date
            fixed_stop_price_actual = fixed_stop_price
        if atr_stop_day and fixed_stop_day:
            break

    end_price = float(after.iloc[-1]['close']) if len(after) > 0 else entry_price

    def _summary(label, stop_day, stop_price):
        if stop_day is None:
            final_value = shares * end_price
            holding_pnl = (end_price - entry_price) / entry_price
            print(f"  [{label}] 全程未触发, 持有到末日 {after.index[-1].strftime('%Y-%m-%d')}, "
                  f"终价 {end_price:.3f}, 浮盈 {holding_pnl:+.2%}, "
                  f"持仓金额 {final_value:,.0f}")
        else:
            final_value = shares * stop_price
            realized = (stop_price - entry_price) / entry_price
            days = (stop_day - entry_date).days
            print(f"  [{label}] 第 {days} 天触发 ({stop_day.strftime('%Y-%m-%d')}), "
                  f"成交价 {stop_price:.3f}, 实亏 {realized:+.2%}, "
                  f"剩余金额 {final_value:,.0f}")

    print()
    _summary("ATR止损   ", atr_stop_day, atr_stop_price_actual)
    _summary("固定5%止损", fixed_stop_day, fixed_stop_price_actual)

    print("\n  解读:")
    if atr_stop_day and fixed_stop_day and atr_stop_day < fixed_stop_day:
        print("    -> ATR 止损先触发: 海龟逻辑认为价格已偏离正常波动 2N, 趋势可能反转")
        print("    -> 固定止损更宽: 损失更大 (或还没触发)")
    elif atr_stop_day and not fixed_stop_day:
        print("    -> ATR 止损保护到位, 固定 5% 止损完全没触发 (亏损一直在扩大)")
    elif fixed_stop_day and not atr_stop_day:
        print("    -> 本段行情 ATR 较大, 固定 5% 太紧反而被止损出局, ATR 给了趋势空间")
    else:
        print("    -> 两者都未触发或同时触发, 本段行情趋势平稳")

    return entry_date, entry_price, atr_stop_day, fixed_stop_day


# ============================================================
# Demo 3: ATR 止损在 Kris 中的实际调用
# ============================================================

def demo_kris_atr_stop_loop(stock_code: str, df: pd.DataFrame):
    """
    展示如何在持仓循环中使用 kris.check_atr_stop()
    入场价、ATR 都用真实数据的最后一天, 价格路径用最近 8 天真实低点
    """
    print("\n" + "=" * 70)
    print(f"  [Demo 3] ATR 止损在 Kris 中的调用方式 -- {stock_code}")
    print("=" * 70)

    atr = calc_atr(df, period=20)
    df_atr = df.copy()
    df_atr['atr'] = atr
    df_atr = df_atr.dropna(subset=['atr'])

    # 取倒数第 9 天作为入场, 之后 8 天做模拟
    entry_idx = -9 if len(df_atr) > 10 else -min(len(df_atr) - 1, 5)
    entry_row = df_atr.iloc[entry_idx]
    entry_price = float(entry_row['close'])
    atr_value = float(entry_row['atr'])
    stop_price = entry_price - 2 * atr_value

    kris = RiskManager()
    kris.start_day(1_000_000)
    kris.macro.update_vix(18.0)
    kris.register_position(stock_code, entry_price, atr_value)
    print(f"  [入场 {entry_row.name.strftime('%Y-%m-%d')}] {stock_code} @ {entry_price:.3f}, "
          f"ATR={atr_value:.3f}, 止损价 = {stop_price:.3f}")

    # 真实后续日的最低价路径
    after = df_atr.iloc[entry_idx + 1:]
    triggered = False
    for date, row in after.iterrows():
        low = float(row['low'])
        d = kris.check_atr_stop(stock_code, low)
        if d:
            print(f"  [{date.strftime('%Y-%m-%d')}] 当日最低 {low:.3f} -> {d}")
            kris.remove_position(stock_code)
            triggered = True
            break
        else:
            print(f"  [{date.strftime('%Y-%m-%d')}] 当日最低 {low:.3f}, 未触发止损")
    if not triggered:
        print(f"  [说明] 数据末尾 {len(after)} 天均未触发, 该段行情趋势平稳")


# ============================================================
# 可视化: ATR 走势 + 止损线对比
# ============================================================

def plot_atr_demo(df: pd.DataFrame, stock_code: str, entry_date, entry_price,
                   atr_stop_day, fixed_stop_day, save_path: str):
    """画两图: 上图价格 + 两种止损线, 下图 ATR 走势"""
    atr = calc_atr(df, period=20)
    df = df.copy()
    df['atr'] = atr
    plot_df = df.iloc[-180:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [2, 1]})

    ax1.plot(plot_df.index, plot_df['close'], color='#2c3e50',
             linewidth=1.6, label='收盘价')
    ax1.fill_between(plot_df.index, plot_df['low'], plot_df['high'],
                      color='#bdc3c7', alpha=0.3, label='High-Low')

    if entry_date in plot_df.index:
        ax1.axvline(entry_date, color='#3498db', linestyle='--',
                    alpha=0.7, label=f'入场 {entry_date.strftime("%Y-%m-%d")}')
        atr_at_entry = float(df.loc[entry_date, 'atr'])
        atr_stop_line = entry_price - 2 * atr_at_entry
        fixed_stop_line = entry_price * 0.95
        ax1.axhline(atr_stop_line, color='#27ae60', linestyle='-',
                    linewidth=1.6, label=f'ATR止损 {atr_stop_line:.3f}')
        ax1.axhline(fixed_stop_line, color='#e74c3c', linestyle='-',
                    linewidth=1.6, label=f'固定5%止损 {fixed_stop_line:.3f}')

    if atr_stop_day and atr_stop_day in plot_df.index:
        ax1.scatter(atr_stop_day, df.loc[atr_stop_day, 'low'],
                    s=120, color='#27ae60', marker='v', zorder=5,
                    label='ATR止损触发')
    if fixed_stop_day and fixed_stop_day in plot_df.index:
        ax1.scatter(fixed_stop_day, df.loc[fixed_stop_day, 'low'],
                    s=120, color='#e74c3c', marker='v', zorder=5,
                    label='固定止损触发')

    ax1.set_title(f"ATR 止损 vs 固定 5% 止损 -- {stock_code}",
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel("价格")
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.plot(plot_df.index, plot_df['atr'], color='#9b59b6',
             linewidth=1.6, label='ATR(20)')
    ax2.fill_between(plot_df.index, 0, plot_df['atr'],
                      color='#9b59b6', alpha=0.2)
    ax2.set_ylabel("ATR")
    ax2.set_xlabel("日期")
    ax2.legend(loc='best')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  [图已保存] {save_path}")


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    # 命令行参数: stock_code [start_date] [end_date]
    raw_code = sys.argv[1] if len(sys.argv) > 1 else '510050.SH'
    start_date = sys.argv[2] if len(sys.argv) > 2 else '2023-01-01'
    end_date = sys.argv[3] if len(sys.argv) > 3 else '2025-12-31'
    stock_code = normalize_stock_code(raw_code)

    print("=" * 70)
    print(f"  CASE: ATR 风控实战")
    print(f"  股票代码: {stock_code}    时间范围: {start_date} ~ {end_date}")
    print("=" * 70)

    # 加载真实数据
    df = load_stock_data(stock_code, start_date, end_date)
    print(f"\n  加载 {stock_code} 数据 {len(df)} 条, "
          f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")

    # Demo 1: 同一只票高低波动期对比
    demo_atr_position_sizing(df, stock_code)

    # Demo 2: ATR 止损 vs 固定止损
    entry_date, entry_price, atr_stop_day, fixed_stop_day = \
        demo_atr_stop_vs_fixed(df, entry_idx=-180)

    # Demo 3: 在 Kris 中调用 ATR 止损 (用真实数据末尾几天)
    demo_kris_atr_stop_loop(stock_code, df)

    # 可视化
    import os
    os.makedirs('outputs', exist_ok=True)
    safe_code = stock_code.replace('.', '_')
    plot_atr_demo(df, stock_code, entry_date, entry_price,
                  atr_stop_day, fixed_stop_day,
                  f'outputs/2-ATR风控实战-{safe_code}.png')
