# -*- coding: utf-8 -*-
"""
CASE: 宏观门控实战

核心命题: 个股都没事, 但天塌了 -- Kris 需要一个 "宏观雷达"

QVIX 是 A 股 50ETF 期权隐含波动率, 由上交所 50ETF 期权价格反推得出,
是 A 股本土的 "恐慌指数", 比美 VIX 更贴近 A 股交易场景。

本脚本演示三件事 (全部基于 akshare 真实 QVIX 历史数据):
    [Part 1] 拉取 A 股 50ETF QVIX 历史数据 (2015 至今)
    [Part 2] 历史 QVIX 极端日 + 最近 N 天: Kris 的决策回放
    [Part 3] VIX 区间 -> 仓位系数 映射可视化 + 真实数据点叠加
    [Part 4] 完整决策链: 同一笔订单在不同 QVIX 区间的命运

调用:
    python 4-宏观门控实战.py             # 默认拉全量历史 QVIX
    python 4-宏观门控实战.py 60          # 历史最高 60 天 (默认 5)

依赖:
    pip install akshare
"""
import os
import sys
import numpy as np
import pandas as pd
import akshare as ak
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

from importlib import import_module
risk_engine = import_module("1-风控引擎")
RiskManager = risk_engine.RiskManager
MacroGate = risk_engine.MacroGate
Order = risk_engine.Order
Decision = risk_engine.Decision


# ============================================================
# Part 1: akshare 拉取 50ETF QVIX 历史数据
# ============================================================

def part1_fetch_qvix() -> pd.DataFrame:
    """[Part 1] 拉 A 股 50ETF QVIX 全部历史数据"""
    print("\n" + "=" * 100)
    print("  [Part 1] QVIX 数据拉取 -- ak.index_option_50etf_qvix()")
    print("=" * 100)

    df = ak.index_option_50etf_qvix()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"\n  共 {len(df)} 个交易日, 时间范围 "
          f"{df['date'].iloc[0].strftime('%Y-%m-%d')} ~ "
          f"{df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"  历史最高 QVIX: {df['close'].max():.2f} "
          f"(于 {df.loc[df['close'].idxmax(), 'date'].strftime('%Y-%m-%d')})")
    print(f"  历史最低 QVIX: {df['close'].min():.2f}")
    print(f"  最新 QVIX:    {df['close'].iloc[-1]:.2f} "
          f"(于 {df['date'].iloc[-1].strftime('%Y-%m-%d')})")
    return df


# ============================================================
# Part 2: 历史 QVIX 极端日 + 最近 N 天的 Kris 决策回放
# ============================================================

def part2_qvix_replay(df: pd.DataFrame, top_n: int = 5):
    """[Part 2] 用真实 QVIX 数据跑 Kris 决策回放"""
    print("\n" + "=" * 100)
    print(f"  [Part 2] Kris 决策回放 -- 历史最高 {top_n} 天 + 最近 5 天")
    print("=" * 100)

    extreme_df = df.nlargest(top_n, 'close').sort_values('close', ascending=False)
    recent_df = df.tail(5)

    gate = MacroGate()
    print(f"\n  --- 历史 QVIX 最高 {top_n} 天 ---")
    print(f"  {'日期':<12} {'QVIX':<8} {'仓位系数':<10} {'风险等级':<12} {'决策':<8}")
    print("  " + "-" * 60)
    for _, row in extreme_df.iterrows():
        coeff = gate.update_vix(float(row['close']))
        d = gate.check()
        print(f"  {row['date'].strftime('%Y-%m-%d'):<12} "
              f"{row['close']:<8.2f} {coeff:<10.0%} "
              f"{gate.risk_level:<12} {d.decision.value:<8}")

    print(f"\n  --- 最近 5 个交易日 ---")
    print(f"  {'日期':<12} {'QVIX':<8} {'仓位系数':<10} {'风险等级':<12} {'决策':<8}")
    print("  " + "-" * 60)
    for _, row in recent_df.iterrows():
        coeff = gate.update_vix(float(row['close']))
        d = gate.check()
        print(f"  {row['date'].strftime('%Y-%m-%d'):<12} "
              f"{row['close']:<8.2f} {coeff:<10.0%} "
              f"{gate.risk_level:<12} {d.decision.value:<8}")


# ============================================================
# Part 3: VIX -> 仓位系数 映射可视化 + 真实数据散点
# ============================================================

def part3_plot_map(df: pd.DataFrame, save_path: str):
    """[Part 3] 画映射函数, 叠加全量真实 QVIX 数据散点 + 标注极端日"""
    print("\n" + "=" * 100)
    print(f"  [Part 3] 映射函数可视化 + {len(df)} 个真实 QVIX 数据点叠加")
    print("=" * 100)

    gate = MacroGate()
    vix_range = np.linspace(0, 90, 901)
    coeffs = [gate.update_vix(float(v)) for v in vix_range]

    real_coeffs = [gate.update_vix(float(v)) for v in df['close']]

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.fill_between(vix_range, 0, coeffs, color='#3498db', alpha=0.2)
    ax.plot(vix_range, coeffs, color='#3498db', linewidth=2.5,
            label='Kris 仓位系数 = f(VIX)')

    ax.scatter(df['close'], real_coeffs, s=10, alpha=0.25,
               color='#16a085', label=f'真实 QVIX ({len(df)} 个交易日)')

    zones = [
        (10, 1.0, '正常\n100%', '#27ae60'),
        (22.5, 0.85, '焦虑\n70-100%', '#f39c12'),
        (30, 0.4, '恐慌\n10-70%', '#e67e22'),
        (60, 0.0, '极度恐慌/末日\n0-10%', '#e74c3c'),
    ]
    for x, y, label, color in zones:
        ax.annotate(label, xy=(x, y), fontsize=11, ha='center',
                    fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.4',
                              facecolor='white', edgecolor=color, alpha=0.9))

    for thr in [20, 25, 35, 50]:
        ax.axvline(thr, color='gray', linestyle='--', alpha=0.4)

    top5 = df.nlargest(5, 'close')
    for _, row in top5.iterrows():
        coeff = gate.update_vix(float(row['close']))
        ax.scatter(row['close'], coeff, s=180, color='#c0392b',
                   edgecolor='white', linewidth=2, zorder=5)
        ax.annotate(f"{row['date'].strftime('%Y-%m-%d')}\nQVIX={row['close']:.1f}",
                    xy=(row['close'], coeff),
                    xytext=(row['close'] - 8, coeff + 0.12),
                    fontsize=9, color='#c0392b',
                    arrowprops=dict(arrowstyle='->', color='#c0392b', alpha=0.6))

    last = df.iloc[-1]
    coeff_last = gate.update_vix(float(last['close']))
    ax.scatter(last['close'], coeff_last, s=180, color='#2980b9',
               edgecolor='white', linewidth=2, zorder=5)
    ax.annotate(f"最新 {last['date'].strftime('%Y-%m-%d')}\nQVIX={last['close']:.1f}",
                xy=(last['close'], coeff_last),
                xytext=(last['close'] + 5, coeff_last - 0.15),
                fontsize=9, color='#2980b9',
                arrowprops=dict(arrowstyle='->', color='#2980b9', alpha=0.6))

    ax.set_xlabel('QVIX (50ETF 期权隐含波动率)', fontsize=13)
    ax.set_ylabel('仓位系数 (0~1)', fontsize=13)
    ax.set_title('Kris 宏观门控: QVIX -> 仓位系数 映射 + 真实历史散点',
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 90)
    ax.set_ylim(-0.05, 1.18)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  [图已保存] {save_path}")


# ============================================================
# Part 4: 同一笔订单在不同 QVIX 区间的命运
# ============================================================

def part4_full_chain(df: pd.DataFrame):
    """
    [Part 4] 从真实 QVIX 历史中, 自动选出 4 个不同档位的代表日期,
            观察同一笔 "买入 50ETF 10万" 订单在 Kris 的不同审批结果。
    """
    print("\n" + "=" * 100)
    print("  [Part 4] 完整决策链 -- 同一笔订单, 不同 QVIX 档位的命运")
    print("=" * 100)

    pickers = [
        ('正常',     df[df['close'] < 18].sample(1, random_state=42)
                     if len(df[df['close'] < 18]) > 0 else None),
        ('焦虑',     df[(df['close'] >= 20) & (df['close'] < 25)].sample(1, random_state=42)
                     if len(df[(df['close'] >= 20) & (df['close'] < 25)]) > 0 else None),
        ('恐慌',     df[(df['close'] >= 25) & (df['close'] < 35)].sample(1, random_state=42)
                     if len(df[(df['close'] >= 25) & (df['close'] < 35)]) > 0 else None),
        ('极度恐慌', df[df['close'] >= 35].sample(1, random_state=42)
                     if len(df[df['close'] >= 35]) > 0 else None),
        ('末日',     df[df['close'] >= 50].sample(1, random_state=42)
                     if len(df[df['close'] >= 50]) > 0 else None),
    ]

    portfolio = {
        'total_asset': 1_000_000,
        'prices': {'510050.SH': 3.00},
        'atr': {'510050.SH': 0.05},
    }
    order_template = lambda: Order('510050.SH', 'buy', 100_000, 3.00)

    for tag, sample_df in pickers:
        if sample_df is None or len(sample_df) == 0:
            continue
        row = sample_df.iloc[0]
        date = row['date'].strftime('%Y-%m-%d')
        qvix = float(row['close'])

        print(f"\n  >>> [{tag}] {date}  QVIX={qvix:.2f}")
        kris = RiskManager()
        kris.start_day(1_000_000)
        kris.macro.update_vix(qvix)

        d = kris.approve(order_template(), portfolio,
                         {'news_text': '50ETF 成交活跃, 资金面平稳'})
        print(f"      宏观: QVIX={qvix:.2f} -> 仓位系数 "
              f"{kris.macro.position_coefficient:.0%} ({kris.macro.risk_level})")
        print(f"      Kris 决策: {d}")
        if d.decision == Decision.WARN:
            adjusted = 100_000 * d.max_position_pct
            print(f"      [建议执行] 把仓位从 100,000 降到 {adjusted:,.0f} 元")
        elif d.decision == Decision.HALT:
            print(f"      [建议执行] 不下单, 等待宏观环境好转")
        else:
            print(f"      [建议执行] 正常下单 100,000 元")


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print("=" * 100)
    print("  CASE: 宏观门控实战")
    print("  天塌时, 个股基本面没用 -- Kris 用 QVIX 做仓位总闸")
    print("=" * 100)

    df = part1_fetch_qvix()

    part2_qvix_replay(df, top_n=top_n)

    os.makedirs('outputs', exist_ok=True)
    part3_plot_map(df, 'outputs/4-宏观门控-QVIX映射.png')

    part4_full_chain(df)
