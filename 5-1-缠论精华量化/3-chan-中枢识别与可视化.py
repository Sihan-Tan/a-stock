# -*- coding: utf-8 -*-
"""
第09讲：缠论精华量化 (chan.py 版)
脚本3-chan：中枢识别与可视化

chan.py 中枢的独特优势:
  - 区分中枢区间 [ZD, ZG] 和波动区间 [peak_low, peak_high]
    中枢区间: 多笔重叠的理论区间 (ZG 和 ZD)
    波动区间: 该中枢区域实际达到的最高/最低价 (可能超出 ZG/ZD)
    这个区分有助于更精确地判断支撑阻力
  - 记录进入段 (bi_in) 和离开段 (bi_out)
    可以分析"进入中枢的笔"和"离开中枢的笔"的力度对比
  - 自动识别线段和线段内的中枢
    线段 = 同方向的多笔构成的更大级别走势
    线段内的中枢可以判断线段是否延续
  - 支持中枢合并 (多个小中枢合并成大中枢)
"""

import os
import numpy as np
from data_loader import load_stock_data
from chan_analyzer import ChanAnalyzer
from chanpy_wrapper import run_chan, draw_chan_chart

STOCK_CODE = '300570.SZ'
START_DATE = '2025-06-01'
END_DATE = '2026-03-31'


def main():
    print("=" * 60)
    print("第09讲 | 脚本3-chan: 中枢识别与可视化 (chan.py版)")
    print("=" * 60)

    print(f"\n[1] 加载 {STOCK_CODE} 日线数据 ({START_DATE} ~ {END_DATE})...")
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    print(f"    共 {len(df)} 根K线")

    # 2. chan.py 分析
    print("\n[2] chan.py 分析...")
    cp = run_chan(df, symbol=STOCK_CODE)

    print(f"    笔: {len(cp['bi_list'])}")
    print(f"    线段: {len(cp['seg_list'])}")
    print(f"    笔级中枢: {len(cp['zs_list'])}")

    # 3. 中枢详情
    # chan.py 的中枢包含:
    #   - 中枢区间 [ZD, ZG]: 多笔重叠的理论区间
    #   - 波动区间 [peak_low, peak_high]: 实际达到的极值
    #   - 中心价位: 中枢区间的中位数
    print(f"\n[3] 笔级中枢列表:")
    for i, zs in enumerate(cp['zs_list']):
        sd = zs['start_date'].strftime('%Y-%m-%d') if zs['start_date'] else '?'
        ed = zs['end_date'].strftime('%Y-%m-%d') if zs['end_date'] else '?'
        width_pct = (zs['ZG'] - zs['ZD']) / zs['center'] * 100
        wave_pct = (zs['peak_high'] - zs['peak_low']) / zs['center'] * 100
        print(f"    [{i+1}] {sd} ~ {ed}")
        print(f"        中枢区间: ZD={zs['ZD']:.2f} ~ ZG={zs['ZG']:.2f} (宽度{width_pct:.1f}%)")
        print(f"        波动区间: {zs['peak_low']:.2f} ~ {zs['peak_high']:.2f} (幅度{wave_pct:.1f}%)")
        print(f"        中心价位: {zs['center']:.2f}")

    # 4. 线段详情
    # 线段 = 多个同方向笔构成的更大级别走势
    # 线段包含多个中枢, 因此比笔更稳定
    print(f"\n[4] 线段列表:")
    for i, seg in enumerate(cp['seg_list']):
        direction = '上升' if seg['direction'] == 'up' else '下降'
        sd = seg['start_date'].strftime('%Y-%m-%d')
        ed = seg['end_date'].strftime('%Y-%m-%d')
        print(f"    [{i+1}] {direction} | {sd} ~ {ed} | 包含{seg['bi_count']}笔")
        for j, sz in enumerate(seg['zs_list']):
            print(f"         线段内中枢: ZD={sz['ZD']:.2f} ZG={sz['ZG']:.2f}")

    # 5. 与自研对比
    print(f"\n[5] 与自研 ChanAnalyzer 对比:")
    analyzer = ChanAnalyzer(df)
    analyzer.analyze()

    print(f"    {'':>16s} | {'chan.py':>8s} | {'自研':>8s}")
    print(f"    {'-'*16}-+-{'-'*8}-+-{'-'*8}")
    print(f"    {'笔数':>14s} | {len(cp['bi_list']):>8d} | {len(analyzer.bi_list):>8d}")
    print(f"    {'线段':>14s} | {len(cp['seg_list']):>8d} | {'(无)':>8s}")
    print(f"    {'笔级中枢':>11s} | {len(cp['zs_list']):>8d} | {len(analyzer.zhongshu_list):>8d}")

    # 6. 可视化
    print(f"\n[6] 生成中枢对比图表...")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.dates as mdates
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 14))

    def _lx(d2x, date_val):
        if date_val in d2x:
            return d2x[date_val]
        if hasattr(date_val, 'date'):
            return d2x.get(date_val.date())
        return None

    # 上图: chan.py (笔 + 线段 + 中枢)
    d2x1 = draw_chan_chart(ax1, df, cp, show_bi=True, show_seg=True,
                           show_zs=True, show_bsp=False, show_fractals=False)
    ax1.set_title(f'chan.py | {STOCK_CODE} | 笔:{len(cp["bi_list"])} '
                  f'线段:{len(cp["seg_list"])} 中枢:{len(cp["zs_list"])}',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('价格')
    ax1.grid(True, alpha=0.3)

    # 图例: 在 ax1 上标注笔、线段、中枢的图示
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], color='#e74c3c', linewidth=1.8, label='笔(上升) - 最小波段'),
        Line2D([0], [0], color='#27ae60', linewidth=1.8, label='笔(下降) - 最小波段'),
        Line2D([0], [0], color='#9b59b6', linewidth=3.5, linestyle='--',
               label='线段 - 多笔构成的大趋势'),
        Line2D([0], [0], color='#3498db', linewidth=6, alpha=0.3,
               label='中枢 - 笔的重叠区域'),
    ]
    ax1.legend(handles=legend_items, loc='upper right', fontsize=9,
               framealpha=0.9, edgecolor='#bdc3c7')

    # 下图: 自研 (笔 + 中枢)
    d2x2 = ChanAnalyzer._draw_candlestick(ax2, df, width_ratio=0.6)
    for bi in analyzer.bi_list:
        c = '#e74c3c' if bi['direction'] == 'up' else '#27ae60'
        sx = _lx(d2x2, bi.get('start_raw_date', bi['start_date']))
        ex = _lx(d2x2, bi.get('end_raw_date', bi['end_date']))
        if sx is None or ex is None:
            continue
        ax2.plot([sx, ex], [bi['start_price'], bi['end_price']],
                 color=c, linewidth=1.8, zorder=4)
    for zs in analyzer.zhongshu_list:
        x_left = _lx(d2x2, zs['start_date'])
        x_right = _lx(d2x2, zs['end_date'])
        if x_left is None or x_right is None:
            continue
        rect = patches.Rectangle(
            (x_left, zs['ZD']), x_right - x_left, zs['ZG'] - zs['ZD'],
            linewidth=1.5, edgecolor='#3498db', facecolor='#3498db', alpha=0.15, zorder=3)
        ax2.add_patch(rect)
        ax2.text(x_left, zs['ZG'], f" ZG={zs['ZG']:.1f}\n ZD={zs['ZD']:.1f}",
                 fontsize=8, color='#2c3e50', va='bottom')
    ax2.set_title(f'ChanAnalyzer (自研) | {STOCK_CODE} | 笔:{len(analyzer.bi_list)} '
                  f'中枢:{len(analyzer.zhongshu_list)}',
                  fontsize=13, fontweight='bold')
    ax2.set_ylabel('价格')
    ax2.set_xlabel('日期')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/3-chan-中枢对比.png', dpi=150, bbox_inches='tight')
    print(f"  图表已保存: outputs/3-chan-中枢对比.png")
    plt.close()

    print("\n完成!")


if __name__ == '__main__':
    main()
