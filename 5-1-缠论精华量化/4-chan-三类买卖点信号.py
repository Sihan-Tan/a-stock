# -*- coding: utf-8 -*-
"""
第09讲：缠论精华量化 (chan.py 版)
脚本4-chan：三类买卖点信号

以宁德时代(300750.SZ)为主例, 它在2023-2025年间展现出完整的三种买点结构:
  - 2024-01-30 一买(背驰): MACD面积缩小, 下跌力度衰减 → 趋势背驰
  - 2024-02-27 二买(回调确认): 反弹后回调不破一买低点 → 双底确认
  - 2024-04-24 三买(中枢突破): 突破中枢ZG后回踩不破 → 主升浪启动

chan.py 买卖点的独特优势:
  - 多类型细分: 二买S(类二买)、三买a/三买b、一买p(盘整背驰)
  - 每笔自带 MACD 面积: 直接用于背驰判断
  - 更成熟的买卖点检测算法

生成两张图:
  - 图1: 全景总览 (K线 + 笔 + 中枢 + 所有买卖点)
  - 图2: 三类买卖点分类详解 (每类一行: K线局部放大 + MACD柱状图)

教学目的:
  - 展示真实股票中三类买点的完整出现过程
  - 通过局部放大和MACD辅助, 帮助理解每类买卖点的内在逻辑
  - 对比 chan.py 的买卖点类型与自研实现
"""

import os
import numpy as np
import pandas as pd
from data_loader import load_stock_data
from chanpy_wrapper import run_chan, draw_chan_chart

STOCK_CODE = '300750.SZ'
STOCK_NAME = '宁德时代'
START_DATE = '2023-01-01'
END_DATE = '2025-12-31'


def calc_macd(df, fast=12, slow=26, signal=9):
    """
    手动计算 MACD 指标

    使用指数移动平均 (EMA), 公式:
      DIF = EMA(close, 12) - EMA(close, 26)    ← 短期均线 - 长期均线
      DEA = EMA(DIF, 9)                           ← DIF 的平滑
      MACD柱 = (DIF - DEA) * 2                    ← 柱状图 (乘以2是为了放大视觉差异)

    判断方法:
      DIF 向上穿越 DEA → 金叉 (买入信号)
      DIF 向下穿越 DEA → 死叉 (卖出信号)
      MACD柱正且扩大 → 多头动能增强
      MACD柱负且扩大 → 空头动能增强

    背驰判断 (缠论中使用):
      价格创新低, 但 MACD 柱面积缩小 → 底背驰 (一买)
      价格创新高, 但 MACD 柱面积缩小 → 顶背驰 (一卖)
    """
    close = df['close'].astype(float)
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    macd_hist = (dif - dea) * 2
    return dif, dea, macd_hist


def main():
    print("=" * 60)
    print("第09讲 | 脚本4-chan: 三类买卖点信号 (chan.py版)")
    print("=" * 60)

    print(f"\n[1] 加载 {STOCK_NAME}({STOCK_CODE}) 日线数据 ({START_DATE} ~ {END_DATE})...")
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    print(f"    共 {len(df)} 根K线")

    # 2. chan.py 分析
    print("\n[2] chan.py 完整分析...")
    cp = run_chan(df, symbol=STOCK_CODE)

    print(f"    笔: {len(cp['bi_list'])}")
    print(f"    线段: {len(cp['seg_list'])}")
    print(f"    中枢: {len(cp['zs_list'])}")
    print(f"    买卖点: {len(cp['bsp_list'])}")

    # 3. 买卖点详情
    bsp_list = cp['bsp_list']
    bsp_names = {'1': '一买', '2': '二买', '2s': '类二买', '3a': '三买a',
                 '3b': '三买b', '3': '三买',
                 '1s': '一卖', '1p': '一买p',
                 '2_sell': '二卖', '3_sell': '三卖',
                 '2,3b': '二买+三买b'}

    print(f"\n[3] chan.py 买卖点列表:")
    print(f"    {'序号':>4} | {'日期':>12} | {'类型':>10} | {'买/卖':>6} | "
          f"{'价格':>10} | {'MACD面积':>10}")
    print("    " + "-" * 70)

    buy_points = []
    for i, bi in enumerate(bsp_list):
        bsp_type = bi['bsp_type']
        is_buy = bi['bsp_is_buy']
        name = bsp_names.get(bsp_type, bsp_type)
        bs = '买入' if is_buy else '卖出'
        date_str = bi['bsp_date'].strftime('%Y-%m-%d')
        print(f"    {i+1:>4} | {date_str:>12} | {name:>10} | {bs:>6} | "
              f"{bi['end_price']:>10.2f} | {bi['macd_area']:>10.1f}")
        if is_buy:
            buy_points.append(bi)

    # 4. 分类统计
    print(f"\n[4] 买卖点分类:")
    type_counts = {}
    for bi in bsp_list:
        t = bi['bsp_type']
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, cnt in sorted(type_counts.items()):
        name = bsp_names.get(t, t)
        print(f"    {name}: {cnt} 个")

    # 5. 买入信号后续收益
    if buy_points:
        print(f"\n[5] 买入信号后续收益:")
        print(f"    {'日期':>12} | {'类型':>10} | {'价格':>10} | "
              f"{'5日':>8} | {'10日':>8} | {'20日':>8}")
        print("    " + "-" * 70)

        for bi in buy_points:
            bsp_date = bi['bsp_date']
            price = bi['end_price']
            name = bsp_names.get(bi['bsp_type'], bi['bsp_type'])
            date_str = bsp_date.strftime('%Y-%m-%d')

            returns = {}
            for days in [5, 10, 20]:
                future_mask = df.index > bsp_date
                future = df.loc[future_mask]
                if len(future) >= days:
                    future_price = float(future['close'].iloc[days - 1])
                    ret = (future_price / price - 1) * 100
                    returns[days] = f'{ret:+.2f}%'
                else:
                    returns[days] = '  N/A'

            print(f"    {date_str:>12} | {name:>10} | {price:>10.2f} | "
                  f"{returns[5]:>8} | {returns[10]:>8} | {returns[20]:>8}")

    # 6. 可视化 - 图1: 全景总览
    # 显示完整的缠论结构 + 所有买卖点
    print(f"\n[6] 生成全景总览图...")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    from chan_analyzer import ChanAnalyzer
    from matplotlib.lines import Line2D

    fig1, ax1 = plt.subplots(1, 1, figsize=(24, 10))
    draw_chan_chart(ax1, df, cp, show_bi=True, show_seg=True,
                   show_zs=True, show_bsp=True, show_fractals=False)
    ax1.set_title(f'chan.py 买卖点总览 | {STOCK_NAME}({STOCK_CODE}) '
                  f'{START_DATE}~{END_DATE} | '
                  f'笔:{len(cp["bi_list"])} 中枢:{len(cp["zs_list"])} '
                  f'买卖点:{len(bsp_list)}',
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('价格')
    ax1.set_xlabel('日期')
    ax1.grid(True, alpha=0.3)

    legend_items = [
        Line2D([0], [0], color='#e74c3c', linewidth=1.8, label='笔(上升)'),
        Line2D([0], [0], color='#27ae60', linewidth=1.8, label='笔(下降)'),
        Line2D([0], [0], color='#9b59b6', linewidth=3.5, linestyle='--', label='线段'),
        Line2D([0], [0], color='#3498db', linewidth=6, alpha=0.3, label='中枢'),
        Line2D([0], [0], marker='^', color='#8e44ad', linestyle='None',
               markersize=10, label='一买(背驰)'),
        Line2D([0], [0], marker='^', color='#e67e22', linestyle='None',
               markersize=10, label='二买(回调确认)'),
        Line2D([0], [0], marker='^', color='#e74c3c', linestyle='None',
               markersize=10, label='三买(中枢突破)'),
    ]
    ax1.legend(handles=legend_items, loc='upper right', fontsize=9,
               framealpha=0.9, edgecolor='#bdc3c7')

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/4-chan-买卖点总览.png', dpi=150, bbox_inches='tight')
    print(f"  图表已保存: outputs/4-chan-买卖点总览.png")
    plt.close()

    # 7. 可视化 - 三类买卖点各自独立一张图 (K线 + MACD 上下排列)
    # 每类买点一张详细的局部放大图:
    #   上图: K线 + 笔 + 中枢 + 信号标注 (局部放大)
    #   下图: MACD 柱状图 + DIF/DEA 线
    # 通过 MACD 辅助理解背驰逻辑
    print(f"\n[7] 生成三类买卖点分类图(各一张)...")

    def _lx(d2x, date_val):
        if date_val in d2x:
            return d2x[date_val]
        if hasattr(date_val, 'date'):
            return d2x.get(date_val.date())
        return None

    # 三类买点的配置
    bsp_type_defs = [
        {
            'name': '第一类买点(一买)',
            'subtitle': '趋势背驰: 下跌还在继续, 但MACD面积缩小, 跌不动了',
            'match_types': ['1', '1p'],
            'color': '#8e44ad',
            'window_days': 120,   # 信号前后各120天
            'filename': '4-chan-一买详解.png',
        },
        {
            'name': '第二类买点(二买)',
            'subtitle': '回调确认: 反弹后回调, 低点不破一买低点 = 双底确认',
            'match_types': ['2', '2s', '2,3b'],
            'color': '#e67e22',
            'window_days': 100,
            'filename': '4-chan-二买详解.png',
        },
        {
            'name': '第三类买点(三买)',
            'subtitle': '中枢突破: 突破中枢ZG后回踩不破ZG = 新趋势确认',
            'match_types': ['3', '3a', '3b'],
            'color': '#e74c3c',
            'window_days': 100,
            'filename': '4-chan-三买详解.png',
        },
    ]

    dif, dea, macd_hist = calc_macd(df)

    for tdef in bsp_type_defs:
        pts = [b for b in buy_points if b['bsp_type'] in tdef['match_types']]

        if not pts:
            print(f"  {tdef['name']}: 无信号, 跳过")
            continue

        sig = pts[0]
        sig_date = sig['bsp_date']

        # 截取信号前后一个窗口的数据用于局部放大
        win_start = sig_date - pd.Timedelta(days=tdef['window_days'])
        win_end = sig_date + pd.Timedelta(days=tdef['window_days'])
        df_win = df[(df.index >= win_start) & (df.index <= win_end)]
        if len(df_win) < 10:
            df_win = df

        fig, (ax_k, ax_macd) = plt.subplots(
            2, 1, figsize=(22, 12),
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})

        # --- 上图: K线 + 笔 + 中枢 + 信号 ---
        d2x = ChanAnalyzer._draw_candlestick(ax_k, df_win, width_ratio=0.6)

        # 绘制笔(淡色背景)
        for bi in cp['bi_list']:
            sx = _lx(d2x, bi['start_raw_date'])
            ex = _lx(d2x, bi['end_raw_date'])
            if sx is not None and ex is not None:
                c = '#e74c3c' if bi['direction'] == 'up' else '#27ae60'
                ax_k.plot([sx, ex], [bi['start_price'], bi['end_price']],
                          color=c, linewidth=1.2, alpha=0.4, zorder=4)

        # 绘制中枢
        for zs in cp['zs_list']:
            if zs['start_date'] and zs['end_date']:
                xl = _lx(d2x, zs['start_date'])
                xr = _lx(d2x, zs['end_date'])
                if xl is not None and xr is not None:
                    rect = mpatches.Rectangle(
                        (xl, zs['ZD']), xr - xl, zs['ZG'] - zs['ZD'],
                        linewidth=1.5, edgecolor='#3498db', facecolor='#3498db',
                        alpha=0.12, zorder=2)
                    ax_k.add_patch(rect)
                    ax_k.text(xl, zs['ZG'], f'ZG={zs["ZG"]:.0f}',
                              fontsize=8, color='#3498db', va='bottom')
                    ax_k.text(xl, zs['ZD'], f'ZD={zs["ZD"]:.0f}',
                              fontsize=8, color='#3498db', va='top')

        # 信号附近笔加粗 (方便观察信号前后的笔结构)
        for bi in cp['bi_list']:
            try:
                diff = abs((bi['end_raw_date'] - sig_date).days)
            except:
                continue
            if diff <= 60:
                sx = _lx(d2x, bi['start_raw_date'])
                ex = _lx(d2x, bi['end_raw_date'])
                if sx is not None and ex is not None:
                    c = '#e74c3c' if bi['direction'] == 'up' else '#27ae60'
                    ax_k.plot([sx, ex], [bi['start_price'], bi['end_price']],
                              color=c, linewidth=2.5, alpha=0.9, zorder=5)

        # 高亮信号点
        x_sig = _lx(d2x, sig_date)
        if x_sig is not None:
            price = sig['end_price']
            name = bsp_names.get(sig['bsp_type'], sig['bsp_type'])
            date_str = sig_date.strftime('%Y-%m-%d')

            ax_k.scatter(x_sig, price, marker='^', color=tdef['color'],
                         s=400, zorder=8, edgecolors='black', linewidths=2)
            ax_k.annotate(
                f'{name}\n{date_str}\n{price:.1f}',
                (x_sig, price),
                textcoords="offset points", xytext=(20, 25),
                fontsize=12, fontweight='bold', color=tdef['color'],
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=tdef['color'], alpha=0.95, linewidth=2),
                arrowprops=dict(arrowstyle='->', color=tdef['color'], lw=2),
                zorder=9)

            # 竖线贯穿 K线 和 MACD
            ax_k.axvline(x=x_sig, color=tdef['color'], linewidth=1.2,
                         linestyle=':', alpha=0.5, zorder=1)

        ax_k.set_title(
            f'{STOCK_NAME}({STOCK_CODE}) | {tdef["name"]} | {tdef["subtitle"]}',
            fontsize=14, fontweight='bold', color=tdef['color'])
        ax_k.set_ylabel('价格', fontsize=11)
        ax_k.grid(True, alpha=0.3)

        # --- 下图: MACD (与K线共享x轴) ---
        macd_win = macd_hist.reindex(df_win.index)
        dif_win = dif.reindex(df_win.index)
        dea_win = dea.reindex(df_win.index)

        dates = list(df_win.index)
        x_positions = list(range(len(dates)))

        colors_bar = ['#e74c3c' if v >= 0 else '#27ae60' for v in macd_win.values]
        ax_macd.bar(x_positions, macd_win.values, color=colors_bar, alpha=0.7, width=0.8)
        ax_macd.plot(x_positions, dif_win.values, color='#2980b9', linewidth=1.3, label='DIF')
        ax_macd.plot(x_positions, dea_win.values, color='#e67e22', linewidth=1.3, label='DEA')
        ax_macd.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)

        # MACD 上也画信号竖线
        if x_sig is not None:
            ax_macd.axvline(x=x_sig, color=tdef['color'], linewidth=1.2,
                            linestyle=':', alpha=0.6)

        ax_macd.set_ylabel('MACD', fontsize=11)
        ax_macd.legend(fontsize=9, loc='upper left')
        ax_macd.grid(True, alpha=0.2)

        fig.subplots_adjust(left=0.06, right=0.97, top=0.93, bottom=0.06)
        save_path = os.path.join('outputs', tdef['filename'])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  图表已保存: {save_path}")
        plt.close()

    print("\n完成!")


if __name__ == '__main__':
    main()
