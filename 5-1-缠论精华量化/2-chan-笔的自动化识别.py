# -*- coding: utf-8 -*-
"""
第09讲：缠论精华量化 (chan.py 版)
脚本2-chan：笔的自动化识别

chan.py 的笔识别特点:
  - 支持严格笔和推笔两种模式 (通过 bi_strict 配置)
    严格笔: 必须满足最低K线数要求, 否则不成笔
    推笔: 放宽条件, 允许虚拟笔存在 (is_sure=False)
  - 每根笔带 is_sure 属性: 是否已确认
    is_sure=True: 笔的结束点是真实的顶底分型
    is_sure=False: 笔的结束点是虚拟的 (还差几根K线才能确认)
  - 内置 MACD 面积计算: per笔的力度指标 (用于背驰判断)
  - 每根笔关联所属线段 (seg_idx): 笔→段的结构关系

与自研 ChanAnalyzer 的对比维度:
  1. 笔的总数是否接近 (算法差异可能导致不同)
  2. 起止日期和价格是否吻合
  3. chan.py 额外提供的字段 (is_sure/macd_area) 有什么价值
"""

import os
import numpy as np
from data_loader import load_stock_data
from chan_analyzer import ChanAnalyzer
from chanpy_wrapper import run_chan, draw_chan_chart

STOCK_CODE = '600519.SH'
START_DATE = '2025-06-01'
END_DATE = '2025-12-31'


def main():
    print("=" * 60)
    print("第09讲 | 脚本2-chan: 笔的自动化识别 (chan.py版)")
    print("=" * 60)

    print(f"\n[1] 加载 {STOCK_CODE} 日线数据 ({START_DATE} ~ {END_DATE})...")
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    print(f"    共 {len(df)} 根K线")

    # 2. chan.py 笔识别
    print("\n[2] chan.py 笔识别...")
    cp = run_chan(df, symbol=STOCK_CODE)
    bi_list = cp['bi_list']

    up_bis = [b for b in bi_list if b['direction'] == 'up']
    down_bis = [b for b in bi_list if b['direction'] == 'down']
    print(f"    笔数: {len(bi_list)} (上升 {len(up_bis)}, 下降 {len(down_bis)})")

    # 3. 笔列表详情
    # chan.py 的笔信息更丰富:
    #   - klu_count: 包含的原始K线数 (比合并K线数更精确)
    #   - macd_area: 该笔的MACD柱状图面积 (力度指标)
    #   - is_sure: 笔的确认状态
    print(f"\n[3] chan.py 笔列表:")
    print(f"    {'序号':>4} | {'方向':>4} | {'起始':>10} | {'结束':>10} | "
          f"{'起价':>8} | {'终价':>8} | {'涨跌幅':>8} | {'K线数':>5} | {'MACD面积':>8} | {'确认':>4}")
    print("    " + "-" * 95)
    for i, bi in enumerate(bi_list):
        pct = (bi['end_price'] / bi['start_price'] - 1) * 100
        direction = '上升' if bi['direction'] == 'up' else '下降'
        sure = 'Y' if bi['is_sure'] else 'N'
        sd = bi['start_date'].strftime('%m-%d')
        ed = bi['end_date'].strftime('%m-%d')
        print(f"    {i+1:>4} | {direction:>4} | {sd:>10} | {ed:>10} | "
              f"{bi['start_price']:>8.2f} | {bi['end_price']:>8.2f} | "
              f"{pct:>+7.2f}% | {bi['klu_count']:>5} | "
              f"{bi['macd_area']:>8.1f} | {sure:>4}")

    # 4. 统计对比
    print(f"\n[4] 笔的统计分析:")
    if up_bis:
        up_pcts = [(b['end_price'] / b['start_price'] - 1) * 100 for b in up_bis]
        up_areas = [b['macd_area'] for b in up_bis]
        print(f"    上升笔 ({len(up_bis)} 笔): 平均涨幅={np.mean(up_pcts):+.2f}%, "
              f"平均MACD面积={np.mean(up_areas):.1f}")
    if down_bis:
        down_pcts = [(b['end_price'] / b['start_price'] - 1) * 100 for b in down_bis]
        down_areas = [b['macd_area'] for b in down_bis]
        print(f"    下降笔 ({len(down_bis)} 笔): 平均跌幅={np.mean(down_pcts):+.2f}%, "
              f"平均MACD面积={np.mean(down_areas):.1f}")

    # 5. 与自研对比
    print(f"\n[5] 与自研 ChanAnalyzer 对比:")
    analyzer = ChanAnalyzer(df)
    analyzer.analyze()
    our_bi = analyzer.bi_list

    print(f"    {'':>16s} | {'chan.py':>8s} | {'自研':>8s}")
    print(f"    {'-'*16}-+-{'-'*8}-+-{'-'*8}")
    print(f"    {'笔数':>14s} | {len(bi_list):>8d} | {len(our_bi):>8d}")
    print(f"    {'上升笔':>13s} | {len(up_bis):>8d} | {sum(1 for b in our_bi if b['direction']=='up'):>8d}")
    print(f"    {'下降笔':>13s} | {len(down_bis):>8d} | {sum(1 for b in our_bi if b['direction']=='down'):>8d}")

    # 逐笔对比: 按顺序展示 chan.py 和自研的每一笔
    # 方便手动检查差异在哪里
    max_len = max(len(bi_list), len(our_bi))
    print(f"\n    逐笔对比:")
    print(f"    {'#':>3} | {'chan.py':^35s} | {'自研':^35s}")
    print(f"    " + "-" * 78)
    for i in range(max_len):
        left = ''
        right = ''
        if i < len(bi_list):
            b = bi_list[i]
            d = 'UP' if b['direction'] == 'up' else 'DN'
            left = f"{d} {b['start_date'].strftime('%m-%d')}({b['start_price']:.0f})->{b['end_date'].strftime('%m-%d')}({b['end_price']:.0f})"
        if i < len(our_bi):
            b = our_bi[i]
            d = 'UP' if b['direction'] == 'up' else 'DN'
            sd = b.get('start_raw_date', b['start_date']).strftime('%m-%d')
            ed = b.get('end_raw_date', b['end_date']).strftime('%m-%d')
            right = f"{d} {sd}({b['start_price']:.0f})->{ed}({b['end_price']:.0f})"
        print(f"    {i+1:>3} | {left:<35s} | {right:<35s}")

    # 6. 可视化
    print(f"\n[6] 生成笔对比图表...")
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 14))

    # 上图: chan.py
    d2x = draw_chan_chart(ax1, df, cp, show_bi=True, show_seg=False,
                          show_zs=False, show_bsp=False, show_fractals=False)
    # 在笔的终点标注价格数值
    for bi in bi_list:
        c = '#e74c3c' if bi['direction'] == 'up' else '#27ae60'
        ex = d2x.get(bi['end_raw_date'])
        if ex is None:
            continue
        ax1.annotate(f'{bi["end_price"]:.0f}', (ex, bi['end_price']),
                     fontsize=7, color=c,
                     va='bottom' if bi['direction'] == 'up' else 'top', ha='center')
    ax1.set_title(f'chan.py | {STOCK_CODE} | 笔:{len(bi_list)}',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('价格')
    ax1.grid(True, alpha=0.3)

    # 下图: 自研
    d2x2 = ChanAnalyzer._draw_candlestick(ax2, df, width_ratio=0.6)
    for bi in our_bi:
        c = '#e74c3c' if bi['direction'] == 'up' else '#27ae60'
        sd = bi.get('start_raw_date', bi['start_date'])
        ed = bi.get('end_raw_date', bi['end_date'])
        sx = d2x2.get(sd)
        ex = d2x2.get(ed)
        if sx is None and hasattr(sd, 'date'):
            sx = d2x2.get(sd.date())
        if ex is None and hasattr(ed, 'date'):
            ex = d2x2.get(ed.date())
        if sx is None or ex is None:
            continue
        ax2.plot([sx, ex], [bi['start_price'], bi['end_price']],
                 color=c, linewidth=1.8, zorder=4)
        ax2.annotate(f'{bi["end_price"]:.0f}', (ex, bi['end_price']),
                     fontsize=7, color=c,
                     va='bottom' if bi['direction'] == 'up' else 'top', ha='center')
    ax2.set_title(f'ChanAnalyzer (自研) | {STOCK_CODE} | 笔:{len(our_bi)}',
                  fontsize=13, fontweight='bold')
    ax2.set_ylabel('价格')
    ax2.set_xlabel('日期')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/2-chan-笔识别对比.png', dpi=150, bbox_inches='tight')
    print(f"  图表已保存: outputs/2-chan-笔识别对比.png")
    plt.close()

    print("\n完成!")


if __name__ == '__main__':
    main()
