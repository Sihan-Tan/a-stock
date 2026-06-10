# -*- coding: utf-8 -*-
"""
第09讲：缠论精华量化 (chan.py 版)
脚本1-chan：K线包含处理与分型识别

对比自研 ChanAnalyzer 与开源 chan.py 在K线合并和分型识别上的差异。

核心差异:
  1. chan.py 的每根合并K线 (klc) 自带 fx 属性, 直接标记分型类型
     而 ChanAnalyzer 需要遍历合并K线手动比较
  2. chan.py 的包含处理算法更成熟 (支持多级别、递归合并)
  3. 两者在大部分情况下结果一致, 但在边界情况下可能略有差异

学习目标:
  - 了解开源社区对缠论的实现情况
  - 验证自研实现的正确性
  - 掌握使用 chan.py 的基本方法
"""

import os
from data_loader import load_stock_data
from chan_analyzer import ChanAnalyzer
from chanpy_wrapper import run_chan, draw_chan_chart

STOCK_CODE = '600519.SH'
START_DATE = '2025-06-01'
END_DATE = '2025-12-31'


def main():
    print("=" * 60)
    print("第09讲 | 脚本1-chan: K线包含处理与分型识别 (chan.py版)")
    print("=" * 60)

    # 1. 加载数据
    print(f"\n[1] 加载 {STOCK_CODE} 日线数据 ({START_DATE} ~ {END_DATE})...")
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    print(f"    共 {len(df)} 根K线")

    # 2. chan.py 分析
    # run_chan 封装了 chan.py 的核心分析流程
    # 返回的字典中包含 klc_list(合并K线), fractals(分型) 等
    print("\n[2] chan.py 分析...")
    cp = run_chan(df, symbol=STOCK_CODE)
    klc_list = cp['klc_list']
    fractals = cp['fractals']

    print(f"    合并K线: {len(klc_list)} 根 (原始 {len(df)} 根)")
    print(f"    合并比例: {len(df) - len(klc_list)} 根 ({(len(df) - len(klc_list)) / len(df) * 100:.1f}%)")

    top_fx = [f for f in fractals if f['fx'] == 'top']
    bot_fx = [f for f in fractals if f['fx'] == 'bottom']
    print(f"\n    分型识别:")
    print(f"      顶分型: {len(top_fx)} 个")
    print(f"      底分型: {len(bot_fx)} 个")
    print(f"      总计:   {len(fractals)} 个")

    # 3. 自研 ChanAnalyzer 对比
    # 用同样的数据运行自研分析器, 对比结果
    print("\n[3] 自研 ChanAnalyzer 对比...")
    analyzer = ChanAnalyzer(df)
    analyzer.analyze()
    our_top = sum(1 for f in analyzer.fractals if f['type'] == 'top')
    our_bot = sum(1 for f in analyzer.fractals if f['type'] == 'bottom')

    # 并排对比表格
    print(f"    {'':>16s} | {'chan.py':>8s} | {'自研':>8s}")
    print(f"    {'-'*16}-+-{'-'*8}-+-{'-'*8}")
    print(f"    {'合并K线':>12s} | {len(klc_list):>8d} | {len(analyzer.merged_df):>8d}")
    print(f"    {'顶分型':>12s} | {len(top_fx):>8d} | {our_top:>8d}")
    print(f"    {'底分型':>12s} | {len(bot_fx):>8d} | {our_bot:>8d}")
    print(f"    {'分型总计':>11s} | {len(fractals):>8d} | {len(analyzer.fractals):>8d}")

    # 4. 展示合并K线中分型分布
    print(f"\n[4] chan.py 分型样例 (前10个):")
    for f in fractals[:10]:
        ft = '顶' if f['fx'] == 'top' else '底'
        price = f['high'] if f['fx'] == 'top' else f['low']
        print(f"      {f['date'].strftime('%Y-%m-%d')} | {ft}分型 | "
              f"高={f['high']:.2f} 低={f['low']:.2f} | 价格={price:.2f}")

    # 5. 可视化: 上下对比两张图
    # 上图: chan.py 的分型标记, 下图: 自研的分型标记
    print(f"\n[5] 生成对比图表...")
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(22, 14))

    # 上图: chan.py 分型
    draw_chan_chart(ax1, df, cp, show_bi=False, show_seg=False,
                   show_zs=False, show_bsp=False, show_fractals=True)
    ax1.set_title(f'chan.py 分型识别 | 合并{len(klc_list)}根 | '
                  f'顶{len(top_fx)} 底{len(bot_fx)} 共{len(fractals)}个',
                  fontsize=13, fontweight='bold')
    ax1.set_ylabel('价格')
    ax1.grid(True, alpha=0.3)

    # 下图: 自研分型
    d2x = ChanAnalyzer._draw_candlestick(ax2, df, width_ratio=0.6)
    for f in analyzer.fractals:
        rd = f.get('raw_date', f['date'])
        x = d2x.get(rd)
        if x is None and hasattr(rd, 'date'):
            x = d2x.get(rd.date())
        if x is None:
            continue
        m = 'v' if f['type'] == 'top' else '^'
        c = '#e74c3c' if f['type'] == 'top' else '#2ecc71'
        ax2.scatter(x, f['price'], marker=m, color=c, s=60, zorder=5, alpha=0.6)
    ax2.set_title(f'ChanAnalyzer 分型识别 | 合并{len(analyzer.merged_df)}根 | '
                  f'顶{our_top} 底{our_bot} 共{len(analyzer.fractals)}个',
                  fontsize=13, fontweight='bold')
    ax2.set_ylabel('价格')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(h_pad=3)
    os.makedirs('outputs', exist_ok=True)
    plt.savefig('outputs/1-chan-分型识别对比.png', dpi=150, bbox_inches='tight')
    print(f"  图表已保存: outputs/1-chan-分型识别对比.png")
    plt.close()

    print("\n完成!")


if __name__ == '__main__':
    main()
