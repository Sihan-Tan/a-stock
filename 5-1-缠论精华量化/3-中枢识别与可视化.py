# -*- coding: utf-8 -*-
"""
第09讲：缠论精华量化
脚本3：中枢识别与可视化

教学目标:
  - 理解"中枢"的定义：至少三段笔构成的价格重叠区间
  - 掌握 ZG（中枢高点）和 ZD（中枢低点）的计算方法
  - 判断中枢的移动方向（上移/下移/震荡）

核心概念:
  如果分型是"字母"、笔是"词语"，那么中枢就是"句子"。
  中枢代表市场的平衡区域，是多空力量博弈的战场。

缠论中枢的精确定义:
  中枢 = 至少连续3笔的价格区间重叠部分

  计算方式:
    - 取至少3笔, 找出每笔的最高价和最低价
    - ZG (中枢高/天花板) = min(所有笔的最高价)
      - 为什么取最小值? 因为中枢的上沿由"最弱的多头"决定
        价格突破ZG意味着所有空头都被打穿了
    - ZD (中枢低/地板) = max(所有笔的最低价)
      - 为什么取最大值? 因为中枢的下沿由"最弱的空头"决定
        价格跌破ZD意味着所有多头都被打穿了
    - 有效条件: ZG > ZD (确实存在价格重叠)
    - 如果 ZG <= ZD: 说明几笔的区间不重叠, 不构成中枢

中枢方向判断 (缠论趋势分类):
  - 中枢上移: 新中枢的 ZG 和 ZD 都高于前一个 → 上升趋势
    每次价格整理后, 重心都在上移 → 牛市特征
  - 中枢下移: 新中枢的 ZG 和 ZD 都低于前一个 → 下跌趋势
    每次价格整理后, 重心都在下移 → 熊市特征
  - 中枢重叠: 新旧中枢有价格交叉 → 震荡走势
    大级别震荡, 没有明确的趋势方向

中枢的实战意义:
  - 作为支撑和阻力: ZG 是压力位, ZD 是支撑位
  - 突破确认: 价格站稳 ZG 上方 = 有效突破
  - 止损参考: 三买入场后以 ZG 为止损位

注意: 本脚本使用了一个与之前不同的股票 300570.SZ (太辰光),
因为该股在测试区间内中枢结构更典型、更丰富。
"""

import numpy as np
from data_loader import load_stock_data
from chan_analyzer import ChanAnalyzer

# ============================================================
# 参数配置
# ============================================================

# 300570.SZ 在 2025-2026 年期间中枢结构显著
STOCK_CODE = '300570.SZ'
START_DATE = '2025-06-01'
END_DATE = '2026-03-31'


# ============================================================
# 主逻辑
# ============================================================

def main():
    print("=" * 60)
    print("第09讲 | 脚本3: 中枢识别与可视化")
    print("=" * 60)

    # 1. 加载数据并分析
    print(f"\n[1] 加载 {STOCK_CODE} 日线数据 ({START_DATE} ~ {END_DATE})...")
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    print(f"    共 {len(df)} 根K线")

    analyzer = ChanAnalyzer(df)
    analyzer.analyze()

    # 2. 中枢基本信息
    # 中枢的个数和分布反映了走势的"节奏":
    #   一个中枢都没有 → 单边趋势, 没有整理
    #   多个中枢 → 趋势+盘整交替
    print(f"\n[2] 中枢识别结果:")
    print(f"    笔数:   {len(analyzer.bi_list)}")
    print(f"    中枢数: {len(analyzer.zhongshu_list)}")

    if not analyzer.zhongshu_list:
        print("    未识别到中枢，可能数据量不足或区间太短")
        return

    # 3. 中枢详情
    # 显示每个中枢: 起止时间、ZG/ZD、中心价位、幅度、包含笔数
    # 中枢幅度 = (ZG-ZD)/center, 反映了盘整区的宽度
    #   幅度小 → 多空平衡, 蓄势待突破
    #   幅度大 → 多空分歧大, 尚未达成平衡
    print(f"\n[3] 中枢列表:")
    print(f"    {'序号':>4} | {'起始':>12} | {'结束':>12} | "
          f"{'ZG':>8} | {'ZD':>8} | {'中心':>8} | {'幅度%':>6} | {'笔数':>4}")
    print("    " + "-" * 80)

    for i, zs in enumerate(analyzer.zhongshu_list, 1):
        amplitude = (zs['ZG'] - zs['ZD']) / zs['center'] * 100
        print(f"    {i:>4} | {zs['start_date'].strftime('%Y-%m-%d'):>12} | "
              f"{zs['end_date'].strftime('%Y-%m-%d'):>12} | "
              f"{zs['ZG']:>8.2f} | {zs['ZD']:>8.2f} | {zs['center']:>8.2f} | "
              f"{amplitude:>5.1f}% | {zs['bi_count']:>4}")

    # 4. 中枢方向分析
    # 判断中枢移动方向: 上移 → 趋势向上, 下移 → 趋势向下
    print(f"\n[4] 中枢方向分析:")
    _analyze_zhongshu_trend(analyzer.zhongshu_list)

    # 5. 中枢统计
    # 汇总中枢的幅度和长度统计
    print(f"\n[5] 中枢统计:")
    amplitudes = [(zs['ZG'] - zs['ZD']) / zs['center'] * 100 for zs in analyzer.zhongshu_list]
    bi_counts = [zs['bi_count'] for zs in analyzer.zhongshu_list]

    print(f"    平均中枢幅度: {np.mean(amplitudes):.2f}%")
    print(f"    最大中枢幅度: {np.max(amplitudes):.2f}%")
    print(f"    最小中枢幅度: {np.min(amplitudes):.2f}%")
    print(f"    平均包含笔数: {np.mean(bi_counts):.1f}")
    print(f"    最大包含笔数: {np.max(bi_counts)}")

    # 6. 可视化: 在K线图上绘制中枢
    # 中枢用蓝色透明方框表示, 标注 ZG 和 ZD
    print(f"\n[6] 生成中枢可视化图表...")
    analyzer.plot(
        title=f'{STOCK_CODE} 中枢识别与可视化',
        save_path='outputs/3-中枢识别.png',
        show_bi=True,
        show_zhongshu=True,   # 本课重点
        show_signals=False,
        show_fractals=False,  # 分型已标记在笔上, 不必单独显示
    )

    print("\n完成!")


def _analyze_zhongshu_trend(zhongshu_list):
    """
    分析中枢之间的移动方向 (缠论趋势分类)

    如果有 >=2 个中枢, 可以判断趋势方向:
      - 上移: ZG 和 ZD 都抬高 → 上升趋势 (多头主导)
      - 下移: ZG 和 ZD 都降低 → 下跌趋势 (空头主导)
      - 重叠: 否则 → 震荡 (无序)

    注意: 这个分类与传统的 E/(E)M A 均线方向分类不同,
    它是基于结构而非价格的, 理论上更稳定但信号更少。
    """
    if len(zhongshu_list) < 2:
        print("    只有1个中枢，无法判断趋势方向")
        return

    for i in range(1, len(zhongshu_list)):
        prev = zhongshu_list[i - 1]
        curr = zhongshu_list[i]

        if curr['ZG'] > prev['ZG'] and curr['ZD'] > prev['ZD']:
            trend = '上移 (上升趋势)'
        elif curr['ZG'] < prev['ZG'] and curr['ZD'] < prev['ZD']:
            trend = '下移 (下跌趋势)'
        else:
            trend = '重叠 (震荡走势)'

        zg_change = curr['ZG'] - prev['ZG']
        zd_change = curr['ZD'] - prev['ZD']

        print(f"    中枢{i} -> 中枢{i+1}: {trend}")
        print(f"      ZG变化: {zg_change:+.2f}  ZD变化: {zd_change:+.2f}")


if __name__ == '__main__':
    main()
