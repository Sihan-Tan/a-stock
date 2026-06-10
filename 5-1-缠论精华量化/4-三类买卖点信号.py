# -*- coding: utf-8 -*-
"""
第09讲：缠论精华量化
脚本4：三类买卖点信号

教学目标:
  - 掌握缠论三类买卖点的精确定义
  - 理解每类买点的实战价值和适用场景
  - 评估信号的后续收益表现

核心概念:
  三类买卖点是缠论最核心的实战输出，构成完整的交易闭环。
  它们是"缠论语言"的终极翻译——从K线图中解读出"什么时候该买/卖"。

缠论三类买点详解:

  第一类买点（一买）: 趋势背驰点
    定义: 下跌趋势中, 最后一段下跌的力度小于前一段
    判断: MACD面积比较法
      - b段 (前中枢到后中枢) 的MACD面积
      - c段 (后中枢之后) 的MACD面积
      - 如果c段面积 < b段面积 * 0.8, 说明"跌不动了"
    特点: 左侧交易, 风险最大但潜在收益最大
    适用: 长线价值投资者, 能承受较大回撤

  第二类买点（二买）: 回调确认点
    定义: 一买后首次回调不破一买低点
    特点: 右侧交易, 安全性高于一买
    适用: 稳健投资者, 等待底部确认后入场
    与一买的关系: 二买是对一买的"确认"

  第三类买点（三买）: 主升浪启动点 [最具实战价值]
    定义: 价格突破中枢后回踩不进入中枢
    条件: 回踩的最低点 > ZG (中枢上沿)
    特点: 趋势延续的标志, 类似传统分析中的"突破回踩确认"
    适用: 趋势交易者, 抓主升浪
    为什么三买最有价值:
      - 一买: 抄底, 可能抄在半山腰
      - 二买: 确认底部, 但可能盘整很久
      - 三买: 趋势已经启动, 回踩确认后加速

  第三类卖点（三卖）: 三买的镜像
    定义: 价格跌破中枢后反弹不进入中枢
    条件: 反弹的最高点 < ZD (中枢下沿)
    用途: 逃顶、做空

数据说明:
  本脚本使用贵州茅台 2023-2025 年更长周期的数据,
  以确保能检测到足够多的买卖点信号。
"""

import pandas as pd
import numpy as np
from data_loader import load_stock_data
from chan_analyzer import ChanAnalyzer

# ============================================================
# 参数配置
# ============================================================

# 使用更长时间区间以确保有足够的信号
STOCK_CODE = '600519.SH'
START_DATE = '2023-01-01'
END_DATE = '2025-12-31'


# ============================================================
# 主逻辑
# ============================================================

def main():
    print("=" * 60)
    print("第09讲 | 脚本4: 三类买卖点信号")
    print("=" * 60)

    # 1. 加载数据并分析
    print(f"\n[1] 加载 {STOCK_CODE} 日线数据 ({START_DATE} ~ {END_DATE})...")
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    print(f"    共 {len(df)} 根K线")

    analyzer = ChanAnalyzer(df)
    analyzer.analyze()

    # 2. 分析摘要
    print(f"\n[2] 缠论分析摘要:")
    analyzer.summary()

    # 3. 信号详情与后续收益
    if analyzer.signals:
        print(f"\n[3] 信号后续收益分析:")
        _analyze_signal_returns(df, analyzer.signals)
    else:
        print(f"\n[3] 未检测到任何买卖点信号")
        print("    可能原因: 数据区间内走势结构不典型")
        print("    建议: 尝试更长的时间区间或不同的股票")

    # 4. 三买信号重点分析
    # 三买被认为是最有实战价值的信号, 单独深入分析
    third_buys = [s for s in analyzer.signals if s['type'] == 'third_buy']
    if third_buys:
        print(f"\n[4] 三买信号重点分析 (共{len(third_buys)}个):")
        _analyze_third_buy_detail(df, third_buys)

    # 5. 可视化
    print(f"\n[5] 生成买卖点信号图表...")
    analyzer.plot(
        title=f'{STOCK_CODE} 三类买卖点信号',
        save_path='outputs/4-三类买卖点.png',
        show_bi=True,
        show_zhongshu=True,
        show_signals=True,   # 本课重点
        show_fractals=False,
    )

    print("\n完成!")


def _analyze_signal_returns(df, signals):
    """
    分析每个信号的后续 N 日收益

    评估逻辑:
      - 信号触发后 5/10/20 日的涨跌幅
      - 10日胜率: 10日后上涨的比例
      - 这对判断信号的实战价值至关重要

    为什么分析多个持有期:
      5日: 短线效果 (信号是否立刻有效)
      10日: 中线效果 (一般持仓周期)
      20日: 长线效果 (趋势是否延续)

    注意:
      这里用"收盘价"而非"盘中价",
      实际交易中可以用信号出现后的开盘价入场。
    """
    sig_names = {
        'first_buy': '一买', 'second_buy': '二买',
        'third_buy': '三买', 'third_sell': '三卖',
    }

    print(f"\n    {'日期':>12} | {'类型':>4} | {'价格':>8} | "
          f"{'5日':>7} | {'10日':>7} | {'20日':>7} | {'判定':>6}")
    print("    " + "-" * 75)

    win_count = 0
    total_count = 0

    for sig in signals:
        date = sig['date']
        try:
            idx = df.index.get_loc(date)
        except KeyError:
            continue

        # 计算 N 日后的收益
        returns = {}
        for n in [5, 10, 20]:
            if idx + n < len(df):
                future_price = float(df['close'].iloc[idx + n])
                ret = (future_price / sig['price'] - 1) * 100
                returns[n] = ret

        name = sig_names.get(sig['type'], sig['type'])

        r5 = f"{returns.get(5, 0):+6.2f}%" if 5 in returns else '   N/A'
        r10 = f"{returns.get(10, 0):+6.2f}%" if 10 in returns else '   N/A'
        r20 = f"{returns.get(20, 0):+6.2f}%" if 20 in returns else '   N/A'

        if 10 in returns:
            total_count += 1
            if returns[10] > 0:
                win_count += 1
                verdict = '盈利'
            else:
                verdict = '亏损'
        else:
            verdict = '待定'

        print(f"    {date.strftime('%Y-%m-%d'):>12} | {name:>4} | {sig['price']:>8.2f} | "
              f"{r5:>7} | {r10:>7} | {r20:>7} | {verdict:>6}")

    if total_count > 0:
        print(f"\n    10日胜率: {win_count}/{total_count} = {win_count/total_count*100:.1f}%")


def _analyze_third_buy_detail(df, third_buys):
    """
    三买信号的详细分析

    针对每个三买信号, 分析:
      - 距离中枢ZG的回踩幅度: 回踩越深越接近ZG说明越"弱"
      - 不同持有期的收益表现
      - 综合评级 (强势/有效/失败)

    评级标准:
      强势信号: 某个持有期收益 > 10%
      有效信号: 所有持有期收益 > 0%
      失败信号: 有亏损的持有期
    """
    for i, sig in enumerate(third_buys, 1):
        date = sig['date']
        zg = sig.get('zhongshu_zg', 0)
        zd = sig.get('zhongshu_zd', 0)

        try:
            idx = df.index.get_loc(date)
        except KeyError:
            continue

        print(f"\n    三买[{i}] {date.strftime('%Y-%m-%d')}:")
        print(f"      信号价格: {sig['price']:.2f}")
        print(f"      对应中枢: ZG={zg:.2f}, ZD={zd:.2f}")
        if zg > 0:
            distance = (sig['price'] / zg - 1) * 100
            print(f"      距ZG距离: {distance:+.2f}% (回踩幅度)")

        max_return = 0
        for n in [5, 10, 20, 40]:
            if idx + n < len(df):
                future_price = float(df['close'].iloc[idx + n])
                ret = (future_price / sig['price'] - 1) * 100
                max_return = max(max_return, ret)
                print(f"      {n}日后收益: {ret:+.2f}%")

        if max_return > 10:
            print(f"      评价: 强势信号")
        elif max_return > 0:
            print(f"      评价: 有效信号")
        else:
            print(f"      评价: 失败信号")


if __name__ == '__main__':
    main()
