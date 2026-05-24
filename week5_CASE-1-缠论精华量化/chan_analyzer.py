# -*- coding: utf-8 -*-
"""
缠论分析引擎 - 核心模块

本模块实现了缠论 (Chan Theory) 的完整分析流水线:
  原始K线 → K线包含处理(合并) → 顶底分型识别 → 笔的识别 → 中枢识别 → 买卖点信号检测

缠论的层级结构 (从微观到宏观):
  1. K线 (蜡烛)    - 市场的最小记录单位
  2. 合并K线        - 处理包含关系后的K线, 反映真实的趋势方向
  3. 分型           - 顶分型/底分型, 是市场的"字母"
  4. 笔             - 连接相邻顶底分型, 是市场的"词语"
  5. 中枢           - 多笔重叠区域, 是市场的"句子"
  6. 买卖点         - 三类买卖点, 是整个分析的实战输出

使用示例:
  from chan_analyzer import ChanAnalyzer

  analyzer = ChanAnalyzer(df)
  analyzer.analyze()          # 一键执行完整分析
  analyzer.summary()          # 打印分析摘要
  signal_df = analyzer.get_signal_df()  # 获取信号DataFrame, 用于回测
  analyzer.plot(title='缠论分析')       # 可视化

依赖:
  - pandas, numpy: 数据处理
  - TA-Lib: MACD 计算 (用于背驰判断)
"""

import pandas as pd
import numpy as np
import talib
import os


class ChanAnalyzer:
    """
    缠论分析器

    这是整个项目的核心类, 将原始 K 线数据经过
    合并 -> 分型 -> 笔 -> 中枢 -> 信号的完整流程,
    输出可用于策略回测的买卖点信号。

    设计理念:
      - 每个分析步骤对应一个私有方法, 职责单一
      - analyze() 作为总入口, 按顺序调用各步骤
      - 中间结果存储在实例变量中, 方便调试和可视化
    """

    def __init__(self, df):
        """
        参数:
            df: DataFrame, 索引为日期 (DatetimeIndex), 需包含 open/high/low/close/volume 列
        """
        self.raw_df = df.copy()           # 原始K线数据 (保存一份副本, 不修改原始数据)
        self.merged_df = None             # 包含处理后的K线数据
        self.fractals = []                 # 所有分型列表 (包含被笔过滤掉的)
        self.confirmed_fractals = []       # 参与成笔的确认分型列表
        self.bi_list = []                  # 笔列表 (走势的最小单位)
        self.zhongshu_list = []            # 中枢列表 (价格重叠区域)
        self.signals = []                  # 买卖点信号列表
        self._macd_hist = None             # MACD 柱状图值 (内部缓存, 用于背驰判断)

    def analyze(self):
        """
        执行完整缠论分析: 合并K线 → 分型 → 笔 → 中枢 → 信号

        这是总入口方法, 按顺序调用五个核心步骤。
        每个步骤的输出是下一步的输入:
          merged_df → fractals → bi_list → zhongshu_list → signals

        返回 self 以支持链式调用: analyzer = ChanAnalyzer(df).analyze()
        """
        self._prepare_macd()              # 预计算 MACD (信号检测需要)
        self.merged_df = self._merge_klines()             # Step 1: K线包含处理
        self.fractals = self._identify_fractals()          # Step 2: 分型识别
        self.bi_list = self._identify_bi()                 # Step 3: 笔识别
        self.zhongshu_list = self._identify_zhongshu()     # Step 4: 中枢识别
        self.signals = self._detect_signals()               # Step 5: 信号检测
        return self

    # ============================================================
    # Step 1: K线包含关系处理
    # ============================================================
    #
    # 缠论原文对包含关系的定义:
    #   包含关系: 一根K线的高低点完全"包含"在相邻K线的范围内
    #     即: 当前K线的最高点 >= 前K线最高点 且 最低点 <= 前K线最低点 (向上包含)
    #     或: 前K线的最高点 >= 当前K线最高点 且 最低点 <= 当前K线最低点 (向下包含)
    #
    # 合并规则:
    #   - 在上升趋势中 (前K线高点向上): 取较高的高点和较高的低点 (顺势取高)
    #     这样合并后的K线延续上升方向
    #   - 在下降趋势中 (前K线高点向下): 取较低的高点和较低的低点 (顺势取低)
    #     这样合并后的K线延续下降方向
    #
    # 为什么需要包含处理:
    #   如果不合并包含关系的K线, 会产生大量"伪分型"。
    #   例如震荡行情中, 小实体K线被大实体包含, 但直接看会形成假突破信号。
    #   包含处理后, K线数量减少, 但趋势结构更清晰、分型更可靠。
    #
    # 额外的设计: high_date / low_date
    #   记录该合并K线的最高点和最低点分别来自哪根原始K线。
    #   这是为了在图表上精确标记分型和笔端点时, 能定位到实际的蜡烛图位置,
    #   而不是合并K线的"虚拟"位置。
    # ============================================================

    def _merge_klines(self):
        """
        处理K线包含关系: 将存在包含关系的相邻K线合并为一根

        算法: 逐根K线扫描, 维护一个合并列表 merged
          - 如果当前K线与列表中最后一根存在包含关系 → 合并之
          - 否则 → 追加为新的K线

        合并方向的判断依据: 列表中最新的两根K线的高点比较
          - 前K线高点 >= 前前K线高点 → 上升趋势 → 向上合并
          - 前K线高点 < 前前K线高点  → 下降趋势 → 向下合并

        返回:
            DataFrame: 合并后的K线, 索引为日期, 包含 open/high/low/close/volume
                      以及 high_date/low_date 列
        """
        df = self.raw_df
        if len(df) < 3:
            # 数据太少, 无需合并
            result = df.copy()
            result['high_date'] = result.index
            result['low_date'] = result.index
            return result

        merged = []  # 合并后的K线列表 (元素为dict)
        for i in range(len(df)):
            # 将当前K线转为字典格式
            row = {
                'date': df.index[i],
                'open': float(df['open'].iloc[i]),
                'high': float(df['high'].iloc[i]),
                'low': float(df['low'].iloc[i]),
                'close': float(df['close'].iloc[i]),
                'volume': float(df['volume'].iloc[i]),
                'high_date': df.index[i],  # 该合并K线的最高点来自哪根原始K线
                'low_date': df.index[i],   # 该合并K线的最低点来自哪根原始K线
            }

            # 前两根K线直接加入, 需要至少2根才能判断趋势方向
            if len(merged) < 2:
                merged.append(row)
                continue

            prev = merged[-1]

            # 判断是否为包含关系:
            # 条件: 两者中一根的 high >= 另一根的 high 且 low <= 另一根的 low
            inclusion = (
                (row['high'] >= prev['high'] and row['low'] <= prev['low']) or
                (prev['high'] >= row['high'] and prev['low'] <= row['low'])
            )

            if inclusion:
                prev_prev = merged[-2]
                # 判断趋势方向: 比较前两根K线的高点
                is_up = prev['high'] >= prev_prev['high']

                if is_up:
                    # 向上合并: 取更高的高点和更高的低点
                    new_high = max(prev['high'], row['high'])
                    new_low = max(prev['low'], row['low'])
                    # 最高点来自两者中更高的那个
                    h_date = row['high_date'] if row['high'] >= prev['high'] else prev['high_date']
                    # 最低点来自两者中更高的那个 (上升趋势中低点也在抬升)
                    l_date = row['low_date'] if row['low'] >= prev['low'] else prev['low_date']
                    merged[-1] = {
                        'date': prev['date'],
                        'open': prev['open'],
                        'high': new_high,
                        'low': new_low,
                        'close': row['close'],        # 使用新K线的收盘价
                        'volume': prev['volume'] + row['volume'],  # 成交量累加
                        'high_date': h_date,
                        'low_date': l_date,
                    }
                else:
                    # 向下合并: 取更低的高点和更低的低点
                    new_high = min(prev['high'], row['high'])
                    new_low = min(prev['low'], row['low'])
                    h_date = row['high_date'] if row['high'] <= prev['high'] else prev['high_date']
                    l_date = row['low_date'] if row['low'] <= prev['low'] else prev['low_date']
                    merged[-1] = {
                        'date': prev['date'],
                        'open': prev['open'],
                        'high': new_high,
                        'low': new_low,
                        'close': row['close'],
                        'volume': prev['volume'] + row['volume'],
                        'high_date': h_date,
                        'low_date': l_date,
                    }
            else:
                # 没有包含关系, 直接追加
                merged.append(row)

        result = pd.DataFrame(merged)
        if not result.empty:
            result['date'] = pd.to_datetime(result['date'])
            result['high_date'] = pd.to_datetime(result['high_date'])
            result['low_date'] = pd.to_datetime(result['low_date'])
            result.set_index('date', inplace=True)
        return result

    # ============================================================
    # Step 2: 分型识别
    # ============================================================
    #
    # 分型是缠论最基础的"结构单元", 分为顶分型和底分型:
    #
    #   顶分型 (Top Fractal): 中间K线的高点是三根中最高的, 低点也是三根中最高的
    #     意味着多方力量衰竭, 可能由涨转跌
    #
    #   底分型 (Bottom Fractal): 中间K线的低点是三根中最低的, 高点也是三根中最低的
    #     意味着空方力量衰竭, 可能由跌转涨
    #
    #  图示:
    #    顶分型:   /\         底分型:      /\
    #            /  \                    /    \
    #           /    \                  /      \/
    #
    # 注意: 分型识别必须在合并后的K线上进行, 否则会有大量噪音。
    # 另外, 这里的分型是"候选分型", 后面笔的识别会进一步过滤。
    # ============================================================

    def _identify_fractals(self):
        """
        在合并后的 K 线上识别顶分型和底分型

        遍历每根合并 K 线 (排除首尾各一根), 检查它和前后两根K线的关系。

        顶分型条件:
          当前K线高点 > 前K线高点 AND 当前K线高点 > 后K线高点
          AND 当前K线低点 > 前K线低点 AND 当前K线低点 > 后K线低点

        底分型条件:
          当前K线低点 < 前K线低点 AND 当前K线低点 < 后K线低点
          AND 当前K线高点 < 前K线高点 AND 当前K线高点 < 后K线高点

        返回:
            list[dict]: 分型列表, 每个分型包含:
              - index: 在 merged_df 中的位置
              - date: 合并K线的日期 (用于算法内部的索引计算)
              - raw_date: 极值实际出现的原始K线日期 (用于图表精确标记)
              - type: 'top' 或 'bottom'
              - price: 顶分型取 high, 底分型取 low
        """
        df = self.merged_df
        if df is None or len(df) < 3:
            return []

        has_raw_dates = 'high_date' in df.columns and 'low_date' in df.columns

        fractals = []
        for i in range(1, len(df) - 1):
            h_prev = df['high'].iloc[i - 1]
            h_curr = df['high'].iloc[i]
            h_next = df['high'].iloc[i + 1]
            l_prev = df['low'].iloc[i - 1]
            l_curr = df['low'].iloc[i]
            l_next = df['low'].iloc[i + 1]

            # 顶分型: 中间最高, 且低点也最高
            if (h_curr > h_prev and h_curr > h_next and
                    l_curr > l_prev and l_curr > l_next):
                # 用 high_date 定位精确的原始K线日期
                raw_date = df['high_date'].iloc[i] if has_raw_dates else df.index[i]
                fractals.append({
                    'index': i,
                    'date': df.index[i],
                    'raw_date': raw_date,
                    'type': 'top',
                    'price': float(h_curr),
                })
            # 底分型: 中间最低, 且高点也最低
            elif (l_curr < l_prev and l_curr < l_next and
                  h_curr < h_prev and h_curr < h_next):
                raw_date = df['low_date'].iloc[i] if has_raw_dates else df.index[i]
                fractals.append({
                    'index': i,
                    'date': df.index[i],
                    'raw_date': raw_date,
                    'type': 'bottom',
                    'price': float(l_curr),
                })

        return fractals

    # ============================================================
    # Step 3: 笔识别
    # ============================================================
    #
    # 笔的定义: 连接相邻顶分型和底分型的最小走势单位
    #   - 上升笔: 从底分型到顶分型 (价格上涨)
    #   - 下降笔: 从顶分型到底分型 (价格下跌)
    #
    # 笔的三个约束条件:
    #   1. 顶底分型必须交替出现 (不能连续两个顶或两个底)
    #   2. 相邻分型之间至少间隔 min_gap 根合并K线 (默认4)
    #      缠论原文要求"至少5根K线含分型端点", 即 gap >= 4
    #   3. 相同类型的分型只保留极值 (更高的顶、更低的底)
    #
    # 算法: 贪心前向扫描 (Greedy Forward Scan)
    #   - 维护一个"已确认"的分型列表 confirmed
    #   - 遍历候选分型列表 fractals:
    #     - 同类型: 如果更极端则替换最后一个 (取最高顶/最低底)
    #     - 异类型且间距足够: 确认成笔, 追加到列表
    #     - 异类型但间距不够: 跳过, 不破坏已确认的结构
    #
    # 为什么用贪心算法:
    #   笔识别本质上是一个在约束条件下寻找最优匹配的问题。
    #   贪心算法虽然简单, 但在实践中效果已经足够好。
    #   更复杂的算法 (如动态规划) 可以找到全局最优但实现更复杂。
    # ============================================================

    def _identify_bi(self, min_gap=4):
        """
        基于分型生成笔

        参数:
            min_gap: 相邻分型之间的最小索引差 (默认4, 对应缠论"至少5根K线含端点")
                     更大的值会产生更少的笔、更稳定的结构, 但可能遗漏短线机会

        返回:
            list[dict]: 笔列表, 每笔包含:
              - start_index/end_index: 在 merged_df 中的起止位置
              - start_date/end_date: 起止日期 (合并K线)
              - start_raw_date/end_raw_date: 精确的原始K线日期
              - start_price/end_price: 起止价格
              - direction: 'up' 或 'down'
        """
        if len(self.fractals) < 2:
            return []

        # 贪心前向扫描: 从第一个分型开始, 逐步确认成笔
        confirmed = [self.fractals[0]]

        for f in self.fractals[1:]:
            last = confirmed[-1]

            if f['type'] == last['type']:
                # 同类型分型: 保留更极端的 (更高的顶, 更低的底)
                if ((f['type'] == 'top' and f['price'] > last['price']) or
                        (f['type'] == 'bottom' and f['price'] < last['price'])):
                    confirmed[-1] = f
            else:
                # 异类型分型: 检查间距是否满足最低K线数要求
                gap = f['index'] - last['index']
                if gap >= min_gap:
                    confirmed.append(f)

        self.confirmed_fractals = list(confirmed)

        # 从确认分型生成笔
        # 一对相邻的顶底分型构成一笔
        bi_list = []
        for i in range(1, len(confirmed)):
            prev_f = confirmed[i - 1]
            curr_f = confirmed[i]

            # 跳过同类型相邻 (理论上不该发生, 但防御性编程)
            if prev_f['type'] == curr_f['type']:
                continue

            # 方向: 从底到顶是上升, 从顶到底是下降
            direction = 'up' if prev_f['type'] == 'bottom' else 'down'

            # 使用精确的原始日期 (用于图表标记时定位到具体蜡烛)
            start_raw = prev_f.get('raw_date', prev_f['date'])
            end_raw = curr_f.get('raw_date', curr_f['date'])

            bi_list.append({
                'start_index': prev_f['index'],
                'end_index': curr_f['index'],
                'start_date': prev_f['date'],
                'end_date': curr_f['date'],
                'start_raw_date': start_raw,
                'end_raw_date': end_raw,
                'start_price': prev_f['price'],
                'end_price': curr_f['price'],
                'direction': direction,
            })

        return bi_list

    # ============================================================
    # Step 4: 中枢识别
    # ============================================================
    #
    # 中枢的定义:
    #   连续至少3笔的价格区间存在重叠区域。
    #   这个重叠区域就是多空双方"拉锯"的战场:
    #     - 多方想推高价格, 空方想压低价格
    #     - 双方反复争夺形成了价格重叠区
    #
    # 中枢的两个关键价位:
    #   ZG (中枢高): 所有参与中枢的笔中, 最高价的"最小值"
    #     可以理解为"天花板", 价格突破ZG意味着多方胜利
    #   ZD (中枢低): 所有参与中枢的笔中, 最低价的"最大值"
    #     可以理解为"地板", 价格跌破ZD意味着空方胜利
    #
    #  有效条件: ZG > ZD (确实存在价格重叠区间)
    #   如果 ZG <= ZD, 说明笔的区间虽然挨着但不重叠, 不构成中枢
    #
    # 扩展机制:
    #   中枢形成后, 后续的笔如果继续与中枢区间重叠,
    #   则中枢可以扩展 (包含更多笔)。
    #   但为了防止中枢过长吞噬所有笔导致无信号,
    #   设置了 max_extend 参数限制扩展次数。
    #
    # 图示:
    #   笔1:  ----\
    #   笔2:        \----/
    #   笔3:        /----\
    #   笔4:       /   ZG\
    #               ZD   \
    #   重叠区域: [ZD, ZG] (蓝色方框)
    # ============================================================

    def _identify_zhongshu(self, min_bi=3, max_extend=4):
        """
        识别中枢 (价格重叠区间)

        参数:
            min_bi:     形成中枢的最少笔数 (默认3, 缠论标准)
            max_extend: 中枢形成后最多再扩展的笔数 (默认4)
                        限制扩展防止中枢过长

        返回:
            list[dict]: 中枢列表, 每个中枢包含:
              - ZG: 中枢高点 (各笔最高价的最小值)
              - ZD: 中枢低点 (各笔最低价的最大值)
              - center: 中枢中心价位 (ZG+ZD)/2
              - start_date/end_date: 起止日期
              - bi_count: 包含的笔数
        """
        if len(self.bi_list) < min_bi:
            return []

        zhongshu_list = []
        i = 0

        # 滑动窗口遍历: 从第 i 笔开始取 min_bi 笔检查重叠
        while i <= len(self.bi_list) - min_bi:
            group = self.bi_list[i:i + min_bi]
            # 每笔的高点 = max(起点价格, 终点价格)
            highs = [max(b['start_price'], b['end_price']) for b in group]
            # 每笔的低点 = min(起点价格, 终点价格)
            lows = [min(b['start_price'], b['end_price']) for b in group]

            zg = min(highs)  # 重叠区间的上沿
            zd = max(lows)   # 重叠区间的下沿

            if zg > zd:
                # 有效中枢! 尝试扩展
                end = i + min_bi
                extend_count = 0
                while end < len(self.bi_list) and extend_count < max_extend:
                    nb = self.bi_list[end]
                    nh = max(nb['start_price'], nb['end_price'])
                    nl = min(nb['start_price'], nb['end_price'])
                    # 如果新笔与中枢区间有重叠 → 扩展中枢
                    if nh > zd and nl < zg:
                        end += 1
                        extend_count += 1
                    else:
                        break

                zhongshu_list.append({
                    'ZG': zg,
                    'ZD': zd,
                    'center': (zg + zd) / 2,
                    'start_index': group[0]['start_index'],
                    'end_index': self.bi_list[end - 1]['end_index'],
                    'start_date': group[0]['start_date'],
                    'end_date': self.bi_list[end - 1]['end_date'],
                    'bi_count': end - i,
                })
                i = end  # 跳过已经纳入中枢的笔
            else:
                i += 1

        return zhongshu_list

    # ============================================================
    # Step 5: 信号检测
    # ============================================================
    #
    # 缠论的三类买卖点是整个分析体系的实战输出:
    #
    #   第一类买点 (First Buy) - 趋势背驰点:
    #     下跌趋势中, 最后一段下跌力度弱于前一段 (MACD面积减小)
    #     → "跌不动了", 反转在即
    #     风险最高但收益空间最大 (抄底)
    #
    #   第二类买点 (Second Buy) - 回调确认点:
    #     一买后价格上涨, 然后回调但不破一买低点
    #     → "确认底部", 安全性高于一买
    #
    #   第三类买点 (Third Buy) - 中枢突破点 [最实战价值]:
    #     价格突破中枢后回踩, 但不跌回中枢内部
    #     → "强势确认", 新趋势启动
    #     这是缠论中最具实战价值的信号
    #
    #   第三类卖点 (Third Sell) - 三买的镜像:
    #     价格跌破中枢后反弹, 但不进入中枢内部
    #     → 下跌趋势确认
    #
    # 背驰判断:
    #   使用 MACD 柱状图的面积来衡量一段走势的"力度"。
    #   面积越小 → 力度越弱 → 越可能发生转折。
    #   这是将 MACD 传统用法 (顶底背离) 做了量化改进:
    #   不用看金叉死叉, 用总面积对比更精确。
    # ============================================================

    def _prepare_macd(self):
        """预计算 MACD 柱状图, 用于背驰判断

        使用 TA-Lib 的标准参数 (12, 26, 9), 这是广泛使用的默认值。
        对于日线级别, 12日和26日EMA分别对应约两周和一个月。

        数据点少于35时, MACD 计算会产生 NaN, 用零填充。
        """
        close = self.raw_df['close'].values.astype(float)
        if len(close) >= 35:
            _, _, self._macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        else:
            self._macd_hist = np.zeros(len(close))

    def _calc_macd_area(self, start_date, end_date):
        """
        计算指定日期区间内 MACD 柱状图的绝对面积

        为什么用绝对面积:
          MACD 柱有正有负, 直接求和会正负抵消。
          绝对面积反映了走势的"总动能", 不论方向。

        参数:
            start_date: 起始日期
            end_date:   结束日期

        返回:
            float: MACD 绝对面积 (越大表示力度越强)
        """
        mask = (self.raw_df.index >= start_date) & (self.raw_df.index <= end_date)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            return 0.0
        segment = self._macd_hist[indices]
        return float(np.nansum(np.abs(segment)))

    def _detect_signals(self):
        """
        检测所有类型的买卖点信号

        检测顺序:
          1. 三买 (最有价值, 优先检测)
          2. 一买 (需要中枢结构完整)
          3. 二买 (依赖一买结果)
          4. 三卖 (对称检查)

        所有信号按日期排序后返回。
        """
        signals = []
        signals.extend(self._detect_third_buy())
        first_buys = self._detect_first_buy()
        signals.extend(first_buys)
        signals.extend(self._detect_second_buy(first_buys))
        signals.extend(self._detect_third_sell())
        signals.sort(key=lambda s: s['date'])
        return signals

    def _detect_third_buy(self):
        """
        第三类买点: 突破中枢后回踩不进入中枢

        三买是缠论中最有实战价值的信号, 它的逻辑是:
          1. 市场形成一个中枢 (多空拉锯区)
          2. 价格强势突破中枢上沿 (ZG)
          3. 突破后回踩但不跌回中枢
          4. 说明突破是"真突破", 趋势将继续

        实现流程:
          state = 'WAIT_BREAKOUT'
            ↓ 遇到上涨笔突破 ZG
          state = 'WAIT_PULLBACK'
            ↓ 遇到下跌笔回踩
          如果回踩低点 > ZG → 三买!
          如果回踩低点 <= ZG → 假突破, 放弃

        参数:
            无 (使用 self.zhongshu_list 和 self.bi_list)

        返回:
            list[dict]: 三买信号列表
        """
        signals = []
        used_dates = set()  # 避免同一日期重复信号

        for zs in self.zhongshu_list:
            zg = zs['ZG']
            zd = zs['ZD']

            # 只考虑中枢完成之后的笔
            post_bis = [b for b in self.bi_list if b['start_index'] >= zs['end_index']]
            state = 'WAIT_BREAKOUT'

            for bi in post_bis:
                if state == 'WAIT_BREAKOUT':
                    # 等待突破: 上涨笔的终点 > ZG
                    if bi['direction'] == 'up' and bi['end_price'] > zg:
                        state = 'WAIT_PULLBACK'
                elif state == 'WAIT_PULLBACK':
                    if bi['direction'] == 'down':
                        pullback_low = bi['end_price']
                        sig_date = bi.get('end_raw_date', bi['end_date'])
                        if pullback_low > zg and sig_date not in used_dates:
                            signals.append({
                                'date': sig_date,
                                'type': 'third_buy',
                                'price': pullback_low,
                                'zhongshu_zg': zg,
                                'zhongshu_zd': zd,
                            })
                            used_dates.add(sig_date)
                            break
                        else:
                            # 回踩跌回中枢 → 假突破
                            break
                    elif bi['direction'] == 'up':
                        # 继续上涨, 不构成回踩, 维持等待状态
                        pass

        return signals

    def _detect_first_buy(self):
        """
        第一类买点: 下跌趋势中的背驰

        一买的逻辑基础:
          在下跌趋势中, 空方力量会逐渐衰竭。
          如果有两个中枢依次下移 (说明是下跌趋势),
          且最后一段下跌 (c段) 的力度小于前一段 (b段),
          那么趋势可能反转。

        判断条件:
          1. 至少2个中枢, 且后中枢的 ZG/ZD 都低于前中枢 (趋势下移)
          2. b段 (前中枢结束 → 后中枢开始) 的 MACD 面积
          3. c段 (后中枢结束 → 下跌笔结束) 的 MACD 面积
          4. c段面积 < b段面积 * 0.8 (背驰: 力度衰减 >= 20%)

        返回:
            list[dict]: 一买信号列表, 包含背驰比
        """
        signals = []

        if len(self.zhongshu_list) < 2:
            return signals

        for j in range(1, len(self.zhongshu_list)):
            prev_zs = self.zhongshu_list[j - 1]
            curr_zs = self.zhongshu_list[j]

            # 检查中枢下移: 当前中枢的 ZD 和 ZG 都低于前一个
            if not (curr_zs['ZD'] < prev_zs['ZD'] and curr_zs['ZG'] < prev_zs['ZG']):
                continue

            # b段: 从前中枢结束到后中枢开始
            b_area = self._calc_macd_area(prev_zs['end_date'], curr_zs['start_date'])

            # c段: 后中枢结束后的第一笔下跌
            post_down = [b for b in self.bi_list
                         if b['start_index'] >= curr_zs['end_index'] and b['direction'] == 'down']
            if not post_down:
                continue

            c_bi = post_down[0]
            c_area = self._calc_macd_area(curr_zs['end_date'], c_bi['end_date'])

            # 背驰判断: c段面积 < b段面积的80%
            if b_area > 0 and c_area < b_area * 0.8:
                sig_date = c_bi.get('end_raw_date', c_bi['end_date'])
                signals.append({
                    'date': sig_date,
                    'type': 'first_buy',
                    'price': c_bi['end_price'],
                    'zhongshu_zg': curr_zs['ZG'],
                    'zhongshu_zd': curr_zs['ZD'],
                    'divergence_ratio': round(c_area / max(b_area, 0.001), 2),
                })

        return signals

    def _detect_second_buy(self, first_buys):
        """
        第二类买点: 一买后首次回调不破一买低点

        二买的逻辑:
          一买之后价格上涨 (脱离底部),
          然后出现回调, 但回调的低点高于一买的低点。
          这说明"双底确认", 支撑有效。

        与一买的区别:
          - 一买是左侧交易 (抄底, 风险大)
          - 二买是右侧交易 (确认后入场, 更安全)

        参数:
            first_buys: 一买信号列表 (由 _detect_first_buy 返回)

        返回:
            list[dict]: 二买信号列表
        """
        signals = []
        for fb in first_buys:
            # 一买之后的所有笔
            post_bis = [b for b in self.bi_list if b['start_date'] > fb['date']]
            saw_up = False
            for bi in post_bis:
                if bi['direction'] == 'up':
                    saw_up = True  # 已经看到上涨笔
                elif bi['direction'] == 'down' and saw_up:
                    # 上涨后首次回调
                    if bi['end_price'] > fb['price']:
                        # 回调低点高于一买低点 → 确认二买
                        sig_date = bi.get('end_raw_date', bi['end_date'])
                        signals.append({
                            'date': sig_date,
                            'type': 'second_buy',
                            'price': bi['end_price'],
                            'first_buy_price': fb['price'],
                        })
                    break

        return signals

    def _detect_third_sell(self):
        """
        第三类卖点: 跌破中枢后反弹不进入中枢

        三卖是"三买的镜像"——方向相反, 逻辑相同:
          1. 市场形成一个中枢
          2. 价格跌破中枢下沿 (ZD)
          3. 反弹但升不回中枢
          4. 说明下跌是"真跌破", 趋势将继续向下

        返回:
            list[dict]: 三卖信号列表
        """
        signals = []
        used_dates = set()

        for zs in self.zhongshu_list:
            zg = zs['ZG']
            zd = zs['ZD']

            post_bis = [b for b in self.bi_list if b['start_index'] >= zs['end_index']]
            state = 'WAIT_BREAKDOWN'

            for bi in post_bis:
                if state == 'WAIT_BREAKDOWN':
                    if bi['direction'] == 'down' and bi['end_price'] < zd:
                        state = 'WAIT_BOUNCE'
                elif state == 'WAIT_BOUNCE':
                    if bi['direction'] == 'up':
                        bounce_high = bi['end_price']
                        sig_date = bi.get('end_raw_date', bi['end_date'])
                        if bounce_high < zd and sig_date not in used_dates:
                            signals.append({
                                'date': sig_date,
                                'type': 'third_sell',
                                'price': bounce_high,
                                'zhongshu_zg': zg,
                                'zhongshu_zd': zd,
                            })
                            used_dates.add(sig_date)
                            break
                        else:
                            break

        return signals

    # ============================================================
    # 信号映射（用于 Backtrader 回测）
    # ============================================================

    def get_signal_df(self):
        """
        将买卖点信号映射到原始 DataFrame, 添加信号列

        这是连接"缠论分析"和"策略回测"的桥梁。
        它将信号列表转化为逐日标记的信号列,
        使 Backtrader 策略能在每个交易日检查信号。

        新增列:
          chan_signal: 信号标记
            0  = 无信号
            1  = 第一类买点
            2  = 第二类买点
            3  = 第三类买点
            -3 = 第三类卖点

          chan_zg: 最近中枢的 ZG (向前填充)
            用于策略中的止损判断: 收盘价 < ZG 时止损

          chan_zd: 最近中枢的 ZD (向前填充)
            用于策略中的突破判断

          weekly_trend: 周线趋势 (占位列, 多周期策略使用, 默认0)

        数据填充策略:
          - 信号只在触发当天标记
          - 中枢的 ZG/ZD 在区间内填充, 然后向前填充到所有后续交易日
          - 这样策略在任何时候都能知道当前最近的中枢位置

        返回:
            DataFrame: 包含原始数据和信号列
        """
        df = self.raw_df.copy()
        df['chan_signal'] = 0
        df['chan_zg'] = np.nan
        df['chan_zd'] = np.nan
        df['weekly_trend'] = 0

        # 信号类型到数字编码的映射
        signal_map = {
            'first_buy': 1, 'second_buy': 2, 'third_buy': 3, 'third_sell': -3
        }

        # 在信号触发日标记信号
        for sig in self.signals:
            date = sig['date']
            if date in df.index:
                df.loc[date, 'chan_signal'] = signal_map.get(sig['type'], 0)
                if 'zhongshu_zg' in sig:
                    df.loc[date, 'chan_zg'] = sig['zhongshu_zg']
                if 'zhongshu_zd' in sig:
                    df.loc[date, 'chan_zd'] = sig['zhongshu_zd']

        # 在中枢持续期间填充 ZG/ZD
        for zs in self.zhongshu_list:
            mask = (df.index >= zs['start_date']) & (df.index <= zs['end_date'])
            df.loc[mask, 'chan_zg'] = df.loc[mask, 'chan_zg'].fillna(zs['ZG'])
            df.loc[mask, 'chan_zd'] = df.loc[mask, 'chan_zd'].fillna(zs['ZD'])

        # 向前填充: 中枢结束后, 最近的中枢 ZG/ZD 仍然有效
        df['chan_zg'] = df['chan_zg'].ffill()
        df['chan_zd'] = df['chan_zd'].ffill()

        return df

    # ============================================================
    # 分析摘要
    # ============================================================

    def summary(self):
        """
        打印缠论分析结果摘要

        输出内容包括:
          - 原始K线数和合并后K线数 (噪音过滤效果)
          - 分型统计 (顶/底分型数量)
          - 笔统计 (上升/下降笔数量, 平均幅度)
          - 中枢列表和详细信息
          - 买卖点信号列表

        这是快速了解分析结果的入口, 适合在 Jupyter/命令行中使用。
        """
        top_count = sum(1 for f in self.fractals if f['type'] == 'top')
        bot_count = sum(1 for f in self.fractals if f['type'] == 'bottom')
        up_count = sum(1 for b in self.bi_list if b['direction'] == 'up')
        down_count = sum(1 for b in self.bi_list if b['direction'] == 'down')

        print("=" * 60)
        print("缠论分析摘要")
        print("=" * 60)
        print(f"  原始K线:   {len(self.raw_df)} 根")
        print(f"  合并后K线: {len(self.merged_df)} 根 (合并了 {len(self.raw_df) - len(self.merged_df)} 根)")
        print(f"  分型:      {len(self.fractals)} 个 (顶分型 {top_count}, 底分型 {bot_count})")
        print(f"  笔:        {len(self.bi_list)} 笔 (上升 {up_count}, 下降 {down_count})")
        print(f"  中枢:      {len(self.zhongshu_list)} 个")

        if self.bi_list:
            up_bis = [b for b in self.bi_list if b['direction'] == 'up']
            down_bis = [b for b in self.bi_list if b['direction'] == 'down']
            if up_bis:
                avg_up = np.mean([abs(b['end_price'] - b['start_price']) for b in up_bis])
                print(f"  上升笔均幅: {avg_up:.2f}")
            if down_bis:
                avg_down = np.mean([abs(b['end_price'] - b['start_price']) for b in down_bis])
                print(f"  下降笔均幅: {avg_down:.2f}")

        if self.zhongshu_list:
            print("\n  中枢列表:")
            for i, zs in enumerate(self.zhongshu_list):
                print(f"    [{i+1}] {zs['start_date'].strftime('%Y-%m-%d')} ~ "
                      f"{zs['end_date'].strftime('%Y-%m-%d')} | "
                      f"ZG={zs['ZG']:.2f} ZD={zs['ZD']:.2f} | "
                      f"包含{zs['bi_count']}笔")

        print(f"\n  信号:      {len(self.signals)} 个")
        sig_names = {
            'first_buy': '一买', 'second_buy': '二买',
            'third_buy': '三买', 'third_sell': '三卖',
        }
        for sig in self.signals:
            name = sig_names.get(sig['type'], sig['type'])
            extra = ''
            if 'divergence_ratio' in sig:
                extra = f" | 背驰比={sig['divergence_ratio']}"
            print(f"    {sig['date'].strftime('%Y-%m-%d')} | {name} | "
                  f"价格={sig['price']:.2f}{extra}")
        print("=" * 60)

    # ============================================================
    # 可视化
    # ============================================================

    @staticmethod
    def _draw_candlestick(ax, df, width_ratio=0.8):
        """
        在指定 matplotlib Axes 上绘制标准K线蜡烛图

        与常见 K线图的不同:
          - x 轴使用连续的整数位置 (0, 1, 2, ...), 没有周末缺口
          - 这样缠论结构 (笔/中枢) 的绘制不受非交易日影响
          - 日期标签智能选择: 数据少时用"月-日", 多时用"年-月"

        参数:
            ax: matplotlib Axes 对象
            df: DataFrame, 需包含 open/high/low/close
            width_ratio: K线实体宽度与间距的比例 (默认0.8)

        返回:
            date_to_x: dict, 将日期映射到 x 轴整数位置
                       (用于在 K线图上叠加分型/笔/中枢等标记)

        K线绘制规则:
          - 阳线 (收盘 >= 开盘): 红色实体 (#e74c3c)
          - 阴线 (收盘 < 开盘):  绿色实体 (#27ae60)
          - 用 ax.bar 绘制实体, ax.plot 绘制上下影线
          - 实体最小高度限制: 防止价格不变时实体不可见
        """
        import matplotlib.ticker as mticker

        n = len(df)
        date_to_x = {}  # 日期 -> x轴整数位置 的映射
        for i, dt in enumerate(df.index):
            date_to_x[dt] = i
            if hasattr(dt, 'date'):
                date_to_x[dt.date()] = i

        body_width = width_ratio

        opens = df['open'].values.astype(float)
        highs = df['high'].values.astype(float)
        lows = df['low'].values.astype(float)
        closes = df['close'].values.astype(float)

        # 计算最小实体高度 (防止价格不变时实体为0)
        price_range = highs.max() - lows.min()
        min_body = price_range * 0.002

        for i in range(n):
            o, h, l, c = opens[i], highs[i], lows[i], closes[i]

            if c >= o:
                color = '#e74c3c'  # 阳线 (红色, 中国习惯)
                body_bottom = o
                body_height = max(c - o, min_body)
            else:
                color = '#27ae60'  # 阴线 (绿色, 中国习惯)
                body_bottom = c
                body_height = max(o - c, min_body)

            # 绘制影线 (上下引线)
            ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=1)
            # 绘制实体 (用 bar 而不是 Rectangle 更简洁)
            ax.bar(i, body_height, bottom=body_bottom, width=body_width,
                   color=color, edgecolor=color, linewidth=0.5, zorder=2)

        # 智能选择 x 轴刻度标签的密度
        total_days = (df.index[-1] - df.index[0]).days if n > 1 else 365
        if n <= 60:
            step = max(1, n // 15)
        elif n <= 200:
            step = max(1, n // 12)
        else:
            step = max(1, n // 10)
        tick_positions = list(range(0, n, step))
        if (n - 1) not in tick_positions:
            tick_positions.append(n - 1)

        # 时间段短用"月-日", 段长用"年-月"
        if total_days <= 180:
            tick_labels = [df.index[i].strftime('%m-%d') for i in tick_positions]
        else:
            tick_labels = [df.index[i].strftime('%Y-%m') for i in tick_positions]

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlim(-1, n)

        return date_to_x

    def plot(self, title='', save_path=None, show_bi=True, show_zhongshu=True,
             show_signals=True, show_fractals=True, show_all_fractals=False):
        """
        绘制缠论分析图表 (K线蜡烛图 + 缠论结构)

        图表布局:
          上图: K线蜡烛图 + 分型标记 + 笔连线 + 中枢方框 + 买卖点标注
          下图: 成交量柱状图 (与K线图共享x轴)

        参数:
            title: 图表标题
            save_path: 保存路径, 如 'outputs/xxx.png' (不传则不保存)
            show_bi: 是否显示笔连线
            show_zhongshu: 是否显示中枢方框
            show_signals: 是否显示买卖点信号
            show_fractals: 是否显示分型标记
            show_all_fractals: True=显示所有分型, False=只显示参与成笔的确认分型
                               (默认只显示确认分型, 减少视觉噪音)
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import matplotlib.dates as mdates
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 10),
                                        gridspec_kw={'height_ratios': [4, 1]})

        df = self.raw_df

        # ============ 上图: K线蜡烛图 + 缠论结构 ============
        d2x = self._draw_candlestick(ax1, df)

        def _lookup_x(date_val):
            """在 date_to_x 字典中查找日期对应的 x 位置, 支持多种日期格式"""
            if date_val in d2x:
                return d2x[date_val]
            if hasattr(date_val, 'date'):
                return d2x.get(date_val.date())
            for k, v in d2x.items():
                if hasattr(k, 'date') and k.date() == date_val:
                    return v
                if k == date_val:
                    return v
            return None

        # --- 分型标记 ---
        if show_fractals:
            frac_list = self.fractals if show_all_fractals else self.confirmed_fractals
            frac_label_suffix = '全部' if show_all_fractals else '确认'
            if frac_list:
                for f in frac_list:
                    rd = f.get('raw_date', f['date'])
                    x = _lookup_x(rd)
                    if x is None:
                        continue
                    if f['type'] == 'top':
                        ax1.scatter(x, f['price'], marker='v', color='#e74c3c',
                                    s=50, zorder=5, alpha=0.7)
                    else:
                        ax1.scatter(x, f['price'], marker='^', color='#2ecc71',
                                    s=50, zorder=5, alpha=0.7)

        # --- 笔连线 ---
        if show_bi and self.bi_list:
            for bi in self.bi_list:
                color = '#e74c3c' if bi['direction'] == 'up' else '#27ae60'
                sx = _lookup_x(bi.get('start_raw_date', bi['start_date']))
                ex = _lookup_x(bi.get('end_raw_date', bi['end_date']))
                if sx is not None and ex is not None:
                    ax1.plot([sx, ex], [bi['start_price'], bi['end_price']],
                             color=color, linewidth=1.8, alpha=0.85, zorder=4)

        # --- 中枢方框 ---
        if show_zhongshu and self.zhongshu_list:
            for zs in self.zhongshu_list:
                xl = _lookup_x(zs['start_date'])
                xr = _lookup_x(zs['end_date'])
                if xl is not None and xr is not None:
                    rect = patches.Rectangle(
                        (xl, zs['ZD']),
                        xr - xl,
                        zs['ZG'] - zs['ZD'],
                        linewidth=1.5,
                        edgecolor='#3498db',
                        facecolor='#3498db',
                        alpha=0.15,
                        zorder=3,
                    )
                    ax1.add_patch(rect)
                    ax1.text(xl, zs['ZG'],
                             f" ZG={zs['ZG']:.1f}\n ZD={zs['ZD']:.1f}",
                             fontsize=7, color='#2c3e50', va='bottom')

        # --- 买卖点信号标记 ---
        if show_signals and self.signals:
            sig_names = {
                'first_buy': '一买', 'second_buy': '二买',
                'third_buy': '三买', 'third_sell': '三卖',
            }
            sig_colors = {
                'first_buy': '#8e44ad', 'second_buy': '#e67e22',
                'third_buy': '#e74c3c', 'third_sell': '#27ae60',
            }
            for sig in self.signals:
                sig_x = _lookup_x(sig['date'])
                if sig_x is None:
                    continue
                marker = '^' if 'buy' in sig['type'] else 'v'
                color = sig_colors.get(sig['type'], '#333333')
                ax1.scatter(sig_x, sig['price'], marker=marker, color=color,
                            s=200, zorder=7, edgecolors='black', linewidths=1)
                ax1.annotate(sig_names.get(sig['type'], sig['type']),
                             (sig_x, sig['price']),
                             textcoords="offset points", xytext=(10, 10 if 'buy' in sig['type'] else -15),
                             fontsize=9, fontweight='bold', color=color,
                             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

        ax1.set_title(title or '缠论分析', fontsize=14, fontweight='bold')
        ax1.set_ylabel('价格')
        handles, labels = ax1.get_legend_handles_labels()
        if handles:
            ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ============ 下图: 成交量 ============
        # 使用与K线图相同的连续 x 轴
        vol_x = list(range(len(df)))
        # 成交量颜色与K线一致: 阳线红色, 阴线绿色
        vol_colors = ['#e74c3c' if df['close'].iloc[i] >= df['open'].iloc[i] else '#27ae60'
                      for i in range(len(df))]
        ax2.bar(vol_x, df['volume'], color=vol_colors, alpha=0.6, width=0.8)
        n = len(df)
        total_days = (df.index[-1] - df.index[0]).days if n > 1 else 365
        step = max(1, n // 12)
        tick_pos = list(range(0, n, step))
        if (n - 1) not in tick_pos:
            tick_pos.append(n - 1)
        if total_days <= 180:
            tick_lbl = [df.index[i].strftime('%m-%d') for i in tick_pos]
        else:
            tick_lbl = [df.index[i].strftime('%Y-%m') for i in tick_pos]
        ax2.set_xticks(tick_pos)
        ax2.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=8)
        ax2.set_xlim(-1, n)
        ax2.set_ylabel('成交量')
        ax2.set_xlabel('日期')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'outputs',
                        exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  图表已保存: {save_path}")

        plt.close()
        return fig

    def plot_compare_merge(self, save_path=None):
        """
        对比K线合并前后的分型识别差异

        并排显示两张图:
          左图: 原始K线 + "伪分型" (直接在原始K线上识别的分型)
          右图: 合并后K线 + 正式分型 (在包含处理后的K线上识别的分型)

        这个对比能直观展示包含处理的价值:
          合并前有很多因为包含关系造成的噪声分型,
          合并后噪声被过滤, 留下的都是有意义的转折点。
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei']
        matplotlib.rcParams['axes.unicode_minus'] = False

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

        raw = self.raw_df
        merged = self.merged_df

        # ============ 左图: 原始K线 + "伪分型" ============
        d2x_raw = self._draw_candlestick(ax1, raw)

        # 直接在原始 K 线上识别分型 (不处理包含关系)
        for i in range(1, len(raw) - 1):
            h_p, h_c, h_n = raw['high'].iloc[i-1], raw['high'].iloc[i], raw['high'].iloc[i+1]
            l_p, l_c, l_n = raw['low'].iloc[i-1], raw['low'].iloc[i], raw['low'].iloc[i+1]
            if h_c > h_p and h_c > h_n and l_c > l_p and l_c > l_n:
                ax1.scatter(i, h_c,
                            marker='v', color='#e74c3c', s=40, alpha=0.6, zorder=5)
            elif l_c < l_p and l_c < l_n and h_c < h_p and h_c < h_n:
                ax1.scatter(i, l_c,
                            marker='^', color='#2ecc71', s=40, alpha=0.6, zorder=5)

        ax1.set_title(f'合并前 (原始{len(raw)}根K线)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('价格')
        ax1.grid(True, alpha=0.3)

        # ============ 右图: 合并后K线 + 正式分型 ============
        d2x_merged = self._draw_candlestick(ax2, merged)

        for f in self.fractals:
            x = d2x_merged.get(f['date'])
            if x is None:
                continue
            if f['type'] == 'top':
                ax2.scatter(x, f['price'],
                            marker='v', color='#e74c3c', s=60, zorder=5)
            else:
                ax2.scatter(x, f['price'],
                            marker='^', color='#2ecc71', s=60, zorder=5)

        ax2.set_title(f'合并后 ({len(merged)}根K线, 合并{len(raw)-len(merged)}根)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('价格')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else 'outputs',
                        exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  图表已保存: {save_path}")

        plt.close()
        return fig
