# -*- coding: utf-8 -*-
"""
chan.py 封装模块
===============

本模块将开源项目 chan.py 的 CChan 接口封装成方便教学和策略使用的数据结构。

chan.py 是什么?
  chan.py 是缠论 (Chan Theory) 的 Python 实现库, 可以自动识别:
    - 分型 (Fractal): 顶分型和底分型, 是缠论的最小构件
    - 笔 (Bi): 相邻的顶底分型连接, 代表一段价格波动
    - 线段 (Segment): 笔的递归组合, 代表更大级别的价格走势
    - 中枢 (ZhongShu / ZS): 至少3笔价格重叠区域, 代表价格震荡区间
    - 买卖点 (BSP): 基于中枢的买卖信号 (一买/二买/三买/一卖/二卖/三卖)

本模块提供的功能:
  1. run_chan(): 对 K 线数据运行缠论分析, 提取所有缠论元素
  2. chan_to_signal_df(): 将缠论分析结果转换为带信号列的 DataFrame,
     可直接输入 ChanPandasData 供回测使用
  3. draw_chan_chart(): 在 matplotlib 坐标轴上绘制缠论结构图

依赖:
  chan.py 库位于上级目录的 chan.py/ 文件夹中
  (具体位置: e:/AI/aStock/chan.py/)

使用方式:
    from chanpy_wrapper import run_chan, chan_to_signal_df, draw_chan_chart

    chan_data = run_chan(df, symbol='600519.SH')
    signal_df = chan_to_signal_df(df, chan_data)
    draw_chan_chart(ax, df, chan_data)
"""

import sys
import os
import pandas as pd
import numpy as np

# chan.py 库的路径: 位于上级目录的 chan.py/ 文件夹
CHAN_PY_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', 'chan.py'))


def _ensure_path():
    """
    确保 chan.py 库的路径在 Python 模块搜索路径中。

    Python 默认只在 site-packages 和脚本所在目录搜索模块。
    chan.py 位于项目根目录下, 需要手动加入 sys.path。
    """
    if CHAN_PY_PATH not in sys.path:
        sys.path.insert(0, CHAN_PY_PATH)


def _ts(t):
    """
    将 chan.py 的时间对象转换为 pandas Timestamp。

    chan.py 内部使用自定义的时间格式, 而 pandas 使用 Timestamp。
    这个转换函数使两种时间格式可以互操作。

    参数:
        t: chan.py 的时间对象 (有 year/month/day 属性)

    返回值:
        pd.Timestamp
    """
    return pd.Timestamp(f'{t.year:04d}-{t.month:02d}-{t.day:02d}')


def run_chan(df, symbol='stock', config_dict=None):
    """
    对 K 线 DataFrame 运行 chan.py 缠论分析。

    这是核心函数, 完成以下工作:
      1. 将 DataFrame 注册到 chan.py 的数据 API
      2. 创建 CChan 实例, 指定日线级别
      3. 运行缠论分析算法
      4. 提取: 合并K线、分型、笔、线段、中枢、买卖点

    参数:
        df: DataFrame, 必须含 open/high/low/close/volume 五列,
           索引为 DatetimeIndex
        symbol: 股票代码标识 (用于 chan.py 内部缓存)
        config_dict: CChanConfig 参数字典 (可选)。
            常用配置项:
            - 'bi_fx_check': 笔的分型确认方式
            - 'gap_as_bi': 是否将跳空视为一笔
            - 'max_klc_count': 合并K线的最大包含数量

    返回值:
        dict, 包含以下键:
          - 'klc_list':  合并K线列表 [{'date', 'high', 'low', 'idx', 'fx', 'raw_count'}, ...]
          - 'fractals':  分型列表 (顶/底分型)
          - 'bi_list':   笔列表 [{start/end日期/价格, 方向, 买卖点, ...}]
          - 'seg_list':  线段列表
          - 'zs_list':   中枢列表 [{ZG, ZD, center, start/end, ...}]
          - 'bsp_list':  买卖点列表 (有信号的笔)
          - 'raw_kl':    chan.py 原始 KLine 对象 (高级使用)

    异常:
        ImportError: 如果找不到 chan.py 库
    """
    _ensure_path()
    from DataAPI import DfApi as DfApiModule
    from Common.CEnum import KL_TYPE
    from Chan import CChan
    from ChanConfig import CChanConfig

    # 将 DataFrame 注册到 chan.py 的数据 API
    DfApiModule._DF_CACHE[symbol] = df

    # 创建 chan.py 配置和实例
    # K_DAY 表示日线级别, lv_list 只传一个级别表示单级别分析
    config = CChanConfig(config_dict or {})
    chan = CChan(
        code=symbol,
        data_src='custom:DfApi.DfApi',  # 使用自定义数据源
        lv_list=[KL_TYPE.K_DAY],       # 日线级别
        config=config,
    )
    kl = chan[0]  # 获取第一个级别的 KLine 对象

    # ---- 提取合并K线 (KLC) ----
    # 缠论的"包含处理": 相邻K线有包含关系时合并成一根
    # klc_list 包含合并后的K线, 以及分型标记
    klc_list = []
    for klc in kl.lst:
        t0 = klc.lst[0].time
        fx_type = 'unknown'
        if hasattr(klc, 'fx') and klc.fx is not None:
            fn = klc.fx.name if hasattr(klc.fx, 'name') else str(klc.fx)
            if 'TOP' in fn:
                fx_type = 'top'       # 顶分型
            elif 'BOTTOM' in fn:
                fx_type = 'bottom'    # 底分型
        klc_list.append({
            'date': _ts(t0),
            'high': float(klc.high),
            'low': float(klc.low),
            'idx': klc.idx,
            'fx': fx_type,            # 'top' / 'bottom' / 'unknown'
            'raw_count': len(klc.lst),# 包含了几根原始K线
        })

    # ---- 提取分型 ----
    # 分型是缠论最基本的构件: 顶分型 = 中间K线最高, 底分型 = 中间K线最低
    fractals = [k for k in klc_list if k['fx'] in ('top', 'bottom')]

    # ---- 提取笔 ----
    # 笔 = 相邻的顶底分型连接, 是缠论中的基本趋势单位
    bi_list = []
    for bi in kl.bi_list:
        is_up = bi.dir.name == 'UP'  # 向上笔或向下笔
        bklc = bi.begin_klc          # 笔的起始KLC
        eklc = bi.end_klc            # 笔的结束KLC
        bt = _ts(bklc.lst[0].time)
        et = _ts(eklc.lst[-1].time)

        # 笔的起始/结束价格: 向上笔关注低点→高点, 向下笔关注高点→低点
        if is_up:
            sp, ep = float(bklc.low), float(eklc.high)
        else:
            sp, ep = float(bklc.high), float(eklc.low)

        # 寻找笔内最高/最低K线 (极值K线)
        if is_up:
            peak_klu = eklc.get_high_peak_klu() if hasattr(eklc, 'get_high_peak_klu') else None
            trough_klu = bklc.get_low_peak_klu() if hasattr(bklc, 'get_low_peak_klu') else None
        else:
            peak_klu = bklc.get_high_peak_klu() if hasattr(bklc, 'get_high_peak_klu') else None
            trough_klu = eklc.get_low_peak_klu() if hasattr(eklc, 'get_low_peak_klu') else None
        start_raw = _ts(trough_klu.time) if (is_up and trough_klu) else bt
        end_raw = _ts(peak_klu.time) if (is_up and peak_klu) else et
        if not is_up:
            start_raw = _ts(peak_klu.time) if peak_klu else bt
            end_raw = _ts(trough_klu.time) if trough_klu else et

        bi_item = {
            'start_date': bt,
            'end_date': et,
            'start_raw_date': start_raw,  # 极值K线日期（用于绘图更精确）
            'end_raw_date': end_raw,
            'start_price': sp,
            'end_price': ep,
            'direction': 'up' if is_up else 'down',
            'is_sure': bi.is_sure if hasattr(bi, 'is_sure') else True,  # 笔是否确认
            'klc_count': bi.get_klc_cnt() if hasattr(bi, 'get_klc_cnt') else 0,  # 包含KLC数
            'klu_count': bi.get_klu_cnt() if hasattr(bi, 'get_klu_cnt') else 0,  # 包含K线数
            'seg_idx': bi.seg_idx if hasattr(bi, 'seg_idx') else -1,  # 所属线段索引
        }

        # MACD 面积: 笔内部的 MACD 柱状图面积和, 衡量笔的动能
        try:
            bi_item['macd_area'] = float(bi.Cal_MACD_area())
        except Exception:
            bi_item['macd_area'] = 0.0

        # 买卖点 (BSP): 笔结束时的买卖信号
        if bi.bsp:
            bsp_types = [str(t.value) if hasattr(t, 'value') else str(t) for t in bi.bsp.type]
            bi_item['bsp_type'] = ','.join(bsp_types)  # 可能多个类型,如 '1,2'
            bi_item['bsp_is_buy'] = bi.bsp.is_buy      # True=买点, False=卖点
            bi_item['bsp_date'] = _ts(bi.bsp.klu.time) if hasattr(bi.bsp, 'klu') else et
        else:
            bi_item['bsp_type'] = None
            bi_item['bsp_is_buy'] = None
            bi_item['bsp_date'] = None

        bi_list.append(bi_item)

    # ---- 提取线段 ----
    # 线段 = 笔的递归组合, 反映更高级别的走势
    seg_list = []
    for seg in kl.seg_list:
        bt = _ts(seg.start_bi.begin_klc.lst[0].time)
        et = _ts(seg.end_bi.end_klc.lst[-1].time)
        is_up = seg.dir.name == 'UP'

        # 线段内包含的中枢
        seg_zs_list = []
        if hasattr(seg, 'zs_lst'):
            for zs in seg.zs_lst:
                seg_zs_list.append({
                    'ZG': float(zs.high),  # 中枢上沿
                    'ZD': float(zs.low),   # 中枢下沿
                })

        seg_list.append({
            'start_date': bt,
            'end_date': et,
            'direction': 'up' if is_up else 'down',
            'is_sure': seg.is_sure if hasattr(seg, 'is_sure') else True,
            'bi_count': seg.cal_bi_cnt() if hasattr(seg, 'cal_bi_cnt') else 0,  # 包含笔数
            'zs_list': seg_zs_list,  # 中枢列表
        })

    # ---- 提取中枢 ----
    # 中枢是缠论最核心的概念: 至少3笔重叠的区间
    # ZG = 中枢上沿 (最高点的最低点), ZD = 中枢下沿 (最低点的最高点)
    zs_list = []
    for zs in kl.zs_list:
        begin_t = None
        end_t = None
        if hasattr(zs, 'begin_bi') and zs.begin_bi is not None:
            begin_t = _ts(zs.begin_bi.begin_klc.lst[0].time)
        elif hasattr(zs, 'begin') and hasattr(zs.begin, 'lst'):
            begin_t = _ts(zs.begin.lst[0].time)
        if hasattr(zs, 'end_bi') and zs.end_bi is not None:
            end_t = _ts(zs.end_bi.end_klc.lst[-1].time)
        elif hasattr(zs, 'end') and hasattr(zs.end, 'lst'):
            end_t = _ts(zs.end.lst[-1].time)
        zs_list.append({
            'ZG': float(zs.high),                             # 中枢上沿
            'ZD': float(zs.low),                              # 中枢下沿
            'center': float(zs.mid) if hasattr(zs, 'mid') else (float(zs.high) + float(zs.low)) / 2,
            'start_date': begin_t,                            # 中枢开始时间
            'end_date': end_t,                                # 中枢结束时间
            'peak_high': float(zs.peak_high) if hasattr(zs, 'peak_high') else float(zs.high),
            'peak_low': float(zs.peak_low) if hasattr(zs, 'peak_low') else float(zs.low),
        })

    # ---- 提取买卖点汇总 ----
    # 只有包含买卖点信号的笔才加入 bsp_list
    bsp_list = [b for b in bi_list if b['bsp_type'] is not None]

    # 清理缓存
    DfApiModule._DF_CACHE.pop(symbol, None)

    return {
        'klc_list': klc_list,
        'fractals': fractals,
        'bi_list': bi_list,
        'seg_list': seg_list,
        'zs_list': zs_list,
        'bsp_list': bsp_list,
        'raw_kl': kl,  # 原始 KLine 对象, 可供高级分析使用
    }


def chan_to_signal_df(df, chan_data):
    """
    将 run_chan() 的结果转换为 ChanPandasData 兼容的 DataFrame。

    新增列:
      chan_signal: 缠论买卖点信号
        0 = 无信号, 1 = 一买, 2 = 二买, 3 = 三买
        -1 = 一卖, -2 = 二卖, -3 = 三卖
      chan_zg: 当前所在中枢的 ZG (中枢上沿), 向前填充
      chan_zd: 当前所在中枢的 ZD (中枢下沿), 向前填充

    参数:
        df: 原始 K 线 DataFrame (含 open/high/low/close/volume)
        chan_data: run_chan() 的返回值

    返回值:
        DataFrame, 包含原始列 + 3个信号列 (chan_signal/chan_zg/chan_zd),
        可直接送入 ChanPandasData 进行回测

    实现细节:
      - 买卖点映射: chan.py 的信号类型和数值信号之间的转换
      - ZG/ZD 填充: 中枢区间内的交易日都标记 ZG/ZD 值
      - 向前填充: 在中枢结束后直到下一个中枢形成前, 保持上一个中枢的 ZG/ZD
    """
    result = df.copy()
    result['chan_signal'] = 0      # 默认无信号
    result['chan_zg'] = np.nan     # 中枢上沿, 初始为 NaN
    result['chan_zd'] = np.nan     # 中枢下沿, 初始为 NaN

    # ---- 买卖点信号映射表 ----
    # key = chan.py 的买卖点类型, value = (信号值, 是否买入)
    # 信号值对照:
    #   正数 = 买点 (越大越"晚", 风险越高)
    #   负数 = 卖点 (绝对值越大越"晚")
    bsp_signal_map = {
        '1': (1, True), '2': (2, True), '2s': (2, True), '3': (3, True),
        '1s': (-1, False), '2_sell': (-2, False), '3_sell': (-3, False),
        '3a': (3, True),
    }

    # 遍历所有有买卖点信号的笔, 在信号日期上标记
    for bi in chan_data['bsp_list']:
        bsp_type = bi['bsp_type']
        bsp_date = bi['bsp_date']
        if bsp_date is None or bsp_type is None:
            continue
        if bsp_date not in result.index:
            continue

        # 一个笔可能有多个信号类型 (逗号分隔), 取绝对值最大的
        best_signal = 0
        for t in bsp_type.split(','):
            t = t.strip()
            if t in bsp_signal_map:
                sig_val, is_buy = bsp_signal_map[t]
                if abs(sig_val) > abs(best_signal):
                    best_signal = sig_val

        if best_signal != 0:
            result.loc[bsp_date, 'chan_signal'] = best_signal

    # ---- 填充中枢的 ZG/ZD ----
    # 在中枢有效期内, 每个交易日都标记该中枢的上下沿
    for zs in chan_data['zs_list']:
        if zs['start_date'] is None or zs['end_date'] is None:
            continue
        mask = (result.index >= zs['start_date']) & (result.index <= zs['end_date'])
        result.loc[mask, 'chan_zg'] = result.loc[mask, 'chan_zg'].fillna(zs['ZG'])
        result.loc[mask, 'chan_zd'] = result.loc[mask, 'chan_zd'].fillna(zs['ZD'])

    # 在信号点也填充对应中枢的 ZG/ZD (有些信号在中枢结束之后)
    for bi in chan_data['bsp_list']:
        bsp_date = bi['bsp_date']
        if bsp_date is None or bsp_date not in result.index:
            continue
        # 倒序查找最近的中枢
        for zs in reversed(chan_data['zs_list']):
            if zs['end_date'] and bsp_date >= zs['end_date']:
                if np.isnan(result.loc[bsp_date, 'chan_zg']):
                    result.loc[bsp_date, 'chan_zg'] = zs['ZG']
                if np.isnan(result.loc[bsp_date, 'chan_zd']):
                    result.loc[bsp_date, 'chan_zd'] = zs['ZD']
                break

    # 向前填充: 中枢结束后到下一个中枢开始之间, 沿用上一个中枢的 ZG/ZD
    result['chan_zg'] = result['chan_zg'].ffill()
    result['chan_zd'] = result['chan_zd'].ffill()

    return result


def draw_chan_chart(ax, df, chan_data, show_bi=True, show_seg=True,
                   show_zs=True, show_bsp=True, show_fractals=False,
                   show_grid_levels=None):
    """
    在 matplotlib Axes 上绘制 chan.py 缠论分析结果的蜡烛图。

    这是教学可视化函数, 可以在同一张图上叠加显示:
      - K线蜡烛图
      - 分型标记 (顶/底)
      - 笔 (红色=向上笔, 绿色=向下笔)
      - 线段 (不同颜色的虚线)
      - 中枢 (蓝色半透明矩形)
      - 买卖点 (带标注的大标记)
      - 网格线 (可选, 教学展示用)

    参数:
        ax: matplotlib Axes 对象
        df: 原始 K 线 DataFrame
        chan_data: run_chan() 返回的缠论分析结果
        show_bi: 是否显示笔
        show_seg: 是否显示线段
        show_zs: 是否显示中枢
        show_bsp: 是否显示买卖点
        show_fractals: 是否显示分型
        show_grid_levels: 可选 list of float, 在图上画水平网格线

    返回值:
        date_to_x: dict, 日期到 x 轴位置的映射,
                  用于在其他图表上标注与 K 线对应的位置
    """
    import matplotlib.patches as patches

    # 先画简化版蜡烛图
    d2x = _draw_candlestick(ax, df)

    def _lx(date_val):
        """
        日期转 x 坐标的内部函数。
        支持 Timestamp、date 对象和字符串三种输入。
        """
        if date_val in d2x:
            return d2x[date_val]
        if hasattr(date_val, 'date'):
            return d2x.get(date_val.date())
        for k, v in d2x.items():
            if hasattr(k, 'date') and k.date() == date_val:
                return v
        return None

    # ---- 分型 ----
    if show_fractals and chan_data['fractals']:
        for f in chan_data['fractals']:
            x = _lx(f['date'])
            if x is None:
                continue
            if f['fx'] == 'top':
                ax.scatter(x, f['high'], marker='v', color='#e74c3c', s=50, zorder=5, alpha=0.6)
            else:
                ax.scatter(x, f['low'], marker='^', color='#2ecc71', s=50, zorder=5, alpha=0.6)

    # ---- 笔 ----
    # 向上笔红色, 向下笔绿色, 从起始极值点画到结束极值点
    if show_bi and chan_data['bi_list']:
        for bi in chan_data['bi_list']:
            c = '#e74c3c' if bi['direction'] == 'up' else '#27ae60'
            sx = _lx(bi['start_raw_date'])
            ex = _lx(bi['end_raw_date'])
            if sx is not None and ex is not None:
                ax.plot([sx, ex], [bi['start_price'], bi['end_price']],
                        color=c, linewidth=1.8, alpha=0.85, zorder=4)

    # ---- 线段 ----
    # 用不同颜色的虚线连接, 体现更高层级的价格走势
    if show_seg and chan_data['seg_list']:
        seg_colors = ['#9b59b6', '#e67e22', '#1abc9c', '#e74c3c']
        for si, seg in enumerate(chan_data['seg_list']):
            # 收集该线段内的笔
            seg_bis = [b for b in chan_data['bi_list']
                       if b['start_date'] >= seg['start_date']
                       and b['end_date'] <= seg['end_date']]
            if seg_bis:
                if seg['direction'] == 'up':
                    sp = min(b['start_price'] for b in seg_bis)
                    ep = max(b['end_price'] for b in seg_bis)
                else:
                    sp = max(b['start_price'] for b in seg_bis)
                    ep = min(b['end_price'] for b in seg_bis)
                sx = _lx(seg['start_date'])
                ex = _lx(seg['end_date'])
                if sx is not None and ex is not None:
                    sc = seg_colors[si % len(seg_colors)]
                    ax.plot([sx, ex], [sp, ep],
                            color=sc, linewidth=3.5, linestyle='--', alpha=0.7, zorder=3)

    # ---- 中枢 ----
    # 用半透明蓝色矩形标示, 标注 ZG/ZD 数值
    if show_zs and chan_data['zs_list']:
        for zs in chan_data['zs_list']:
            if zs['start_date'] and zs['end_date']:
                xl = _lx(zs['start_date'])
                xr = _lx(zs['end_date'])
                if xl is not None and xr is not None:
                    rect = patches.Rectangle(
                        (xl, zs['ZD']),         # 左下角坐标
                        xr - xl,                 # 宽度
                        zs['ZG'] - zs['ZD'],     # 高度
                        linewidth=1.5, edgecolor='#3498db', facecolor='#3498db',
                        alpha=0.15, zorder=2,
                    )
                    ax.add_patch(rect)
                    ax.text(xl, zs['ZG'],
                            f" ZG={zs['ZG']:.1f}\n ZD={zs['ZD']:.1f}",
                            fontsize=7, color='#2c3e50', va='bottom')

    # ---- 买卖点 ----
    # 用大标记 + 中文标注, 区分买点(^)和卖点(v)
    if show_bsp and chan_data['bsp_list']:
        bsp_names = {'1': '一买', '2': '二买', '2s': '二买S', '3': '三买',
                     '1s': '一卖', '2_sell': '二卖', '3_sell': '三卖', '3a': '三买'}
        bsp_colors = {'1': '#8e44ad', '2': '#e67e22', '2s': '#f39c12',
                      '3': '#e74c3c', '1s': '#2c3e50', '2_sell': '#27ae60',
                      '3_sell': '#16a085', '3a': '#e74c3c'}
        for bi in chan_data['bsp_list']:
            bsp_type = bi['bsp_type']
            is_buy = bi['bsp_is_buy']
            x = _lx(bi['bsp_date'])
            if x is None:
                continue
            marker = '^' if is_buy else 'v'
            first_type = bsp_type.split(',')[0].strip()
            color = bsp_colors.get(first_type, '#333333')
            name = bsp_names.get(first_type, bsp_type)
            price = bi['end_price']
            ax.scatter(x, price, marker=marker, color=color,
                       s=200, zorder=7, edgecolors='black', linewidths=1)
            offset_y = 10 if is_buy else -15
            ax.annotate(name, (x, price),
                        textcoords="offset points", xytext=(10, offset_y),
                        fontsize=9, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    # ---- 网格线（可选）----
    # 用于展示网格策略中的价位线
    if show_grid_levels:
        n = len(df)
        for level in show_grid_levels:
            ax.axhline(y=level, color='#f39c12', linewidth=0.8, linestyle=':', alpha=0.6, zorder=1)

    return d2x


def _draw_candlestick(ax, df, width_ratio=0.6):
    """
    在 matplotlib Axes 上绘制标准 K 线蜡烛图。

    每根 K 线显示为:
      - 实体: 开盘价到收盘价的矩形 (阳线红色, 阴线绿色)
      - 影线: 最高价到最低价的细线

    参数:
        ax: matplotlib Axes
        df: K 线 DataFrame (需含 open/high/low/close)
        width_ratio: K线实体宽度比例 (0~1)

    返回值:
        date_to_x: dict {date: x_position}, 日期到 x 轴整数位置的映射
    """
    n = len(df)
    d2x = {}
    for i, dt in enumerate(df.index):
        d2x[dt] = i                      # Timestamp → 整数索引
        if hasattr(dt, 'date'):
            d2x[dt.date()] = i           # date对象 → 整数索引

    opens = df['open'].values.astype(float)
    highs = df['high'].values.astype(float)
    lows = df['low'].values.astype(float)
    closes = df['close'].values.astype(float)
    price_range = highs.max() - lows.min()
    min_body = price_range * 0.002  # 最小实体高度, 防止阳/阴线不可见

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        if c >= o:
            color = '#e74c3c'                # 阳线: 红色 (A股惯例)
            body_bottom, body_height = o, max(c - o, min_body)
        else:
            color = '#27ae60'                # 阴线: 绿色
            body_bottom, body_height = c, max(o - c, min_body)
        # 影线
        ax.plot([i, i], [l, h], color=color, linewidth=0.8, zorder=1)
        # 实体
        ax.bar(i, body_height, bottom=body_bottom, width=width_ratio,
               color=color, edgecolor=color, linewidth=0.5, zorder=2)

    # x 轴刻度: 均匀取约12个标签
    step = max(1, n // 12)
    tick_pos = list(range(0, n, step))
    if (n - 1) not in tick_pos:
        tick_pos.append(n - 1)
    total_days = (df.index[-1] - df.index[0]).days if n > 1 else 365
    if total_days <= 180:
        tick_lbl = [df.index[i].strftime('%m-%d') for i in tick_pos]       # 短区间: 月-日
    else:
        tick_lbl = [df.index[i].strftime('%Y-%m') for i in tick_pos]       # 长区间: 年-月
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, ha='right', fontsize=8)
    ax.set_xlim(-1, n)

    return d2x
