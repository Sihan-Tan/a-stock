# -*- coding: utf-8 -*-
"""
chan.py 封装模块

将开源缠论库 chan.py 的 CChan 接口封装成统一的数据结构,
方便教学脚本调用和可视化对比。

为什么需要这个封装:
  chan.py 是一个功能完整的开源缠论实现库, 支持:
    - K线包含处理、分型识别、笔识别、线段识别、中枢识别、买卖点检测
    - 严格笔/推笔模式、多周期分析、MACD背驰判断

  但与我们的自研 ChanAnalyzer 有以下差异:
    - chan.py 的数据结构更复杂 (CChan → CKLine → CBi → CZS → CBsp)
    - 我们封装成统一的 dict 格式, 便于教学脚本使用
    - 同时保留了一些 chan.py 特有的字段 (is_sure, macd_area, seg_idx)

封装的内容:
  - run_chan(): 对 DataFrame 运行 chan.py 分析, 返回统一的 dict
  - draw_chan_chart(): 在 Axes 上绘制 chan.py 结果的蜡烛图

依赖:
  需要安装 chan.py: pip install chan-py
  (非必须: 如果没有安装, 相关脚本会报 ImportError, 不影响自研模块)
"""

import sys
import os
import pandas as pd
import numpy as np

# chan.py 库的路径: 假设在项目根目录下的 chan.py 目录中
CHAN_PY_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'chan.py'))


def _ensure_path():
    """确保 chan.py 所在的路径在 sys.path 中"""
    if CHAN_PY_PATH not in sys.path:
        sys.path.insert(0, CHAN_PY_PATH)


def _ts(t):
    """
    将 chan.py 的时间对象转换为 pandas Timestamp

    chan.py 使用自己的时间对象 (year/month/day 属性),
    需要转换为 pandas Timestamp 以便与我们的 DataFrame 索引兼容。
    """
    return pd.Timestamp(f'{t.year:04d}-{t.month:02d}-{t.day:02d}')


def run_chan(df, symbol='stock', config_dict=None):
    """
    对 DataFrame 运行 chan.py 缠论分析

    参数:
        df: DataFrame, 含 open/high/low/close/volume 列, DatetimeIndex
        symbol: 股票代码标识 (用于内部缓存)
        config_dict: CChanConfig 配置字典 (可选)
          常用配置项:
            - 'bi_strict': True 使用严格笔 (默认), False 使用推笔
            - 'gap_as_bi': False 缺口是否当成笔
            - 'seg_algo': 'chan' 线段算法 ('chan' 或 'normal')

    返回:
        dict: 包含以下键:
          - klc_list: 合并K线列表 [{'date', 'high', 'low', 'idx', 'fx', 'raw_count'}]
          - fractals: 分型列表 (从 klc_list 提取的 fx='top'/'bottom' 项)
          - bi_list: 笔列表 [{'start_date', 'end_date', 'direction', ...}]
          - seg_list: 线段列表
          - zs_list: 中枢列表
          - bsp_list: 买卖点列表
          - raw_kl: chan.py 原始对象 (高级用户调试用)
    """
    _ensure_path()
    from DataAPI import DfApi as DfApiModule
    from Common.CEnum import KL_TYPE
    from Chan import CChan
    from ChanConfig import CChanConfig

    # 将 DataFrame 存入 chan.py 的缓存, 供其内部 API 读取
    DfApiModule._DF_CACHE[symbol] = df
    config = CChanConfig(config_dict or {})

    # 创建 CChan 实例, 指定数据源为自定义的 DfApi
    # lv_list=[K_DAY] 表示使用日线级别
    chan = CChan(
        code=symbol,
        data_src='custom:DfApi.DfApi',
        lv_list=[KL_TYPE.K_DAY],
        config=config,
    )
    kl = chan[0]  # 获取第一个级别 (日线) 的分析结果

    # ==================== 提取合并K线 (KLC) ====================
    klc_list = []
    for klc in kl.lst:  # kl.lst 是所有合并K线 (KLine_Unit) 的列表
        t0 = klc.lst[0].time
        # 读取分型标记: chan.py 的每根合并K线自带 fx 属性
        fx_type = 'unknown'
        if hasattr(klc, 'fx') and klc.fx is not None:
            fn = klc.fx.name if hasattr(klc.fx, 'name') else str(klc.fx)
            if 'TOP' in fn:
                fx_type = 'top'
            elif 'BOTTOM' in fn:
                fx_type = 'bottom'
        klc_list.append({
            'date': _ts(t0),
            'high': float(klc.high),
            'low': float(klc.low),
            'idx': klc.idx,       # 索引位置
            'fx': fx_type,        # 分型类型
            'raw_count': len(klc.lst),  # 该合并K线由多少根原始K线合并而成
        })

    # ==================== 提取分型 ====================
    fractals = [k for k in klc_list if k['fx'] in ('top', 'bottom')]

    # ==================== 提取笔 ====================
    bi_list = []
    for bi in kl.bi_list:
        is_up = bi.dir.name == 'UP'
        bklc = bi.begin_klc  # 笔的起始K线
        eklc = bi.end_klc    # 笔的结束K线
        bt = _ts(bklc.lst[0].time)
        et = _ts(eklc.lst[-1].time)

        # 确定笔的起止价格:
        # 上升笔: 从底分型的低点 → 顶分型的高点
        # 下降笔: 从顶分型的高点 → 底分型的低点
        if is_up:
            sp, ep = float(bklc.low), float(eklc.high)
        else:
            sp, ep = float(bklc.high), float(eklc.low)

        # 获取精确日期: 在原始K线中找到实际最高/最低日的日期
        # 这比合并K线的边界更精确, 用于图表上精确定位
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
            'start_raw_date': start_raw,
            'end_raw_date': end_raw,
            'start_price': sp,
            'end_price': ep,
            'direction': 'up' if is_up else 'down',
            'is_sure': bi.is_sure if hasattr(bi, 'is_sure') else True,  # 是否已确认 (非虚拟结束)
            'klc_count': bi.get_klc_cnt() if hasattr(bi, 'get_klc_cnt') else 0,  # 包含合并K线数
            'klu_count': bi.get_klu_cnt() if hasattr(bi, 'get_klu_cnt') else 0,  # 包含原始K线数
            'seg_idx': bi.seg_idx if hasattr(bi, 'seg_idx') else -1,  # 所属线段索引
        }

        # MACD 面积: chan.py 内置的计算, 用于背驰判断
        try:
            bi_item['macd_area'] = float(bi.Cal_MACD_area())
        except Exception:
            bi_item['macd_area'] = 0.0

        # 买卖点: chan.py 的笔可能自带买卖点标记
        if bi.bsp:
            bsp_types = [str(t.value) if hasattr(t, 'value') else str(t) for t in bi.bsp.type]
            bi_item['bsp_type'] = ','.join(bsp_types)
            bi_item['bsp_is_buy'] = bi.bsp.is_buy
            bi_item['bsp_date'] = _ts(bi.bsp.klu.time) if hasattr(bi.bsp, 'klu') else et
        else:
            bi_item['bsp_type'] = None
            bi_item['bsp_is_buy'] = None
            bi_item['bsp_date'] = None

        bi_list.append(bi_item)

    # ==================== 提取线段 ====================
    seg_list = []
    for seg in kl.seg_list:
        bt = _ts(seg.start_bi.begin_klc.lst[0].time)
        et = _ts(seg.end_bi.end_klc.lst[-1].time)
        is_up = seg.dir.name == 'UP'

        # 线段内部的中枢
        seg_zs_list = []
        if hasattr(seg, 'zs_lst'):
            for zs in seg.zs_lst:
                seg_zs_list.append({
                    'ZG': float(zs.high),
                    'ZD': float(zs.low),
                })

        seg_list.append({
            'start_date': bt,
            'end_date': et,
            'direction': 'up' if is_up else 'down',
            'is_sure': seg.is_sure if hasattr(seg, 'is_sure') else True,
            'bi_count': seg.cal_bi_cnt() if hasattr(seg, 'cal_bi_cnt') else 0,
            'zs_list': seg_zs_list,
        })

    # ==================== 提取中枢 ====================
    zs_list = []
    for zs in kl.zs_list:
        # 获取中枢起止日期: chan.py 的中枢对象可能通过不同属性访问
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
            'ZG': float(zs.high),
            'ZD': float(zs.low),
            'center': float(zs.mid) if hasattr(zs, 'mid') else (float(zs.high) + float(zs.low)) / 2,
            'start_date': begin_t,
            'end_date': end_t,
            'peak_high': float(zs.peak_high) if hasattr(zs, 'peak_high') else float(zs.high),
            'peak_low': float(zs.peak_low) if hasattr(zs, 'peak_low') else float(zs.low),
        })

    # ==================== 提取买卖点汇总 ====================
    # 从笔列表筛选出带有买卖点标记的
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
        'raw_kl': kl,
    }


def draw_chan_chart(ax, df, chan_data, show_bi=True, show_seg=True,
                    show_zs=True, show_bsp=True, show_fractals=False):
    """
    在指定的 matplotlib Axes 上绘制 chan.py 分析结果的蜡烛图

    这个函数复用了 ChanAnalyzer._draw_candlestick() 绘制K线,
    然后在其基础上叠加 chan.py 的分析结果 (笔/线段/中枢/买卖点)。

    参数:
        ax: matplotlib Axes 对象
        df: 原始 DataFrame (用于绘制K线)
        chan_data: run_chan() 返回的结果字典
        show_bi: 是否显示笔
        show_seg: 是否显示线段
        show_zs: 是否显示中枢
        show_bsp: 是否显示买卖点
        show_fractals: 是否显示分型

    返回:
        date_to_x: dict, 日期到 x 位置的映射 (用于在上层继续添加元素)
    """
    import matplotlib.patches as patches

    sys.path.insert(0, os.path.dirname(__file__))
    from chan_analyzer import ChanAnalyzer
    d2x = ChanAnalyzer._draw_candlestick(ax, df, width_ratio=0.6)

    def _lx(date_val):
        """查找日期对应的 x 位置 (支持多种日期格式)"""
        if date_val in d2x:
            return d2x[date_val]
        if hasattr(date_val, 'date'):
            return d2x.get(date_val.date())
        for k, v in d2x.items():
            if hasattr(k, 'date') and k.date() == date_val:
                return v
        return None

    # --- 分型 ---
    if show_fractals and chan_data['fractals']:
        for f in chan_data['fractals']:
            x = _lx(f['date'])
            if x is None:
                continue
            if f['fx'] == 'top':
                ax.scatter(x, f['high'], marker='v', color='#e74c3c', s=50, zorder=5, alpha=0.6)
            else:
                ax.scatter(x, f['low'], marker='^', color='#2ecc71', s=50, zorder=5, alpha=0.6)

    # --- 笔 ---
    if show_bi and chan_data['bi_list']:
        for bi in chan_data['bi_list']:
            c = '#e74c3c' if bi['direction'] == 'up' else '#27ae60'
            sx = _lx(bi['start_raw_date'])
            ex = _lx(bi['end_raw_date'])
            if sx is not None and ex is not None:
                ax.plot([sx, ex], [bi['start_price'], bi['end_price']],
                        color=c, linewidth=1.8, alpha=0.85, zorder=4)

    # --- 线段 ---
    # 线段是多笔构成的更大级别走势, 用粗虚线绘制
    if show_seg and chan_data['seg_list']:
        seg_colors = ['#9b59b6', '#e67e22', '#1abc9c', '#e74c3c']
        for si, seg in enumerate(chan_data['seg_list']):
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
                    dir_name = '上升' if seg['direction'] == 'up' else '下降'
                    ax.plot([sx, ex], [sp, ep],
                            color=sc, linewidth=3.5, linestyle='--', alpha=0.7, zorder=3)
                    mid_x = (sx + ex) / 2
                    mid_y = (sp + ep) / 2
                    ax.annotate(
                        f'线段{si+1}({dir_name},{seg["bi_count"]}笔)',
                        (mid_x, mid_y), fontsize=9, fontweight='bold', color=sc,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=sc, alpha=0.85),
                        ha='center', va='center', zorder=8)

    # --- 中枢 ---
    if show_zs and chan_data['zs_list']:
        for zs in chan_data['zs_list']:
            if zs['start_date'] and zs['end_date']:
                xl = _lx(zs['start_date'])
                xr = _lx(zs['end_date'])
                if xl is not None and xr is not None:
                    rect = patches.Rectangle(
                        (xl, zs['ZD']),
                        xr - xl,
                        zs['ZG'] - zs['ZD'],
                        linewidth=1.5, edgecolor='#3498db', facecolor='#3498db',
                        alpha=0.15, zorder=2,
                    )
                    ax.add_patch(rect)
                    ax.text(xl, zs['ZG'],
                            f" ZG={zs['ZG']:.1f}\n ZD={zs['ZD']:.1f}",
                            fontsize=7, color='#2c3e50', va='bottom')

    # --- 买卖点 ---
    if show_bsp and chan_data['bsp_list']:
        bsp_names = {'1': '一买', '2': '二买', '2s': '二买S', '3': '三买',
                     '1s': '一卖', '2_sell': '二卖', '3_sell': '三卖'}
        bsp_colors = {'1': '#8e44ad', '2': '#e67e22', '2s': '#f39c12',
                      '3': '#e74c3c', '1s': '#2c3e50', '2_sell': '#27ae60',
                      '3_sell': '#16a085'}
        for bi in chan_data['bsp_list']:
            bsp_type = bi['bsp_type']
            is_buy = bi['bsp_is_buy']
            x = _lx(bi['bsp_date'])
            if x is None:
                continue
            marker = '^' if is_buy else 'v'
            color = bsp_colors.get(bsp_type, '#333333')
            name = bsp_names.get(bsp_type, bsp_type)
            price = bi['end_price']
            ax.scatter(x, price, marker=marker, color=color,
                       s=200, zorder=7, edgecolors='black', linewidths=1)
            offset_y = 10 if is_buy else -15
            ax.annotate(name, (x, price),
                        textcoords="offset points", xytext=(10, offset_y),
                        fontsize=9, fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    return d2x
