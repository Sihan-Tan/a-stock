# -*- coding: utf-8 -*-
"""
贵州茅台（600519.SH）单因子分析脚本

本脚本是"机器学习因子挖掘"系列的第一步，目的是：
  1. 展示完整的因子分类体系（6 大类 50+ 因子）
  2. 对贵州茅台进行单因子检验（RankIC 分析）
  3. 判断哪些因子对茅台有预测能力

什么是 RankIC？
  RankIC（Rank Information Coefficient）是因子评价的核心指标：
    - 计算因子值与未来收益的 Spearman 秩相关系数
    - > 0：因子正向预测未来收益（因子值越大，未来涨得越多）
    - < 0：因子负向预测未来收益（因子值越大，未来跌得越多）
    - |IC| > 0.05：强有效因子，|IC| > 0.03：中等有效

为什么只分析茅台这一只股票？
  单只股票的因子分析可以帮助我们理解：
    - 哪些类型的因子对该股票最有效
    - 因子的稳定性和一致性
  但实战中我们会对全市场股票做截面分析。

运行方式: python 1-贵州茅台因子分析.py

依赖:
  - data_loader.py: 加载日K线和财务数据
  - feature_engine.py: 计算技术因子和基本面因子
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr

from data_loader import load_stock_data, load_financial_data
from feature_engine import (
    FACTOR_TAXONOMY,
    calc_features,
    calc_fundamental_features,
    get_all_feature_cols,
)


def calc_rank_ic(factor_values, forward_returns):
    """
    计算单因子 RankIC。

    RankIC = Spearman 秩相关系数（因子值, 未来收益率）

    为什么用 Spearman 而不是 Pearson？
      - Spearman 基于秩（排名），不要求线性关系
      - 对异常值不敏感（极端价格不会扭曲结果）
      - 更适合因子评估（我们只关心排序能力，不关心精确数值）

    参数:
        factor_values: 因子值序列（如 RSI, MACD 等）
        forward_returns: 未来收益率序列（N天后的涨跌幅）

    返回:
        float: RankIC 值，有效样本 < 30 时返回 NaN
    """
    valid = pd.DataFrame({
        'factor': factor_values,
        'fwd_ret': forward_returns
    }).dropna()
    if len(valid) < 30:
        return np.nan
    ic, _ = spearmanr(valid['factor'], valid['fwd_ret'])
    return ic


if __name__ == '__main__':

    # ============================================================
    # 配置区（可以根据需要修改）
    # ============================================================
    STOCK_CODE = '600519.SH'          # 贵州茅台，A股第一高价股
    START_DATE = '2024-01-01'         # 数据起始日期
    END_DATE = '2025-12-31'           # 数据结束日期

    # ============================================================
    # 第1步：展示因子分类体系（FACTOR_TAXONOMY）
    # ============================================================
    # 这步只是为了教学展示，让我们清楚知道有哪些因子可用
    print('\n[1] FACTOR_TAXONOMY')
    total_features = 0
    for cat_key, cat_info in FACTOR_TAXONOMY.items():
        n = len(cat_info['features'])
        total_features += n
        feat_str = ', '.join(cat_info['features'][:5])
        if n > 5:
            feat_str += f' ... (共{n}个)'
        print(f"  {cat_info['name']} ({cat_key}): {n} 个 | {feat_str}")
    print(f'  技术因子合计: {total_features} | 课件中另含 4 个基本面定义')

    # ============================================================
    # 第2步：加载日K线数据 + 计算技术因子
    # ============================================================
    print(f'\n[2] load_stock_data + calc_features | {STOCK_CODE} {START_DATE}~{END_DATE}')
    df = load_stock_data(STOCK_CODE, START_DATE, END_DATE)
    print(f'  交易日: {len(df)} | {df.index[0].strftime("%Y-%m-%d")} ~ {df.index[-1].strftime("%Y-%m-%d")}')
    print(f'  close: {df["close"].min():.2f} ~ {df["close"].max():.2f}')

    # calc_features: 从原始 OHLCV 计算 50+ 技术因子
    df = calc_features(df)
    tech_cols = get_all_feature_cols()
    available_tech = [c for c in tech_cols if c in df.columns]
    print(f'  技术因子列数: {len(available_tech)}')

    # 构建未来1日收益率（预测目标）
    # shift(-1)：把明天的收益率拉到今天这一行
    # 这样今天的因子值就对应明天的收益率，形成预测关系
    df['fwd_ret_1d'] = df['close'].pct_change(1).shift(-1)

    # ============================================================
    # 第3步：加载财务数据 + 计算基本面因子（可选）
    # ============================================================
    print('\n[3] load_financial_data + calc_fundamental_features')
    fundamental_cols = []
    try:
        fin_df = load_financial_data(report_date_min='2022-01-01')
        if fin_df.empty:
            print('  财务表为空, 跳过')
        else:
            print(f'  财务记录: {len(fin_df)} | 股票数: {fin_df["stock_code"].nunique()}')
            fund_features = calc_fundamental_features(df, fin_df, STOCK_CODE)
            for col in fund_features.columns:
                df[col] = fund_features[col]
                if df[col].notna().sum() > 0:
                    fundamental_cols.append(col)
            print(f'  合并基本面列: {fundamental_cols}')
    except Exception as e:
        print(f'  异常: {e}')

    # ============================================================
    # 第4步：行业 One-Hot 编码演示
    # ============================================================
    # one-hot 编码将行业分类转化为数值特征
    # 为什么需要？ML 模型不能直接处理"食品饮料""银行"这样的文本分类
    # 通过 one-hot，每个行业成为一个 0/1 列
    print('\n[4] 行业哑变量示例 get_dummies')
    industry_demo = pd.DataFrame({
        'stock_code': ['600519.SH', '000858.SZ', '601318.SH', '600036.SH', '000001.SZ'],
        'stock_name': ['贵州茅台', '五粮液', '中国平安', '招商银行', '平安银行'],
        'industry': ['食品饮料', '食品饮料', '非银金融', '银行', '银行'],
    })
    industry_dummies = pd.get_dummies(industry_demo['industry'], prefix='ind')
    demo_out = pd.concat([industry_demo[['stock_code', 'stock_name']], industry_dummies], axis=1)
    print(demo_out.to_string(index=False))
    print(f'  哑变量列数: {industry_dummies.shape[1]}')

    # ============================================================
    # 第5步：单因子 RankIC 检验
    # ============================================================
    # 遍历所有因子（技术和基本面），计算每个因子与未来收益的 RankIC
    # 这是因子筛选的核心步骤：IC 越高的因子越有价值
    print('\n[5] 单因子 RankIC vs fwd_ret_1d')
    all_factor_cols = available_tech + fundamental_cols
    ic_results = []
    for col in all_factor_cols:
        ic_val = calc_rank_ic(df[col], df['fwd_ret_1d'])
        if np.isnan(ic_val):
            continue
        # 找到这个因子属于哪个类别
        cat_name = '基本面'
        for cat_key, cat_info in FACTOR_TAXONOMY.items():
            if col in cat_info['features']:
                cat_name = cat_info['name']
                break
        ic_results.append({
            'factor': col,
            'category': cat_name,
            'RankIC': round(ic_val, 4),
            '|IC|': round(abs(ic_val), 4),  # 取绝对值，正负都表示有预测能力
        })

    # 按 |IC| 降序排列，IC 绝对值越大的因子排在越前面
    ic_df = pd.DataFrame(ic_results).sort_values('|IC|', ascending=False)
    ic_df = ic_df.reset_index(drop=True)
    ic_df.index = ic_df.index + 1
    ic_df.index.name = '排名'

    # ============================================================
    # 第6步：输出汇总结果
    # ============================================================
    print(f'\n[6] 完成检验因子数: {len(ic_df)} | {STOCK_CODE}')
    top_n = min(15, len(ic_df))
    print('\nTOP 15 by |IC|')
    print(ic_df.head(top_n).to_string())

    # 按 IC 强度分档
    # 评判标准（华泰证券经验值）：
    #   |IC| >= 0.05：强有效因子，可直接用于选股
    #   0.03 <= |IC| < 0.05：中等有效，可组合使用
    #   0.02 <= |IC| < 0.03：弱有效，需要谨慎
    #   |IC| < 0.02：无效，对预测几乎没有帮助
    strong = ic_df[ic_df['|IC|'] >= 0.05]
    effective = ic_df[(ic_df['|IC|'] >= 0.03) & (ic_df['|IC|'] < 0.05)]
    weak = ic_df[(ic_df['|IC|'] >= 0.02) & (ic_df['|IC|'] < 0.03)]
    ineffective = ic_df[ic_df['|IC|'] < 0.02]
    print(f'\n分档: >=0.05={len(strong)} | [0.03,0.05)={len(effective)} | '
          f'[0.02,0.03)={len(weak)} | <0.02={len(ineffective)}')

    # 按类别汇总：哪类因子整体表现更好？
    # 这可以帮助我们了解，对于贵州茅台来说，
    # 是动量因子更有效，还是技术指标因子更有效
    cat_ic = ic_df.groupby('category')['|IC|'].agg(['mean', 'max', 'count'])
    cat_ic.columns = ['平均|IC|', '最大|IC|', '因子数']
    cat_ic = cat_ic.sort_values('平均|IC|', ascending=False)
    print('\n按类别 |IC|')
    print(cat_ic.to_string())
