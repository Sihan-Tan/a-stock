# -*- coding: utf-8 -*-
"""
工业级特征工程流程演示脚本

本脚本展示了完整的特征工程流水线，从原始数据到可用于模型训练的特征矩阵。

完整流程：
  第1步：批量加载多只股票的日K线数据
  第2步：对每只股票计算 50+ 技术特征（6 大类）
  第3步：加载财务数据，补充基本面因子
  第4步：构造行业因子（one-hot 编码）
  第5步：华泰标准预处理流水线（MAD 去极值 + Z-score 标准化）
  第6步：行业市值中性化
  第7步：特征相关性分析，识别冗余特征

教学目的：
  1. 理解特征工程在量化投资中的重要性
  2. 掌握 MAD 去极值、Z-score、中性化等技术细节
  3. 学会如何识别和处理冗余特征（多重共线性问题）

什么是"冗余特征"？
  当两个特征高度相关时（如 5 日动量与 10 日动量），它们携带的信息高度重叠。
  冗余特征会导致：
    - 模型不稳定（微小的输入变化导致输出大幅变化）
    - 可解释性下降（无法判断哪个特征真正起作用）
    - 训练时间增加（模型需要处理更多特征）

运行方式: python 2-工业级特征工程.py

依赖:
  - data_loader.py: 加载日K线和财务数据
  - feature_engine.py: 计算因子、预处理、中性化
"""

import numpy as np
import pandas as pd
from data_loader import load_stock_data, load_financial_data
from feature_engine import (
    calc_features, calc_fundamental_features,
    preprocess_features, neutralize,
    get_all_feature_cols, FACTOR_TAXONOMY,
)


# ============================================================
# 配置区
# ============================================================

# 目标股票池：涵盖不同行业，增加分析的多样性
# 选择逻辑：大盘蓝筹股，数据质量高，代表不同行业
STOCK_POOL = [
    '600519.SH',   # 贵州茅台     - 食品饮料（消费龙头）
    '688981.SH',   # 中芯国际     - 半导体（科技龙头）
    '000001.SZ',   # 平安银行     - 银行（金融）
    '159941.SZ',   # 纳指ETF     - 指数基金（跨境投资）
    '300750.SZ',   # 宁德时代     - 新能源（电池龙头）
]

START_DATE = '2023-01-01'
END_DATE = '2025-12-31'

# 行业映射表（手动标注，实战中可以从数据库获取）
INDUSTRY_MAP = {
    '600519.SH': '食品饮料',
    '688981.SH': '半导体',
    '000001.SZ': '银行',
    '159941.SZ': '指数基金',
    '300750.SZ': '新能源',
}


def print_section(title):
    """打印分节标题（美化输出用）"""
    width = 60
    print('\n' + '=' * width)
    print(f'  {title}')
    print('=' * width)


# ============================================================
# 第1步：批量加载数据
# ============================================================

def step1_load_data():
    """
    逐只加载目标股票的日K线数据。

    为什么逐只加载而不是批量加载？
      - 演示目的：展示单只股票的加载过程
      - 实际生产环境中，batch_load_daily 更高效
    """
    print_section('第1步: 加载目标股票日K线')

    target_stocks = {}
    for code in STOCK_POOL:
        try:
            df = load_stock_data(code, START_DATE, END_DATE)
            if len(df) >= 120:
                target_stocks[code] = df
        except Exception as e:
            print(f'  [跳过] {code}: {e}')

    print(f'成功加载 {len(target_stocks)} 只股票')
    for code, df in target_stocks.items():
        name = INDUSTRY_MAP.get(code, '未知')
        print(f'  {code} ({name}): {len(df)} 个交易日, '
              f'{df.index[0].strftime("%Y-%m-%d")} ~ {df.index[-1].strftime("%Y-%m-%d")}')

    if len(target_stocks) == 0:
        print('[警告] 未加载到任何目标股票, 请检查数据库数据')

    return target_stocks


# ============================================================
# 第2步：计算 50+ 技术特征
# ============================================================

def step2_calc_technical_features(stock_data):
    """
    对每只股票计算 50+ 技术特征。

    这一步是最耗时的特征工程步骤，在一个快速循环中完成。
    计算结果被缓存，不会重复计算（除非数据改变）。
    """
    print_section('第2步: 计算50+技术特征 (calc_features)')

    featured_data = {}
    for code, df in stock_data.items():
        df_feat = calc_features(df)
        featured_data[code] = df_feat

    all_feature_cols = get_all_feature_cols()
    sample_code = list(featured_data.keys())[0]
    sample_df = featured_data[sample_code]
    available_features = [c for c in all_feature_cols if c in sample_df.columns]

    # 展示因子分类体系
    print(f'因子分类体系 ({len(FACTOR_TAXONOMY)} 大类):')
    total_count = 0
    for cat_key, cat_info in FACTOR_TAXONOMY.items():
        n = len(cat_info['features'])
        total_count += n
        print(f'  {cat_info["name"]} ({cat_key}): {n} 个因子')
        features_str = ', '.join(cat_info['features'][:4])
        if n > 4:
            features_str += f' ... 共{n}个'
        print(f'    -> {features_str}')

    print(f'\n因子总数: {total_count} 个, 实际可用: {len(available_features)} 个')

    print(f'\n以 {sample_code} 为例, 最近5行部分特征:')
    show_cols = ['close', 'ret_1d', 'momentum_20d', 'rsi_14', 'macd_hist', 'ma20_bias']
    show_cols = [c for c in show_cols if c in sample_df.columns]
    print(sample_df[show_cols].tail(5).to_string())

    return featured_data


# ============================================================
# 第3步：加载财务数据，补充基本面因子
# ============================================================

def step3_add_fundamental(featured_data):
    """
    加载财务数据，为每只股票添加基本面因子。

    基本面因子的特点：
      - 更新频率低（每季度一次）
      - 但"信号半衰期"长（一个季度的财务数据对未来数月都有影响）
      - 与技术因子互补（技术因子捕捉短期情绪，基本面因子捕捉长期价值）
    """
    print_section('第3步: 加载财务数据, 补充基本面因子')

    fin_df = load_financial_data(report_date_min='2022-01-01')
    if fin_df.empty:
        print('[警告] 财务数据为空, 跳过基本面因子计算')
        return featured_data

    print(f'加载财务数据: {len(fin_df)} 条记录, '
          f'覆盖 {fin_df["stock_code"].nunique()} 只股票')

    fundamental_cols = ['pe_ratio', 'roe_factor', 'gross_margin_factor', 'debt_ratio_factor']

    for code, df in featured_data.items():
        fund_df = calc_fundamental_features(df, fin_df, code)
        for col in fundamental_cols:
            if col in fund_df.columns:
                featured_data[code][col] = fund_df[col]

    sample_code = list(featured_data.keys())[0]
    sample_df = featured_data[sample_code]
    avail_fund = [c for c in fundamental_cols if c in sample_df.columns]
    print(f'\n基本面因子: {avail_fund}')
    if avail_fund:
        print(f'以 {sample_code} 为例, 最近5行:')
        print(sample_df[avail_fund].tail(5).to_string())

    return featured_data


# ============================================================
# 第4步：构造行业因子（one-hot 编码）
# ============================================================

def step4_industry_factors(featured_data):
    """
    构造行业哑变量（one-hot 编码）。

    为什么需要行业哑变量？
      - ML 模型无法直接处理"食品饮料""银行"这样的文本分类
      - one-hot 编码将行业转化为数值特征（每行业一列，0/1）
      - 后续中性化需要行业哑变量来消除行业偏见

    行业哑变量的维度 = 行业数量
    例如：['食品饮料', '半导体', '银行', ...] -> 5 列 0/1 数据
    """
    print_section('第4步: 构造行业因子 (one-hot编码)')

    print('行业映射:')
    for code, industry in INDUSTRY_MAP.items():
        print(f'  {code} -> {industry}')

    industries = sorted(set(INDUSTRY_MAP.values()))
    print(f'\n行业列表: {industries}')

    # 为每只股票生成数据（含行业哑变量），然后合并
    all_frames = []
    for code, df in featured_data.items():
        tmp = df.copy()
        tmp['stock_code'] = code
        tmp['trade_date'] = tmp.index
        industry = INDUSTRY_MAP.get(code, '其他')
        for ind in industries:
            tmp[f'ind_{ind}'] = 1.0 if ind == industry else 0.0
        all_frames.append(tmp)

    merged = pd.concat(all_frames, ignore_index=True)
    ind_cols = [f'ind_{ind}' for ind in industries]

    print(f'\n合并后数据: {merged.shape[0]} 行 x {merged.shape[1]} 列')
    print(f'行业哑变量列: {ind_cols}')

    print('\n行业分布:')
    for ind in industries:
        col = f'ind_{ind}'
        count = int(merged[col].sum())
        print(f'  {ind}: {count} 条记录')

    return merged, ind_cols, industries


# ============================================================
# 第5步：华泰标准预处理流水线
# ============================================================

def step5_preprocess(merged):
    """
    MAD 去极值 + Z-score 标准化。

    预处理的目标：
      1. 消除极端值（如因数据错误导致的离谱价格）
      2. 统一量纲（让 PE=100 和 RSI=50 在模型中可比）
      3. 处理缺失值

    预处理效果验证：
      - 去极值前：min 和 max 可能异常大/小
      - 去极值后：极端值被截断到合理范围
      - 标准化后：均值接近 0，标准差接近 1
    """
    print_section('第5步: 华泰标准预处理 (MAD去极值 + Z-score标准化)')

    feature_cols = get_all_feature_cols()
    feature_cols = [c for c in feature_cols if c in merged.columns]

    demo_factors = ['momentum_20d', 'rsi_14', 'macd_hist']
    demo_factors = [f for f in demo_factors if f in merged.columns]

    print('--- 去极值前的分布统计 ---')
    for col in demo_factors:
        s = merged[col].dropna()
        print(f'  {col:20s}: min={s.min():10.4f}  max={s.max():10.4f}  '
              f'mean={s.mean():10.4f}  std={s.std():10.4f}')

    preprocessed = merged.copy()
    # 对每只股票独立预处理（同一只股票内去极值和标准化）
    for code in merged['stock_code'].unique():
        mask = preprocessed['stock_code'] == code
        stock_df = preprocessed.loc[mask].copy()

        stock_df_processed = preprocess_features(stock_df, feature_cols=feature_cols, method='mad')

        for col in feature_cols:
            preprocessed.loc[mask, col] = stock_df_processed[col].values

    print('\n--- 去极值+标准化后的分布统计 ---')
    for col in demo_factors:
        s = preprocessed[col].dropna()
        print(f'  {col:20s}: min={s.min():10.4f}  max={s.max():10.4f}  '
              f'mean={s.mean():10.4f}  std={s.std():10.4f}')

    print('\n预处理效果:')
    print('  - 极端值被MAD方法截断 (中位数 +/- 5*1.4826*MAD)')
    print('  - 标准化后均值接近0, 标准差接近1')
    print('  - 不同因子量纲统一, 可直接输入模型')

    return preprocessed, feature_cols


# ============================================================
# 第6步：行业市值中性化
# ============================================================

def step6_neutralize(preprocessed, ind_cols):
    """
    行业市值中性化。

    原理：对因子值做回归，剔除行业和市值的影响。
    factor = beta_industry * industry + beta_mktcap * ln(mktcap) + residual
    残差 residual 就是"纯 alpha"，不受行业和市值偏见影响。

    为什么需要中性化？
      假设某因子在"食品饮料"行业天然高，在"银行"行业天然低。
      模型可能会学到"买入食品饮料行业"而不是真正的因子信号。
      中性化后，模型看到的是"该股票在行业内的相对强弱"。
    """
    print_section('第6步: 行业市值中性化')

    print('原理: factor = beta_industry * industry + beta_mktcap * ln(mktcap) + residual')
    print('残差residual即为中性化后的因子值, 消除了行业和市值的影响\n')

    # 用近20日均价 * 近20日均量作为市值代理变量
    # 注意：这不是真实市值，但可以反映股票的市场关注度和流动性
    preprocessed['mktcap_proxy'] = (
        preprocessed['close'] * preprocessed['volume']
    ).rolling(20, min_periods=1).mean()
    preprocessed['mktcap_log'] = np.log(preprocessed['mktcap_proxy'].clip(lower=1))

    industry_dummies = preprocessed[ind_cols]
    mktcap_log = preprocessed['mktcap_log']

    target_factor = 'momentum_20d'
    if target_factor not in preprocessed.columns:
        print(f'[警告] {target_factor} 不在数据中, 跳过中性化演示')
        return preprocessed

    factor_before = preprocessed[target_factor].copy()

    factor_neutralized = neutralize(
        factor_series=preprocessed[target_factor],
        industry_dummies=industry_dummies,
        mktcap_log=mktcap_log,
    )
    preprocessed[f'{target_factor}_neutral'] = factor_neutralized

    print(f'因子: {target_factor}')
    print('\n--- 中性化前 ---')
    before_stats = factor_before.dropna()
    print(f'  mean={before_stats.mean():.6f}  std={before_stats.std():.6f}  '
          f'min={before_stats.min():.6f}  max={before_stats.max():.6f}')

    print('\n--- 中性化后 ---')
    after_stats = factor_neutralized.dropna()
    print(f'  mean={after_stats.mean():.6f}  std={after_stats.std():.6f}  '
          f'min={after_stats.min():.6f}  max={after_stats.max():.6f}')

    print('\n各行业因子均值对比:')
    print(f'  {"行业":<10s} {"中性化前":>12s} {"中性化后":>12s}')
    print(f'  {"-"*10} {"-"*12} {"-"*12}')
    for col in ind_cols:
        ind_name = col.replace('ind_', '')
        mask = preprocessed[col] == 1.0
        if mask.sum() == 0:
            continue
        mean_before = factor_before.loc[mask].mean()
        mean_after = factor_neutralized.loc[mask].mean()
        print(f'  {ind_name:<10s} {mean_before:>12.6f} {mean_after:>12.6f}')

    print('\n中性化效果: 消除行业间的因子均值差异, 使因子反映个股相对行业的超额信息')

    return preprocessed


# ============================================================
# 第7步：特征相关性分析
# ============================================================

def step7_correlation_analysis(preprocessed, feature_cols):
    """
    计算特征间相关系数，识别冗余特征。

    多重共线性问题：
      当两个特征高度相关时，它们携带几乎相同的信息。
      在模型中同时使用它们会导致：
        - 模型权重不稳定
        - 可解释性降低
        - 对训练数据的微小变化非常敏感

    处理方法：
      对于 |corr| > 0.8 的特征对，可以考虑：
        - 删除其中一个（通常保留与目标 IC 更高的那个）
        - 用 PCA 降维
        - 用正则化自动处理（如 L1 正则化）
    """
    print_section('第7步: 特征相关性分析')

    avail_cols = [c for c in feature_cols if c in preprocessed.columns]
    corr_matrix = preprocessed[avail_cols].corr()

    print(f'计算 {len(avail_cols)} 个特征的相关系数矩阵: {corr_matrix.shape}')

    threshold = 0.8  # 相关性阈值：|r| > 0.8 视为高相关
    high_corr_pairs = []
    for i in range(len(avail_cols)):
        for j in range(i + 1, len(avail_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > threshold:
                high_corr_pairs.append((avail_cols[i], avail_cols[j], corr_val))

    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f'\n高相关特征对 (|corr| > {threshold}): 共 {len(high_corr_pairs)} 对')
    print(f'  {"特征A":<25s} {"特征B":<25s} {"相关系数":>10s}')
    print(f'  {"-"*25} {"-"*25} {"-"*10}')

    show_limit = 20
    for feat_a, feat_b, corr_val in high_corr_pairs[:show_limit]:
        print(f'  {feat_a:<25s} {feat_b:<25s} {corr_val:>10.4f}')
    if len(high_corr_pairs) > show_limit:
        print(f'  ... 共 {len(high_corr_pairs)} 对, 仅展示前 {show_limit} 对')

    # 对于每对高相关特征，保留一个（通常是第一个），
    # 将另一个标记为冗余
    redundant_features = set()
    for feat_a, feat_b, _ in high_corr_pairs:
        redundant_features.add(feat_b)

    print(f'\n冗余特征建议 (可考虑剔除): {len(redundant_features)} 个')
    if redundant_features:
        for feat in sorted(redundant_features):
            print(f'  - {feat}')

    print(f'\n保留后特征数: {len(avail_cols) - len(redundant_features)} 个 '
          f'(原 {len(avail_cols)} 个)')

    return corr_matrix, high_corr_pairs


# ============================================================
# 教学总结
# ============================================================

def summary():
    """打印特征工程流水线总结"""
    print_section('特征工程流水线总结')

    print('特征工程流水线:')
    print('  原始OHLCV -> 50+因子 -> 去极值 -> 中性化 -> 标准化 -> 建模')
    print()
    print('各环节要点:')
    print('  1. 原始OHLCV: 从数据库批量加载日K线数据')
    print('  2. 50+因子:    calc_features() 计算价量/动量/波动率/技术/均线/交互 6大类因子')
    print('  3. 基本面因子: calc_fundamental_features() 补充PE/ROE/毛利率等')
    print('  4. 行业因子:   构造行业哑变量 (one-hot), 用于后续中性化')
    print('  5. MAD去极值:  中位数 +/- 5*1.4826*MAD 截断, 消除极端离群值')
    print('  6. 中性化:     回归法消除行业和市值对因子的影响')
    print('  7. Z-score:    标准化到均值0/标准差1, 统一量纲')
    print('  8. 相关性分析: 识别冗余特征, 降低多重共线性')


# ============================================================
# 主流程
# ============================================================

def main():
    # 第1步: 加载数据
    stock_data = step1_load_data()
    if not stock_data:
        print('没有加载到数据, 程序退出')
        return

    # 第2步: 计算技术特征
    featured_data = step2_calc_technical_features(stock_data)

    # 第3步: 加载财务数据, 补充基本面因子
    featured_data = step3_add_fundamental(featured_data)

    # 第4步: 构造行业因子
    merged, ind_cols, industries = step4_industry_factors(featured_data)

    # 第5步: 华泰预处理流水线
    preprocessed, feature_cols = step5_preprocess(merged)

    # 第6步: 行业市值中性化
    preprocessed = step6_neutralize(preprocessed, ind_cols)

    # 第7步: 特征相关性分析
    step7_correlation_analysis(preprocessed, feature_cols)

    # 教学总结
    summary()


if __name__ == '__main__':
    main()
