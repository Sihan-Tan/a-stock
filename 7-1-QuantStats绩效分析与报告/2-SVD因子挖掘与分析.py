# -*- coding: utf-8 -*-
"""
3-SVD因子挖掘与分析.py
=========================
AI量化交易课程 - 第13讲: SVD矩阵分解与隐因子挖掘

核心思想:
  Fama-French等资产定价模型认为: R = B * F + e
    R (N x T): N只股票在T个交易日的收益率矩阵
    B (N x K): 因子暴露矩阵 (每只股票对K个隐因子的敏感度)
    F (K x T): 因子收益矩阵 (K个隐因子在T天的收益)
    e (N x T): 残差矩阵 (个股特有风险, 无法被共同因子解释)

  用SVD分解 R = U * Sigma * V^T, 可以在不预设因子的前提下,
  "让数据自己说话", 从收益矩阵中挖掘出隐含的驱动因子。

实战场景:
  1. 收益矩阵SVD分解 -- 发现市场的隐含驱动因子
  2. 奇异值衰减分析 -- 判断"几个因子就够了"
  3. 隐因子解读 -- 将主成分与已知因子(行业/动量/波动率)关联
  4. 行业因子结构 -- 不同行业的因子暴露差异
  5. 滚动窗口SVD -- 不同时间段的驱动因子变化(牛熊市切换)
  6. 因子压缩降维 -- 50+因子 -> 5个主成分, 对比预测效果

运行:
  python 3-SVD因子挖掘与分析.py
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

import numpy as np
import pandas as pd
from scipy import stats
from data_loader import load_stock_data
from feature_engine import calc_features, preprocess_features, get_all_feature_cols, FACTOR_TAXONOMY
from db_config import execute_query

os.makedirs('outputs', exist_ok=True)


# ============================================================
# 股票池定义 (覆盖多行业, 方便观察行业因子结构)
# ============================================================

STOCK_POOL = {
    # 食品饮料
    '600519.SH': ('贵州茅台', '食品饮料'),
    '000858.SZ': ('五粮液',   '食品饮料'),
    '002304.SZ': ('洋河股份', '食品饮料'),
    # 银行
    '000001.SZ': ('平安银行', '银行'),
    '601398.SH': ('工商银行', '银行'),
    '600036.SH': ('招商银行', '银行'),
    # 新能源
    '300750.SZ': ('宁德时代', '新能源'),
    '601012.SH': ('隆基绿能', '新能源'),
    '002459.SZ': ('晶澳科技', '新能源'),
    # 半导体/科技
    '688981.SH': ('中芯国际', '科技'),
    '002371.SZ': ('北方华创', '科技'),
    '688012.SH': ('中微公司', '科技'),
    # 医药
    '600276.SH': ('恒瑞医药', '医药'),
    '300760.SZ': ('迈瑞医疗', '医药'),
    '000538.SZ': ('云南白药', '医药'),
    # ETF/指数
    '159941.SZ': ('纳指ETF', 'ETF'),
}

START_DATE = '2023-01-01'
END_DATE = '2025-12-31'

INDUSTRY_COLORS = {
    '食品饮料': 'tab:red',
    '银行': 'tab:blue',
    '新能源': 'tab:green',
    '科技': 'tab:purple',
    '医药': 'tab:orange',
    'ETF': 'tab:gray',
}


def print_section(title):
    """保留接口供步骤函数调用，不在控制台输出课件式标题。"""
    pass


# ============================================================
# 第1步: 构建收益率矩阵 R (N x T)
# ============================================================

def step1_build_return_matrix():
    """加载多只股票数据, 构建对齐的日收益率矩阵"""
    print_section('第1步: 构建收益率矩阵 R (N x T)')

    returns_dict = {}
    stock_info = {}

    for code, (name, industry) in STOCK_POOL.items():
        try:
            df = load_stock_data(code, START_DATE, END_DATE)
            if len(df) < 200:
                print(f'  [跳过] {code} {name}: 数据不足 ({len(df)}条)')
                continue
            daily_ret = df['close'].pct_change()
            returns_dict[code] = daily_ret
            stock_info[code] = (name, industry)
            print(f'  {code} {name:6s} ({industry:4s}): {len(df)} 个交易日')
        except Exception as e:
            print(f'  [跳过] {code}: {e}')

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna()

    N, T = returns_df.shape[1], returns_df.shape[0]
    print(f'\n  收益率矩阵 R: {N} 只股票 x {T} 个交易日')
    print(f'  时间范围: {returns_df.index[0].strftime("%Y-%m-%d")} '
          f'~ {returns_df.index[-1].strftime("%Y-%m-%d")}')

    return returns_df, stock_info


# ============================================================
# 第2步: SVD分解 + 奇异值衰减分析
# ============================================================

def step2_svd_decomposition(returns_df, stock_info):
    """对收益率矩阵做SVD, 分析奇异值衰减"""
    print_section('第2步: SVD分解 + 奇异值衰减分析')

    R = returns_df.values.T  # (N, T)
    R_centered = R - R.mean(axis=1, keepdims=True)

    U, sigma, Vt = np.linalg.svd(R_centered, full_matrices=False)

    # 方差解释率
    total_var = np.sum(sigma ** 2)
    explained_ratio = sigma ** 2 / total_var
    cumulative_ratio = np.cumsum(explained_ratio)

    print(f'\n  SVD分解完成: R_centered ({R_centered.shape[0]} x {R_centered.shape[1]})')
    print(f'  奇异值个数: {len(sigma)}')
    print(f'\n  奇异值衰减分析:')
    print(f'    {"因子#":>6}  {"奇异值":>10}  {"方差占比":>10}  {"累积占比":>10}')
    print(f'    {"-"*6}  {"-"*10}  {"-"*10}  {"-"*10}')

    show_n = min(10, len(sigma))
    for i in range(show_n):
        print(f'    {i+1:>6}  {sigma[i]:>10.4f}  {explained_ratio[i]:>10.2%}  '
              f'{cumulative_ratio[i]:>10.2%}')

    # 确定有效因子数量 (累积方差>80%)
    n_factors_80 = np.searchsorted(cumulative_ratio, 0.80) + 1
    n_factors_90 = np.searchsorted(cumulative_ratio, 0.90) + 1

    print(f'\n  结论:')
    print(f'    累积方差达80%需要 {n_factors_80} 个因子')
    print(f'    累积方差达90%需要 {n_factors_90} 个因子')
    print(f'    前3个因子解释 {cumulative_ratio[2]:.1%} 的总方差')

    # 可视化
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        ax1.bar(range(1, show_n + 1), explained_ratio[:show_n] * 100, color='steelblue')
        ax1.set_xlabel('Principal Component')
        ax1.set_ylabel('Variance Explained (%)')
        ax1.set_title('SVD Scree Plot')
        ax1.set_xticks(range(1, show_n + 1))

        ax2 = axes[1]
        ax2.plot(range(1, show_n + 1), cumulative_ratio[:show_n] * 100,
                 'o-', color='darkorange', linewidth=2)
        ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80%')
        ax2.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90%')
        ax2.set_xlabel('Number of Components')
        ax2.set_ylabel('Cumulative Variance (%)')
        ax2.set_title('Cumulative Variance Explained')
        ax2.set_xticks(range(1, show_n + 1))
        ax2.legend()

        plt.tight_layout()
        plt.savefig('outputs/svd_scree_plot.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  图表已保存: outputs/svd_scree_plot.png')
    except Exception as e:
        print(f'  [可视化跳过] {e}')

    return U, sigma, Vt, R_centered, explained_ratio, cumulative_ratio


# ============================================================
# 第3步: 隐因子解读 -- 因子暴露与行业关系
# ============================================================

def step3_interpret_factors(U, sigma, Vt, returns_df, stock_info):
    """分析隐因子的含义: 哪些股票在哪些因子上暴露最大"""
    print_section('第3步: 隐因子解读 -- 股票在隐因子上的暴露')

    stocks = list(returns_df.columns)
    n_show = min(5, U.shape[1])

    for k in range(n_show):
        print(f'\n  --- 隐因子 #{k+1} (方差占比: {sigma[k]**2/np.sum(sigma**2):.1%}) ---')

        loadings = U[:, k]

        sorted_idx = np.argsort(loadings)
        top_pos = sorted_idx[-3:][::-1]
        top_neg = sorted_idx[:3]

        print(f'    正暴露 top3:')
        for idx in top_pos:
            code = stocks[idx]
            name, industry = stock_info.get(code, (code, ''))
            print(f'      {code} {name:6s} ({industry:4s})  暴露={loadings[idx]:+.4f}')

        print(f'    负暴露 top3:')
        for idx in top_neg:
            code = stocks[idx]
            name, industry = stock_info.get(code, (code, ''))
            print(f'      {code} {name:6s} ({industry:4s})  暴露={loadings[idx]:+.4f}')

        # 行业平均暴露
        industry_exposure = {}
        for i, code in enumerate(stocks):
            _, industry = stock_info.get(code, ('', '其他'))
            if industry not in industry_exposure:
                industry_exposure[industry] = []
            industry_exposure[industry].append(loadings[i])

        print(f'    行业平均暴露:')
        for ind in sorted(industry_exposure.keys()):
            vals = industry_exposure[ind]
            mean_exp = np.mean(vals)
            print(f'      {ind:6s}: {mean_exp:+.4f}')

    return n_show


# ============================================================
# 第4步: 隐因子的时间序列 -- 发现不同时段的驱动力
# ============================================================

def step4_factor_time_series(Vt, sigma, returns_df):
    """分析隐因子的时间演化, 发现市场状态切换"""
    print_section('第4步: 隐因子时间序列 -- 不同时段的驱动力变化')

    dates = returns_df.index
    n_show = min(3, Vt.shape[0])

    factor_returns = {}
    for k in range(n_show):
        fk = Vt[k, :] * sigma[k]
        factor_returns[f'Factor_{k+1}'] = fk

    # 按季度统计因子收益
    print(f'\n  各隐因子季度平均收益:')
    print(f'    {"季度":<10}', end='')
    for k in range(n_show):
        print(f'  {"因子"+str(k+1):>8}', end='')
    print()
    print(f'    {"-"*10}', end='')
    for _ in range(n_show):
        print(f'  {"-"*8}', end='')
    print()

    quarters = pd.Series(dates).dt.to_period('Q').unique()
    for q in quarters:
        q_mask = pd.Series(dates).dt.to_period('Q') == q
        q_mask = q_mask.values
        print(f'    {str(q):<10}', end='')
        for k in range(n_show):
            fk = Vt[k, q_mask] * sigma[k]
            q_mean = fk.mean() * 100
            print(f'  {q_mean:>+7.3f}%', end='')
        print()

    # 因子强度随时间变化 (滚动标准差)
    window = 60
    print(f'\n  因子活跃度 ({window}日滚动标准差):')
    for k in range(n_show):
        fk = pd.Series(Vt[k, :] * sigma[k], index=dates)
        rolling_std = fk.rolling(window).std()
        peak_date = rolling_std.idxmax()
        peak_val = rolling_std.max()
        trough_date = rolling_std.dropna().idxmin()
        trough_val = rolling_std.dropna().min()
        print(f'    因子{k+1}: 最活跃={peak_date.strftime("%Y-%m-%d")}({peak_val:.4f}), '
              f'最沉寂={trough_date.strftime("%Y-%m-%d")}({trough_val:.4f})')

    # 可视化
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, axes = plt.subplots(n_show, 1, figsize=(14, 4 * n_show), sharex=True)
        if n_show == 1:
            axes = [axes]

        colors = ['steelblue', 'darkorange', 'seagreen']
        for k in range(n_show):
            ax = axes[k]
            fk = pd.Series(Vt[k, :] * sigma[k], index=dates)
            cumulative = fk.cumsum()
            ax.plot(dates, cumulative, color=colors[k % len(colors)], linewidth=1.5)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_ylabel(f'Factor {k+1}\nCumulative')
            ax.set_title(f'Hidden Factor #{k+1} Cumulative Return')
            ax.grid(True, alpha=0.3)

        plt.xlabel('Date')
        plt.tight_layout()
        plt.savefig('outputs/svd_factor_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'\n  图表已保存: outputs/svd_factor_timeseries.png')
    except Exception as e:
        print(f'  [可视化跳过] {e}')

    return factor_returns


# ============================================================
# 第5步: 行业因子结构对比
# ============================================================

def step5_industry_factor_structure(U, sigma, stock_info, returns_df):
    """对比不同行业在隐因子上的暴露差异"""
    print_section('第5步: 行业因子结构对比')

    stocks = list(returns_df.columns)
    n_factors = min(5, U.shape[1])

    industries = {}
    for i, code in enumerate(stocks):
        _, ind = stock_info.get(code, ('', '其他'))
        if ind not in industries:
            industries[ind] = []
        industries[ind].append(i)

    print(f'\n  行业因子暴露 (前{n_factors}个隐因子):')
    print(f'\n    {"行业":8s}', end='')
    for k in range(n_factors):
        print(f'  {"F"+str(k+1):>8}', end='')
    print()
    print(f'    {"-"*8}', end='')
    for _ in range(n_factors):
        print(f'  {"-"*8}', end='')
    print()

    industry_profiles = {}
    for ind in sorted(industries.keys()):
        idx_list = industries[ind]
        mean_exposure = np.mean(U[idx_list, :n_factors], axis=0)
        industry_profiles[ind] = mean_exposure
        print(f'    {ind:8s}', end='')
        for k in range(n_factors):
            print(f'  {mean_exposure[k]:>+8.4f}', end='')
        print()

    # 行业间距离矩阵 (因子空间中的欧氏距离)
    ind_names = sorted(industries.keys())
    n_ind = len(ind_names)
    print(f'\n  行业间距离矩阵 (欧氏距离):')
    print()
    print(f'    {"":8s}', end='')
    for ind in ind_names:
        print(f'  {ind:>8s}', end='')
    print()

    for i, ind_i in enumerate(ind_names):
        print(f'    {ind_i:8s}', end='')
        for j, ind_j in enumerate(ind_names):
            dist = np.linalg.norm(industry_profiles[ind_i] - industry_profiles[ind_j])
            if i == j:
                print(f'  {"---":>8s}', end='')
            else:
                print(f'  {dist:>8.4f}', end='')
        print()

    # 找到最相似和最不同的行业对
    max_dist = 0
    min_dist = float('inf')
    max_pair = ('', '')
    min_pair = ('', '')
    for i in range(n_ind):
        for j in range(i + 1, n_ind):
            dist = np.linalg.norm(industry_profiles[ind_names[i]] - industry_profiles[ind_names[j]])
            if dist > max_dist:
                max_dist = dist
                max_pair = (ind_names[i], ind_names[j])
            if dist < min_dist:
                min_dist = dist
                min_pair = (ind_names[i], ind_names[j])

    print(f'\n  最相似行业对: {min_pair[0]} - {min_pair[1]} (距离={min_dist:.4f})')
    print(f'  最不同行业对: {max_pair[0]} - {max_pair[1]} (距离={max_dist:.4f})')

    return industry_profiles


# ============================================================
# 第6步: 滚动窗口SVD -- 发现市场状态切换
# ============================================================

def step6_rolling_svd(returns_df, stock_info):
    """滚动窗口做SVD, 观察隐因子的解释力随时间变化"""
    print_section('第6步: 滚动窗口SVD -- 市场驱动因子的时变性')

    window = 120  # 约半年
    step = 20     # 每月滚动一次
    T = returns_df.shape[0]

    rolling_results = []

    for start in range(0, T - window, step):
        end = start + window
        window_data = returns_df.iloc[start:end]
        R_w = window_data.values.T
        R_w = R_w - R_w.mean(axis=1, keepdims=True)

        _, sigma_w, _ = np.linalg.svd(R_w, full_matrices=False)
        total_var = np.sum(sigma_w ** 2)
        top1_ratio = sigma_w[0] ** 2 / total_var
        top3_ratio = np.sum(sigma_w[:3] ** 2) / total_var

        mid_date = window_data.index[window // 2]
        rolling_results.append({
            'date': mid_date,
            'top1_var': top1_ratio,
            'top3_var': top3_ratio,
            'sigma_1': sigma_w[0],
            'sigma_2': sigma_w[1] if len(sigma_w) > 1 else 0,
            'sigma_3': sigma_w[2] if len(sigma_w) > 2 else 0,
        })

    roll_df = pd.DataFrame(rolling_results).set_index('date')

    print(f'\n  滚动窗口: {window}天, 步长: {step}天, 共 {len(roll_df)} 个窗口')
    print(f'\n  第一因子方差占比:')
    print(f'    {"时间段":<12}  {"Factor1占比":>12}  {"Top3占比":>10}  {"市场状态":>12}')
    print(f'    {"-"*12}  {"-"*12}  {"-"*10}  {"-"*12}')

    for _, row in roll_df.iterrows():
        date_str = row.name.strftime('%Y-%m')
        f1_pct = row['top1_var']
        f3_pct = row['top3_var']
        if f1_pct > 0.50:
            state = '齐涨齐跌'
        elif f1_pct > 0.35:
            state = '板块分化'
        else:
            state = '个股行情'
        print(f'    {date_str:<12}  {f1_pct:>11.1%}  {f3_pct:>9.1%}  {state:>12}')

    # 统计
    mean_f1 = roll_df['top1_var'].mean()
    max_f1 = roll_df['top1_var'].max()
    min_f1 = roll_df['top1_var'].min()
    max_date = roll_df['top1_var'].idxmax().strftime('%Y-%m')
    min_date = roll_df['top1_var'].idxmin().strftime('%Y-%m')

    print(f'\n  统计:')
    print(f'    Factor1平均占比: {mean_f1:.1%}')
    print(f'    Factor1最高: {max_f1:.1%} ({max_date})')
    print(f'    Factor1最低: {min_f1:.1%} ({min_date})')

    # 可视化
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.fill_between(roll_df.index, roll_df['top1_var'] * 100,
                         alpha=0.3, color='steelblue', label='Factor 1')
        ax.fill_between(roll_df.index, roll_df['top3_var'] * 100,
                         alpha=0.2, color='darkorange', label='Top 3')
        ax.plot(roll_df.index, roll_df['top1_var'] * 100,
                color='steelblue', linewidth=2)
        ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% (High Concentration)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Variance Explained (%)')
        ax.set_title('Rolling SVD: Factor Concentration Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('outputs/svd_rolling_concentration.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  图表已保存: outputs/svd_rolling_concentration.png')
    except Exception as e:
        print(f'  [可视化跳过] {e}')

    return roll_df


# ============================================================
# 第7步: SVD降维 vs 原始因子 -- 预测效果对比
# ============================================================

def step7_svd_factor_compression(returns_df, stock_info):
    """将50+因子用SVD压缩为K个主成分, 对比XGBoost预测效果"""
    print_section('第7步: SVD因子压缩 -- 50+因子 vs 主成分降维')

    test_stock = '600519.SH'
    stock_name = stock_info.get(test_stock, (test_stock, ''))[0]

    df = load_stock_data(test_stock, START_DATE, END_DATE)
    df = calc_features(df)
    feature_cols = [c for c in get_all_feature_cols() if c in df.columns]
    df = df.dropna(subset=feature_cols)

    label = (df['close'].shift(-1) > df['close']).astype(int)
    label = label.iloc[:-1]
    df = df.iloc[:-1]

    X = df[feature_cols].values
    y = label.values
    N_total = len(X)

    # Z-score
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1
    X_norm = (X - X_mean) / X_std

    # SVD降维
    U_full, S_full, Vt_full = np.linalg.svd(X_norm, full_matrices=False)
    total_var = np.sum(S_full ** 2)
    cum_var = np.cumsum(S_full ** 2) / total_var

    print(f'\n  股票: {test_stock} ({stock_name})')
    print(f'  原始因子数: {len(feature_cols)}')
    print(f'  样本量: {N_total}')

    k_values = [3, 5, 8, 10, 15, len(feature_cols)]
    print(f'\n  不同主成分数量的方差保留率:')
    for k in k_values:
        k_real = min(k, len(S_full))
        var_pct = cum_var[k_real - 1] if k_real <= len(cum_var) else 1.0
        print(f'    K={k_real:>3}: 保留 {var_pct:.1%} 的方差')

    # 用XGBoost对比预测效果
    train_size = int(N_total * 0.7)
    X_train_raw, X_test_raw = X_norm[:train_size], X_norm[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    results = []
    try:
        from xgboost import XGBClassifier
        from sklearn.metrics import roc_auc_score, accuracy_score

        # 原始特征
        model_full = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
        model_full.fit(X_train_raw, y_train)
        pred_full = model_full.predict_proba(X_test_raw)[:, 1]
        auc_full = roc_auc_score(y_test, pred_full)
        acc_full = accuracy_score(y_test, (pred_full > 0.5).astype(int))
        results.append(('全部因子', len(feature_cols), 100.0, auc_full, acc_full))

        # 不同K值的SVD降维
        for k in [3, 5, 8, 10, 15]:
            if k >= len(S_full):
                continue
            X_svd = U_full[:, :k] * S_full[:k]
            X_train_k, X_test_k = X_svd[:train_size], X_svd[train_size:]

            model_k = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                use_label_encoder=False, eval_metric='logloss', verbosity=0
            )
            model_k.fit(X_train_k, y_train)
            pred_k = model_k.predict_proba(X_test_k)[:, 1]
            auc_k = roc_auc_score(y_test, pred_k)
            acc_k = accuracy_score(y_test, (pred_k > 0.5).astype(int))
            var_pct = cum_var[k - 1] * 100
            results.append((f'SVD K={k}', k, var_pct, auc_k, acc_k))

        print(f'\n  XGBoost预测效果对比:')
        print(f'    {"方法":12s}  {"维度":>6}  {"方差保留":>8}  {"AUC":>8}  {"Accuracy":>8}')
        print(f'    {"-"*12}  {"-"*6}  {"-"*8}  {"-"*8}  {"-"*8}')
        for name, dim, var_pct, auc, acc in results:
            print(f'    {name:12s}  {dim:>6}  {var_pct:>7.1f}%  {auc:>8.4f}  {acc:>8.4f}')

        best_svd = max([r for r in results if r[0].startswith('SVD')], key=lambda x: x[3])
        print(f'\n  结论: 全量因子 AUC={auc_full:.4f}, {best_svd[0]} AUC={best_svd[3]:.4f}')

    except ImportError:
        print(f'  [跳过XGBoost对比] xgboost未安装')

    return results, Vt_full, S_full, feature_cols, cum_var


# ============================================================
# 第7.5步: 隐因子溯源 -- SVD因子到底是什么
# ============================================================

def step7b_factor_tracing(Vt_full, S_full, feature_cols, cum_var):
    """将SVD隐因子映射回原始因子空间, 解答'隐因子是什么'"""
    print_section('第7.5步: 隐因子溯源 -- SVD因子到底是什么?')

    n_show = min(5, Vt_full.shape[0])
    category_map = {}
    for cat_key, cat_info in FACTOR_TAXONOMY.items():
        for feat in cat_info['features']:
            category_map[feat] = cat_info['name']

    for k in range(n_show):
        var_pct = (S_full[k] ** 2) / np.sum(S_full ** 2)
        cum_pct = cum_var[k] if k < len(cum_var) else 1.0
        print(f'  --- PC{k+1} (方差占比: {var_pct:.1%}, 累积: {cum_pct:.1%}) ---')

        loadings = Vt_full[k, :]  # 这一行是PC_k对52个原始因子的权重

        sorted_idx = np.argsort(np.abs(loadings))[::-1]

        print(f'    原始因子权重 top8:')
        print(f'      {"原始因子":<25s} {"权重":>8} {"分类":>10}')
        print(f'      {"-"*25} {"-"*8} {"-"*10}')

        top_n = 8
        category_weights = {}
        for j in range(len(loadings)):
            feat = feature_cols[j]
            cat = category_map.get(feat, '其他')
            if cat not in category_weights:
                category_weights[cat] = 0
            category_weights[cat] += abs(loadings[j])

        for rank, idx in enumerate(sorted_idx[:top_n]):
            feat = feature_cols[idx]
            w = loadings[idx]
            cat = category_map.get(feat, '其他')
            sign = '+' if w > 0 else '-'
            print(f'      {feat:<25s} {sign}{abs(w):>7.4f} {cat:>10}')

        # 按因子大类汇总权重
        sorted_cats = sorted(category_weights.items(), key=lambda x: x[1], reverse=True)
        top_cat = sorted_cats[0][0]
        second_cat = sorted_cats[1][0] if len(sorted_cats) > 1 else ''

        print(f'\n    按因子类别汇总权重:')
        for cat_name, total_w in sorted_cats:
            bar_len = int(total_w * 30)
            bar = '#' * bar_len
            print(f'      {cat_name:12s}: {total_w:.3f}  {bar}')

        print(f'\n    PC{k+1} 主导类别: {top_cat}, {second_cat}')

    return


# ============================================================
# 第8步: 残差分析 -- 发现"独立行情"的个股
# ============================================================

def step8_residual_analysis(U, sigma, Vt, R_centered, returns_df, stock_info):
    """分析残差矩阵, 找到走出独立行情的个股"""
    print_section('第8步: 残差分析 -- 发现走独立行情的个股')

    dates = returns_df.index
    stocks = list(returns_df.columns)

    # 构建等权大盘收益率 (16只股票等权)
    market_ret = returns_df.mean(axis=1).values  # (T,)

    # 构建各行业等权收益率
    industry_returns = {}
    for i, code in enumerate(stocks):
        _, ind = stock_info.get(code, ('', '其他'))
        if ind not in industry_returns:
            industry_returns[ind] = []
        industry_returns[ind].append(returns_df[code].values)
    for ind in industry_returns:
        industry_returns[ind] = np.mean(industry_returns[ind], axis=0)

    n_show_factors = min(3, Vt.shape[0])
    print(f'\n  隐因子 vs 等权大盘/行业 相关性:')
    print(f'    {"":12s}  {"等权大盘":>10}', end='')
    for ind in sorted(industry_returns.keys()):
        print(f'  {ind:>8}', end='')
    print()
    print(f'    {"-"*12}  {"-"*10}', end='')
    for _ in industry_returns:
        print(f'  {"-"*8}', end='')
    print()

    for k in range(n_show_factors):
        factor_ts = Vt[k, :] * sigma[k]
        corr_market = np.corrcoef(factor_ts, market_ret)[0, 1]
        print(f'    Factor{k+1:5d}  {corr_market:>+10.3f}', end='')
        for ind in sorted(industry_returns.keys()):
            corr_ind = np.corrcoef(factor_ts, industry_returns[ind])[0, 1]
            print(f'  {corr_ind:>+8.3f}', end='')
        print()

    n_factors_reconstruct = 5
    R_approx = U[:, :n_factors_reconstruct] @ np.diag(sigma[:n_factors_reconstruct]) @ Vt[:n_factors_reconstruct, :]
    residual = R_centered - R_approx  # (N, T)

    stocks = list(returns_df.columns)
    residual_stats = []
    for i, code in enumerate(stocks):
        name, industry = stock_info.get(code, (code, ''))
        res_std = np.std(residual[i, :])
        res_total = np.sum(residual[i, :] ** 2)
        total_var = np.sum(R_centered[i, :] ** 2)
        unexplained_ratio = res_total / total_var if total_var > 0 else 0
        residual_stats.append({
            'code': code,
            'name': name,
            'industry': industry,
            'residual_std': res_std,
            'unexplained_ratio': unexplained_ratio,
            'total_var': total_var,
        })

    res_df = pd.DataFrame(residual_stats).sort_values('unexplained_ratio', ascending=False)

    print(f'\n  使用前{n_factors_reconstruct}个隐因子重构, 各股残差占比:')
    print(f'    {"代码":12s} {"名称":8s} {"行业":6s} {"残差占比":>8} {"残差波动":>8} {"标签":>15}')
    print(f'    {"-"*12} {"-"*8} {"-"*6} {"-"*8} {"-"*8} {"-"*15}')

    for _, row in res_df.iterrows():
        unexp = row['unexplained_ratio']
        if unexp > 0.5:
            comment = '独立行情强'
        elif unexp > 0.3:
            comment = '有alpha信号'
        else:
            comment = '跟随大盘'
        print(f'    {row["code"]:12s} {row["name"]:8s} {row["industry"]:6s} '
              f'{unexp:>7.1%} {row["residual_std"]:>8.4f} {comment:>15}')

    # ---- 同行业内部对比: 解释为什么同是银行/科技, 残差差异大 ----
    print(f'\n  同行业内部对比:')
    industries_in_data = {}
    for _, row in res_df.iterrows():
        ind = row['industry']
        if ind not in industries_in_data:
            industries_in_data[ind] = []
        industries_in_data[ind].append((row['name'], row['unexplained_ratio'], row['code']))

    for ind in sorted(industries_in_data.keys()):
        members = industries_in_data[ind]
        if len(members) < 2:
            continue
        members_sorted = sorted(members, key=lambda x: x[1], reverse=True)
        spread = members_sorted[0][1] - members_sorted[-1][1]
        if spread < 0.1:
            continue
        print(f'    [{ind}] 残差占比差异 = {spread:.1%}:')
        for name, unexp, code in members_sorted:
            # 计算该股票与行业等权的相关性
            stock_idx = stocks.index(code)
            stock_ret = returns_df[code].values
            ind_avg_ret = np.mean(
                [returns_df[c].values for c in stocks
                 if stock_info.get(c, ('', ''))[1] == ind and c != code],
                axis=0
            )
            corr_with_peers = np.corrcoef(stock_ret, ind_avg_ret)[0, 1]
            label = '与同行高度同步' if corr_with_peers > 0.6 else \
                    '与同行中度同步' if corr_with_peers > 0.4 else '与同行不同步'
            print(f'      {name:8s}: 残差{unexp:.1%}  同行相关={corr_with_peers:.3f} ({label})')

    return res_df


# ============================================================
# 主流程
# ============================================================

def main():
    # 第1步: 构建收益率矩阵
    returns_df, stock_info = step1_build_return_matrix()
    if returns_df.empty:
        print('没有加载到数据, 程序退出')
        return

    # 第2步: SVD分解 + 奇异值衰减
    U, sigma, Vt, R_centered, explained_ratio, cum_ratio = step2_svd_decomposition(
        returns_df, stock_info
    )

    # 第3步: 隐因子解读
    step3_interpret_factors(U, sigma, Vt, returns_df, stock_info)

    # 第4步: 隐因子时间序列
    step4_factor_time_series(Vt, sigma, returns_df)

    # 第5步: 行业因子结构对比
    step5_industry_factor_structure(U, sigma, stock_info, returns_df)

    # 第6步: 滚动窗口SVD
    step6_rolling_svd(returns_df, stock_info)

    # 第7步: SVD因子压缩 vs 原始因子
    compression_results, Vt_feat, S_feat, feat_cols, cum_var_feat = \
        step7_svd_factor_compression(returns_df, stock_info)

    # 第7.5步: 隐因子溯源 -- SVD因子到底是什么
    if Vt_feat is not None:
        step7b_factor_tracing(Vt_feat, S_feat, feat_cols, cum_var_feat)

    # 第8步: 残差分析
    step8_residual_analysis(U, sigma, Vt, R_centered, returns_df, stock_info)


if __name__ == '__main__':
    main()
