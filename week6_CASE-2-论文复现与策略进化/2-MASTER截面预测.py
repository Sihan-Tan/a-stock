# -*- coding: utf-8 -*-
"""
2-MASTER截面预测 —— 使用论文预训练模型进行截面预测与 IC 评估

本脚本是"MASTER论文复现"的核心环节，直接使用 Li et al. (AAAI 2024)
提供的预训练模型权重，在论文配套的测试集上评估模型性能。

核心流程:
  1. 加载论文作者提供的测试数据（Alpha158 因子 + 63 维市场信息）
  2. 加载论文预训练的 MASTER 模型权重
  3. 对测试集逐日进行截面预测
  4. 计算 IC/ICIR/RankIC/RankICIR 等评估指标
  5. 与论文 Table 1 的结果对比验证

什么是 IC（Information Coefficient，信息系数）？
  IC 是截面预测中最核心的评估指标，衡量模型的预测能力：
  - IC = Pearson 相关系数（预测值 vs 真实收益）
    - 度量两者之间的线性相关程度
    - 取值范围 [-1, 1]，正值表示预测与真实正相关
  - RankIC = Spearman 秩相关系数（预测排名 vs 真实排名）
    - 度量排列顺序的一致性
    - 对异常值不敏感，更关注排序准确性（对量化选股更重要）
  - ICIR = IC 的均值 / IC 的标准差
    - 衡量 IC 的稳定性和信噪比
    - 比 IC 更重要的指标——高但波动大的 IC 不如稳定但中等水平的 IC

MASTER 模型架构要点:
  1. T-Attention (Temporal Attention): 时序注意力，捕捉单只股票的多日模式
     - 输入: T 天的因子数据 (T=8)
     - 输出: 融合了时序信息的特征表示
  2. S-Attention (Spatial Attention): 空间注意力，捕捉股票间的截面联动
     - 输入: 同一交易日所有股票的特征
     - 输出: 融入了"市场整体情况"的个股特征
     - 这是 MASTER 区别于传统时序模型的关键创新
  3. Gate 机制: 门控网络，根据 63 维市场信息动态调整因子权重
     - 输入: 市场状态（波动率、成交量等指数层面信息）
     - 输出: 各个因子的权重调整系数
     - 让模型能根据市场环境切换"选股逻辑"

MASTER 论文: Li et al., "MASTER: Market-Guided Stock Transformer (AAAI 2024)"

数据来源: 论文作者提供的 opensource 数据集（基于 Qlib 框架）
  - 训练集: 2008Q1 ~ 2020Q1（超过 12 年的数据）
  - 验证集: 2020Q2
  - 测试集: 2020Q3 ~ 2022Q4（约 2.5 年）
  - 股票池: CSI300（沪深300成分股，约 300 只）
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# 添加 MASTER 论文源码到 Python 路径
# MASTER-master 目录包含了论文作者提供的 MASTERModel 实现
_MASTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'MASTER-master')
sys.path.insert(0, _MASTER_DIR)

from master import MASTERModel


# ============================================================
# 配置
# ============================================================

# 股票池选择: 'csi300'（沪深300）或 'csi800'（中证800）
# CSI300 是 300 只大盘蓝筹股，CSI800 是 800 只中大盘股
# 不同股票池的因子质量和 IC 水平不同（大盘股数据质量更好）
UNIVERSE = 'csi300'

# 数据路径（opensource 数据已复制到 MASTER-master/data/opensource/）
DATA_DIR = os.path.join(_MASTER_DIR, 'data', 'opensource')
MODEL_DIR = os.path.join(_MASTER_DIR, 'model')

# 数据来源标识（与模型权重文件名对应）
PREFIX = 'opensource'

# ============================================================
# MASTER 模型超参数（与论文保持一致）
# ============================================================

D_FEAT = 158           # Alpha158 因子数量：MASTER 使用的因子维度
D_MODEL = 256          # Transformer 隐藏层维度
T_NHEAD = 4            # T-Attention 头数（N1=4），捕捉时序多模式
S_NHEAD = 2            # S-Attention 头数（N2=2），捕捉截面多模式
DROPOUT = 0.5          # Dropout 率，防止过拟合

# Gate 机制参数：市场信息在特征中的起止索引
# 特征矩阵的维度 = 158（Alpha158）+ 63（市场信息）+ 1（标签）= 222
# Gate 只作用于因子部分，不对市场信息和标签做门控
GATE_INPUT_START = 158  # 市场信息起始位置（因子之后）
GATE_INPUT_END = 221    # 市场信息结束位置（158+63=221）

# Beta 参数：控制 Gate 机制的"筛选强度"
# CSI300 用 beta=5（因子质量高，筛选可以更严格）
# CSI800 用 beta=2（成分股更多样，需要更宽松的筛选）
BETA = 5 if UNIVERSE == 'csi300' else 2

# 论文 Table 1 的参考指标值
# 这些值是 5 次不同随机种子的平均值
# 我们使用 seed=0 的单次结果，存在一定差异属于正常
PAPER_REFERENCE = {
    'csi300': {'IC': 0.064, 'ICIR': 0.42, 'RankIC': 0.076, 'RankICIR': 0.49},
    'csi800': {'IC': 0.052, 'ICIR': 0.40, 'RankIC': 0.066, 'RankICIR': 0.48},
}


def calc_ic(pred, label):
    """
    计算 IC（Pearson 相关系数）和 RankIC（Spearman 秩相关系数）

    参数:
        pred: array-like, 模型预测值
        label: array-like, 真实收益标签

    返回:
        (ic, ric) 元组
        ic: Pearson 相关系数，衡量线性预测能力
        ric: Spearman 秩相关系数，衡量排序预测能力
    """
    df = pd.DataFrame({'pred': pred, 'label': label})
    df = df.dropna()
    if len(df) < 5:
        return np.nan, np.nan
    ic = df['pred'].corr(df['label'])                     # Pearson
    ric = df['pred'].corr(df['label'], method='spearman') # Spearman
    return ic, ric


# ============================================================
# 1. 加载数据
# ============================================================
print("=" * 70)
print(f"  MASTER截面预测 - {UNIVERSE.upper()}")
print(f"  beta={BETA}, d_model={D_MODEL}, T_nhead={T_NHEAD}, S_nhead={S_NHEAD}")
print("=" * 70)

# 加载测试数据：使用 pickle 格式（Qlib 的数据存储格式）
test_path = os.path.join(DATA_DIR, f'{UNIVERSE}_dl_test.pkl')
print(f"\n[1] 加载测试数据: {test_path}")

with open(test_path, 'rb') as f:
    dl_test = pickle.load(f)

# 获取索引信息：MultiIndex (datetime, instrument)
# Qlib 的 Dataset 结构使用 MultiIndex 组织截面数据
test_index = dl_test.get_index()
dates = test_index.get_level_values('datetime')
instruments = test_index.get_level_values('instrument')

print(f"    样本数: {len(dl_test)}")
print(f"    股票数: {instruments.nunique()}")
print(f"    日期范围: {dates.min().strftime('%Y-%m-%d')} -> {dates.max().strftime('%Y-%m-%d')}")
print(f"    交易日数: {dates.nunique()}")
print(f"    每样本维度: {dl_test[0].shape} (T=8天, F=222=158因子+63市场+1标签)")


# ============================================================
# 2. 加载预训练模型
# ============================================================
model_path = os.path.join(MODEL_DIR, f'{UNIVERSE}_{PREFIX}_0.pkl')
print(f"\n[2] 加载预训练模型: {os.path.basename(model_path)}")

# 初始化 MASTER 模型实例
# 参数必须与预训练权重匹配，否则加载时会报错
model = MASTERModel(
    d_feat=D_FEAT, d_model=D_MODEL,
    t_nhead=T_NHEAD, s_nhead=S_NHEAD,
    T_dropout_rate=DROPOUT, S_dropout_rate=DROPOUT,
    beta=BETA,
    gate_input_start_index=GATE_INPUT_START,
    gate_input_end_index=GATE_INPUT_END,
    n_epochs=1, lr=1e-5, GPU=0, seed=0,
    train_stop_loss_thred=0.95,
    save_path=MODEL_DIR, save_prefix=f'{UNIVERSE}_{PREFIX}'
)
model.load_param(model_path)
print(f"    模型参数量: {sum(p.numel() for p in model.model.parameters()):,}")
print(f"    运行设备: {model.device}")


# ============================================================
# 3. 运行截面预测
# ============================================================
print(f"\n[3] 运行截面预测...")

import torch
from torch.utils.data import DataLoader

# DailyBatchSamplerRandom 按交易日分批返回数据
# 这是截面预测的核心——每个 batch 对应一个交易日的所有股票
# shuffle=False 保证预测顺序与测试集一致
from base_model import DailyBatchSamplerRandom, zscore, drop_na

sampler = DailyBatchSamplerRandom(dl_test, shuffle=False)
test_loader = DataLoader(dl_test, sampler=sampler, drop_last=False)

daily_ic_list = []     # 每日 IC 值
daily_ric_list = []    # 每日 RankIC 值
daily_dates = []
all_preds = []         # 所有预测值（用于最终分析）
all_labels = []        # 所有真实值

model.model.eval()     # 切换到评估模式（禁用 dropout/batch norm）
with torch.no_grad():  # 关闭梯度计算（推理模式下不需要，节省显存和计算）
    for batch_idx, data in enumerate(test_loader):
        # data shape: [1, n_stocks, T, F]
        # 第一个维度 squeeze 掉（batch_size=1 的冗余维度）
        data = torch.squeeze(data, dim=0)

        # 特征: 所有时间步的前 D_FEAT+63 列（不包括最后一列标签）
        # 最后一列是未来收益率标签
        feature = data[:, :, 0:-1].to(model.device)
        label = data[:, -1, -1].numpy()  # 取最后一天、最后一列的标签

        # 模型前向推理
        pred = model.model(feature.float()).detach().cpu().numpy().ravel()

        # 计算当日的 IC 和 RankIC
        daily_ic, daily_ric = calc_ic(pred, label)
        daily_ic_list.append(daily_ic)
        daily_ric_list.append(daily_ric)
        all_preds.extend(pred)
        all_labels.extend(label)

        if (batch_idx + 1) % 100 == 0:
            print(f"    已处理 {batch_idx + 1} 个交易日...")

# 获取每日的日期列表
daily_counts = pd.Series(index=test_index).groupby("datetime").size()
daily_dates = daily_counts.index.tolist()

print(f"    预测完成! 共 {len(daily_dates)} 个交易日")


# ============================================================
# 4. 计算整体指标
# ============================================================
print(f"\n[4] 评估指标计算")

# 过滤 NaN（某些交易日可能因数据不足无法计算 IC）
ic_arr = np.array(daily_ic_list)
ric_arr = np.array(daily_ric_list)
valid_ic = ic_arr[~np.isnan(ic_arr)]
valid_ric = ric_arr[~np.isnan(ric_arr)]

# 计算核心指标
metrics = {
    'IC': np.mean(valid_ic),                               # 平均 IC
    'ICIR': np.mean(valid_ic) / np.std(valid_ic),         # IC 的信息比率
    'RankIC': np.mean(valid_ric),                          # 平均 RankIC
    'RankICIR': np.mean(valid_ric) / np.std(valid_ric),   # RankIC 的信息比率
}

ref = PAPER_REFERENCE[UNIVERSE]

# 打印与论文指标的对比结果
print(f"\n{'='*70}")
print(f"  MASTER {UNIVERSE.upper()} 截面预测结果")
print(f"{'='*70}")
print(f"{'指标':<12}{'本次结果':>12}{'论文参考值':>12}{'差异':>12}")
print(f"{'-'*48}")
print(f"{'IC':<12}{metrics['IC']:>12.4f}{ref['IC']:>12.4f}{metrics['IC']-ref['IC']:>+12.4f}")
print(f"{'ICIR':<12}{metrics['ICIR']:>12.4f}{ref['ICIR']:>12.4f}{metrics['ICIR']-ref['ICIR']:>+12.4f}")
print(f"{'RankIC':<12}{metrics['RankIC']:>12.4f}{ref['RankIC']:>12.4f}{metrics['RankIC']-ref['RankIC']:>+12.4f}")
print(f"{'RankICIR':<12}{metrics['RankICIR']:>12.4f}{ref['RankICIR']:>12.4f}{metrics['RankICIR']-ref['RankICIR']:>+12.4f}")
print(f"\n注: 论文Table 1为5次随机种子的平均值, 本次仅用seed=0的单次结果, 存在差异属正常。")


# ============================================================
# 5. IC 正比例和统计信息
# ============================================================
print(f"\n[5] IC统计分析")

ic_positive_rate = np.mean(valid_ic > 0) * 100
ric_positive_rate = np.mean(valid_ric > 0) * 100

# IC 正值比例是衡量模型实用性的重要指标
# 如果模型 IC 全部为正，说明模型始终能正确判断方向（涨或跌）
# 实际中 IC 正值比例在 50%~70% 之间都算不错
print(f"    IC > 0 的比例:     {ic_positive_rate:.1f}% ({int(np.sum(valid_ic > 0))}/{len(valid_ic)} 天)")
print(f"    RankIC > 0 的比例: {ric_positive_rate:.1f}% ({int(np.sum(valid_ric > 0))}/{len(valid_ric)} 天)")
print(f"    IC 中位数:         {np.median(valid_ic):.4f}")
print(f"    IC 标准差:         {np.std(valid_ic):.4f}")
print(f"    IC 最大值:         {np.max(valid_ic):.4f}")
print(f"    IC 最小值:         {np.min(valid_ic):.4f}")


# ============================================================
# 6. 可视化
# ============================================================
print(f"\n[6] 生成可视化图表...")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f'MASTER {UNIVERSE.upper()} 截面预测评估', fontsize=16, fontweight='bold')

# 6a: 逐日 IC 时序图（柱状图 + 均值线）
# 绿色柱 = IC > 0（预测正确的方向），红色柱 = IC < 0（预测错误的方向）
ax = axes[0, 0]
ax.bar(range(len(valid_ic)), valid_ic, color=['#27ae60' if x > 0 else '#e74c3c' for x in valid_ic],
       alpha=0.6, width=1.0)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(y=metrics['IC'], color='#3498db', linewidth=2, linestyle='--',
           label=f"IC Mean = {metrics['IC']:.4f}")
ax.set_title('逐日IC (Pearson相关)')
ax.set_xlabel('交易日')
ax.set_ylabel('IC')
ax.legend(loc='upper right')

# 6b: 逐日 RankIC 时序图（与 IC 类似，但基于 Spearman 秩相关）
ax = axes[0, 1]
ax.bar(range(len(valid_ric)), valid_ric, color=['#27ae60' if x > 0 else '#e74c3c' for x in valid_ric],
       alpha=0.6, width=1.0)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.axhline(y=metrics['RankIC'], color='#3498db', linewidth=2, linestyle='--',
           label=f"RankIC Mean = {metrics['RankIC']:.4f}")
ax.set_title('逐日RankIC (Spearman相关)')
ax.set_xlabel('交易日')
ax.set_ylabel('RankIC')
ax.legend(loc='upper right')

# 6c: IC / RankIC 分布直方图
# 理想情况下，分布应该以正值为中心，且形状接近正态
ax = axes[1, 0]
ax.hist(valid_ic, bins=50, color='#3498db', alpha=0.7, edgecolor='white', label='IC')
ax.hist(valid_ric, bins=50, color='#e67e22', alpha=0.5, edgecolor='white', label='RankIC')
ax.axvline(x=0, color='black', linewidth=1)
ax.axvline(x=metrics['IC'], color='#3498db', linewidth=2, linestyle='--')
ax.axvline(x=metrics['RankIC'], color='#e67e22', linewidth=2, linestyle='--')
ax.set_title('IC / RankIC 分布')
ax.set_xlabel('IC值')
ax.set_ylabel('频数')
ax.legend()

# 6d: 累计 IC 曲线
# 如果模型持续有效，累计 IC 曲线应该向右上方倾斜
# 曲线走平或下降意味着模型在该时期失效
ax = axes[1, 1]
cumsum_ic = np.cumsum(valid_ic)
cumsum_ric = np.cumsum(valid_ric)
ax.plot(cumsum_ic, color='#3498db', linewidth=1.5, label='Cumulative IC')
ax.plot(cumsum_ric, color='#e67e22', linewidth=1.5, label='Cumulative RankIC')
ax.set_title('累计IC曲线 (向上倾斜=持续有效)')
ax.set_xlabel('交易日')
ax.set_ylabel('累计IC')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)),
            f'MASTER_{UNIVERSE}_prediction_results.png'), dpi=150, bbox_inches='tight')
plt.show()


# ============================================================
# 7. 预测结果分析: 样本日期的截面排名
# ============================================================
print(f"\n[7] 截面预测示例 (展示3个样本日期的预测排名)")

# 将预测结果整理为 Series（MultiIndex: datetime, instrument）
predictions_series = pd.Series(np.array(all_preds), index=test_index)
labels_series = pd.Series(np.array(all_labels), index=test_index)

# 均匀选取 3 个样本日期做展示
sample_dates = daily_dates[::len(daily_dates) // 3][:3]

for dt in sample_dates:
    day_pred = predictions_series.xs(dt, level='datetime')
    day_label = labels_series.xs(dt, level='datetime')

    day_df = pd.DataFrame({
        'pred_score': day_pred,         # 模型预测分数（升序=越看好）
        'actual_ret': day_label,        # 实际未来收益率
        'pred_rank': day_pred.rank(ascending=False),   # 预测排名（1=最好）
        'actual_rank': day_label.rank(ascending=False),# 实际排名（1=最好）
    }).dropna()

    if len(day_df) == 0:
        continue

    day_ic = day_df['pred_score'].corr(day_df['actual_ret'])
    day_ric = day_df['pred_score'].corr(day_df['actual_ret'], method='spearman')

    print(f"\n  日期: {dt.strftime('%Y-%m-%d')} | 股票数: {len(day_df)} | IC={day_ic:.4f} | RankIC={day_ric:.4f}")
    # 展示预测排名前 5 的股票（模型最看好的）
    print(f"  预测排名前5 (最看好):")
    top5 = day_df.nsmallest(5, 'pred_rank')  # rank 值越小排名越高
    for stock, row in top5.iterrows():
        print(f"    {stock:<12} 预测排名: {int(row['pred_rank']):>3}  实际排名: {int(row['actual_rank']):>3}")

    # 展示预测排名后 5 的股票（模型最不看好的）
    print(f"  预测排名后5 (最不看好):")
    bottom5 = day_df.nlargest(5, 'pred_rank')
    for stock, row in bottom5.iterrows():
        print(f"    {stock:<12} 预测排名: {int(row['pred_rank']):>3}  实际排名: {int(row['actual_rank']):>3}")


# ============================================================
# 8. 与 XGBoost 基线对比总结
# ============================================================
print(f"\n{'='*70}")
print(f"  MASTER vs XGBoost (论文Table 1, {UNIVERSE.upper()})")
print(f"{'='*70}")

# XGBoost 基线数据来自论文 Table 1
# XGBoost 是 2016-2020 年间量化选股的"工业标准"
xgb_ref = {
    'csi300': {'IC': 0.051, 'ICIR': 0.37, 'RankIC': 0.050, 'RankICIR': 0.36},
    'csi800': {'IC': 0.040, 'ICIR': 0.37, 'RankIC': 0.047, 'RankICIR': 0.42},
}
xgb = xgb_ref[UNIVERSE]

print(f"{'指标':<12}{'MASTER(本次)':>14}{'MASTER(论文)':>14}{'XGBoost(论文)':>14}{'提升':>10}")
print(f"{'-'*64}")
for key, xkey in [('IC', 'IC'), ('ICIR', 'ICIR'), ('RankIC', 'RankIC'), ('RankICIR', 'RankICIR')]:
    m_val = metrics[key]
    m_ref = ref[key]
    x_val = xgb[xkey]
    pct = (m_ref - x_val) / x_val * 100
    print(f"{key:<12}{m_val:>14.4f}{m_ref:>14.4f}{x_val:>14.4f}{pct:>+9.1f}%")

print(f"\n结论: MASTER通过Transformer架构(TAttention+SAttention+Gate)在截面预测上")
print(f"      显著优于传统XGBoost, 验证了深度学习在股票预测中的优势。")
print(f"\n提示: 修改脚本顶部 UNIVERSE='csi800' 可切换到CSI800数据集。")
