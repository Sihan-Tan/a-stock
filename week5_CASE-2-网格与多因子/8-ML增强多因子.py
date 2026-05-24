# -*- coding: utf-8 -*-
"""
ML增强多因子选股 - LightGBM 预测未来收益
==========================================

核心思路:
---------
脚本5的多因子打分用的是"人工定权重" — 权重是拍脑袋的。
本脚本用 LightGBM 从数据中自动学习最优的因子组合和权重。

ML选股流程 (Walk-Forward / 滚动回测):
-----------------------------------
1. 特征工程: 构建 13 维特征矩阵 (X)
   - 技术因子: 5/10/20/60日动量, 波动率, RSI, ADX, MACD, OBV斜率, 换手率, 价格位置
   - 交互特征: 动量*波动率, ADX*(RSI-50)  — 捕捉因子的非线性组合
2. 构建标签: 未来 20 日收益率 (Y)
3. 滚动训练:
   - 每月末, 用过去 12 个月的数据训练模型
   - 预测当月所有股票的未来收益
   - 选出预测值最高的 Top-N 构建组合
4. 等权持有至下月末

对比:
  - 等权因子排序 (基准策略, 如仅用动量因子)
  - LightGBM 选股 (本脚本)

特征工程详解:
-------------
  技术因子: TA-Lib 计算的标准指标
  交互特征: 捕捉因子间的非线性关系 (如"高动量+低波动"比单独用动量更有效)
  时序特征: OBV斜率 (量能变化方向)

Walk-Forward 回测:
  这是 ML 在金融领域回测的标准方法:
  每月重新训练模型, 用新数据预测, 避免未来信息泄露。

运行方式:
  python 8-ML增强多因子.py
"""
import numpy as np
import pandas as pd
import talib
import time
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM 未安装, 将使用 XGBoost 或 RandomForest 替代")
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        HAS_GBR = True
    except ImportError:
        HAS_GBR = False

from db_config import execute_query, INITIAL_CASH


# ============================================================
# 数据加载
# ============================================================

def batch_load_daily(start_date, end_date, min_bars=120):
    """批量加载日K线数据, 返回 dict {code: DataFrame}"""
    sql = """
        SELECT stock_code, trade_date, open_price, high_price, low_price,
               close_price, volume
        FROM trade_stock_daily
        WHERE trade_date >= %s AND trade_date <= %s
        ORDER BY stock_code, trade_date ASC
    """
    rows = execute_query(sql, [start_date, end_date])
    if not rows:
        return {}

    df_all = pd.DataFrame(rows)
    df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])
    for col in ['open_price', 'high_price', 'low_price', 'close_price', 'volume']:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    result = {}
    for code, group in df_all.groupby('stock_code'):
        sub = group.set_index('trade_date').sort_index()
        sub = sub[['open_price', 'high_price', 'low_price', 'close_price', 'volume']]
        sub.columns = ['open', 'high', 'low', 'close', 'volume']
        if len(sub) >= min_bars:
            result[code] = sub
    return result


# ============================================================
# 特征工程
# ============================================================

FEATURE_NAMES = [
    # 动量特征: 不同周期的价格变化率
    'momentum_5d', 'momentum_10d', 'momentum_20d', 'momentum_60d',
    # 风险特征
    'volatility',       # 归一化波动率
    'rsi_14',           # 相对强弱指标
    'adx_14',           # 趋势强度
    # 量能特征
    'macd_hist',        # MACD柱状图
    'obv_slope',        # OBV斜率 (量能变化方向)
    'vol_ratio',        # 换手率 (当日量/20日均量)
    'price_position',   # 价格在60日区间的位置
    # 交互特征 (捕捉非线性关系)
    'mom_vol_cross',    # 动量 * 波动率 (高动量低波动 = 好)
    'adx_rsi_cross',    # ADX * (RSI-50) (趋势+方向)
]


def calc_features(df, idx=-1):
    """
    计算单只股票在指定时点的特征向量。

    这是一个增强版的特征计算函数, 相比 factor_engine.py 的版本:
      - 增加 5/10 日短期动量
      - 增加 OBV 斜率 (量能指标)
      - 增加交互特征 (mom_vol_cross, adx_rsi_cross)
      - 支持任意时点的特征提取 (通过 idx 参数)

    参数:
        df: DataFrame, 含 open/high/low/close/volume
        idx: 计算时点索引 (-1 = 最新)

    返回值:
        dict {feature_name: value} 或 None (数据不足时)
    """
    n = len(df)
    if n < 80:
        return None

    h = df['high'].values.astype(np.float64)
    l = df['low'].values.astype(np.float64)
    c = df['close'].values.astype(np.float64)
    o = df['open'].values.astype(np.float64)
    v = df['volume'].values.astype(np.float64)

    if c[idx] <= 0 or np.isnan(c[idx]):
        return None

    try:
        # ---- 基础技术因子 ----
        roc_5 = talib.ROC(c, timeperiod=5)         # 5日动量
        roc_10 = talib.ROC(c, timeperiod=10)        # 10日动量
        roc_20 = talib.ROC(c, timeperiod=20)        # 20日动量
        roc_60 = talib.ROC(c, timeperiod=60)        # 60日动量
        atr = talib.ATR(h, l, c, timeperiod=14)     # ATR波动率
        rsi = talib.RSI(c, timeperiod=14)           # RSI
        adx = talib.ADX(h, l, c, timeperiod=14)     # ADX趋势强度
        _, _, macd_hist = talib.MACD(c)             # MACD柱状图
        vol_ma = talib.SMA(v, timeperiod=20)        # 20日均量

        # ---- OBV 斜率 (量能变化方向) ----
        # OBV = On-Balance Volume, 累积成交量指标
        obv = talib.OBV(c, v)
        obv_ma5 = talib.SMA(obv, timeperiod=5)
        obv_slope = (obv[idx] - obv_ma5[idx]) / (obv_ma5[idx] + 1e-10) if not np.isnan(obv_ma5[idx]) else 0

        # ---- 价格位置 ----
        high_60 = np.nanmax(h[max(0, idx-60):idx+1]) if idx >= 0 else np.nanmax(h[-60:])
        low_60 = np.nanmin(l[max(0, idx-60):idx+1]) if idx >= 0 else np.nanmin(l[-60:])
        price_range = high_60 - low_60

        vol_ma_val = vol_ma[idx] if not np.isnan(vol_ma[idx]) and vol_ma[idx] > 0 else 1
        atr_val = atr[idx] if not np.isnan(atr[idx]) else 0
        rsi_val = rsi[idx] if not np.isnan(rsi[idx]) else 50
        adx_val = adx[idx] if not np.isnan(adx[idx]) else 0
        m20 = roc_20[idx] if not np.isnan(roc_20[idx]) else 0

        features = {
            # 动量特征
            'momentum_5d': float(roc_5[idx]) if not np.isnan(roc_5[idx]) else 0,
            'momentum_10d': float(roc_10[idx]) if not np.isnan(roc_10[idx]) else 0,
            'momentum_20d': float(m20),
            'momentum_60d': float(roc_60[idx]) if not np.isnan(roc_60[idx]) else 0,
            # 风险特征
            'volatility': float(atr_val / c[idx]) if c[idx] > 0 else 0,
            'rsi_14': float(rsi_val),
            'adx_14': float(adx_val),
            # 量能特征
            'macd_hist': float(macd_hist[idx]) if not np.isnan(macd_hist[idx]) else 0,
            'obv_slope': float(obv_slope),
            'vol_ratio': float(v[idx] / vol_ma_val),
            'price_position': float((c[idx] - low_60) / price_range) if price_range > 0 else 0.5,
            # 交互特征: 捕捉非线性关系
            'mom_vol_cross': float(m20 * atr_val / c[idx]) if c[idx] > 0 else 0,
            'adx_rsi_cross': float(adx_val * (rsi_val - 50) / 100),
        }
        return features
    except Exception:
        return None


# ============================================================
# 构建训练数据集
# ============================================================

def build_dataset(all_data, holding_days=20):
    """
    构建 (日期, 股票, 特征, 未来收益) 的训练集。

    每月末, 对每只股票:
      1. 使用截至当日的 K 线数据计算特征向量
      2. 计算未来 holding_days 的收益率作为标签

    参数:
        all_data: dict {code: DataFrame}
        holding_days: 持有天数 (默认20日 ≈ 1个月)

    返回值:
        DataFrame, 列 = [date, code] + FEATURE_NAMES + [forward_return]
        每行是一个样本 (某月某只股票的特征和标签)
    """
    # 取所有交易日期的并集
    all_dates = set()
    for df in all_data.values():
        all_dates.update(df.index.tolist())
    all_dates = sorted(all_dates)

    # 月末日期
    monthly_dates = []
    for i, d in enumerate(all_dates):
        if i + 1 < len(all_dates) and all_dates[i + 1].month != d.month:
            monthly_dates.append(d)

    records = []
    for calc_date in monthly_dates:
        for code, df in all_data.items():
            if calc_date not in df.index:
                continue
            idx = df.index.get_loc(calc_date)
            if idx < 80 or idx + holding_days >= len(df):
                continue

            # 只用该日期之前的数据计算特征
            sub = df.iloc[:idx + 1]
            features = calc_features(sub)
            if features is None:
                continue

            # 未来收益率 (标签)
            c_now = float(df['close'].iloc[idx])
            c_future = float(df['close'].iloc[idx + holding_days])
            if c_now <= 0:
                continue
            fwd_ret = c_future / c_now - 1

            record = {'date': calc_date, 'code': code, 'forward_return': fwd_ret}
            record.update(features)
            records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 滚动训练 + 预测
# ============================================================

def walk_forward_backtest(dataset, train_months=12, top_n=10):
    """
    Walk-Forward 滚动回测 (核心 ML 回测函数)。

    这是金融领域 ML 回测的标准方法:
      每月, 用过去 train_months 个月的数据训练模型,
      预测当月所有股票的未来收益, 选出 Top-N。

    为什么不用"全部数据训练, 一部分测试"?
      金融时间序列有很强的时序依赖性, 随机划分会导致未来信息泄露。
      Walk-Forward 保证模型只用历史数据预测未来。

    参数:
        dataset: build_dataset() 的输出
        train_months: 滚动训练窗口大小 (月数, 默认12)
        top_n: 选股数量 (默认10)

    返回值:
        dict {nav_log, prediction_log, final_nav, feature_cols}
    """
    dates = sorted(dataset['date'].unique())
    if len(dates) < train_months + 2:
        print(f"  数据期数不足: {len(dates)}期, 需要 {train_months + 2}期")
        return None

    nav = INITIAL_CASH
    nav_log = []
    prediction_log = []

    feature_cols = [c for c in FEATURE_NAMES if c in dataset.columns]

    for i in range(train_months, len(dates)):
        test_date = dates[i]                                  # 当月 (测试)
        train_dates = dates[max(0, i - train_months):i]       # 过去12个月 (训练)

        train_data = dataset[dataset['date'].isin(train_dates)]
        test_data = dataset[dataset['date'] == test_date]

        if len(train_data) < 50 or len(test_data) < top_n:
            nav_log.append({'date': test_date, 'nav': nav, 'return': 0})
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data['forward_return'].values
        X_test = test_data[feature_cols].values

        # ---- 训练模型 ----
        if HAS_LGB:
            # LightGBM: 高效梯度提升树, 适合表格数据
            model = lgb.LGBMRegressor(
                n_estimators=100,          # 100棵树
                max_depth=5,               # 最大深度, 控制过拟合
                learning_rate=0.05,        # 学习率
                subsample=0.8,             # 行采样, 防过拟合
                colsample_bytree=0.8,      # 列采样, 防过拟合
                min_child_samples=10,      # 叶子节点最小样本数
                verbose=-1,                # 不输出训练日志
            )
        elif HAS_GBR:
            # 备选: sklearn 的 GradientBoosting
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
            )
        else:
            nav_log.append({'date': test_date, 'nav': nav, 'return': 0})
            continue

        try:
            model.fit(X_train, y_train)
        except Exception:
            nav_log.append({'date': test_date, 'nav': nav, 'return': 0})
            continue

        # ---- 预测 + 选股 ----
        predictions = model.predict(X_test)
        test_data = test_data.copy()
        test_data['pred'] = predictions

        # 选预测值最高的 Top-N
        top = test_data.nlargest(top_n, 'pred')
        port_return = top['forward_return'].mean()
        nav *= (1 + port_return)

        nav_log.append({
            'date': test_date,
            'nav': nav,
            'return': port_return,
            'top_codes': top['code'].tolist(),
            'avg_pred': top['pred'].mean(),
        })

        prediction_log.append({
            'date': test_date,
            'train_size': len(train_data),
            'test_size': len(test_data),
            'top_n': len(top),
            'port_return': port_return,
        })

    return {
        'nav_log': nav_log,
        'prediction_log': prediction_log,
        'final_nav': nav,
        'feature_cols': feature_cols,
    }


def equal_weight_backtest(dataset, top_n=10, factor_name='momentum_20d'):
    """
    等权因子排序回测 (基准策略)。

    直接用某个因子排序选股, 没有 ML 模型。
    用于和 ML 选股做对比。

    参数:
        dataset: build_dataset() 的输出
        top_n: 选股数量
        factor_name: 排序因子 (默认动量20日)

    返回值:
        dict {nav_log, final_nav}
    """
    dates = sorted(dataset['date'].unique())
    nav = INITIAL_CASH
    nav_log = []

    for date in dates:
        day_data = dataset[dataset['date'] == date]
        if len(day_data) < top_n:
            nav_log.append({'date': date, 'nav': nav, 'return': 0})
            continue

        # 按因子值排序选股
        top = day_data.nlargest(top_n, factor_name)
        port_return = top['forward_return'].mean()
        nav *= (1 + port_return)
        nav_log.append({'date': date, 'nav': nav, 'return': port_return})

    return {'nav_log': nav_log, 'final_nav': nav}


def calc_metrics(result):
    """
    计算 ML 回测的绩效指标。
    逻辑与脚本5/6的 calc_metrics 相同。
    """
    nav_log = result['nav_log']
    if not nav_log:
        return {}

    navs = [x['nav'] for x in nav_log]
    rets = [x['return'] for x in nav_log if 'return' in x]

    total_return = navs[-1] / INITIAL_CASH - 1
    dates = [x['date'] for x in nav_log]
    days = (dates[-1] - dates[0]).days if len(dates) > 1 else 365
    years = days / 365.25 if days > 0 else 1
    ann_ret = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else total_return

    peak = navs[0]
    max_dd = 0
    for v in navs:
        if v > peak:
            peak = v
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)

    if rets:
        std_m = np.std(rets) if len(rets) > 1 else 1
        sharpe = (np.mean(rets) * 12 - 0.02) / (std_m * np.sqrt(12)) if std_m > 0 else 0
        win_rate = sum(1 for r in rets if r > 0) / len(rets) if rets else 0
    else:
        sharpe = 0
        win_rate = 0

    calmar = ann_ret / max_dd if max_dd > 0 else 0

    return {
        'total_return': total_return,
        'annual_return': ann_ret,
        'max_drawdown': max_dd,
        'sharpe': round(sharpe, 4),
        'calmar': round(calmar, 4),
        'win_rate': win_rate,
        'periods': len(rets),
    }


# ============================================================
# 主程序
# ============================================================

if __name__ == '__main__':
    start_date = '2023-01-01'
    end_date = '2025-12-31'

    print("=" * 70)
    print("ML增强多因子选股 - LightGBM")
    print("=" * 70)
    print("\n核心流程:")
    print("  1. TA-Lib 计算13维特征 (动量/波动/RSI/ADX/MACD/交互)")
    print("  2. 构建 (月末, 股票, 特征, 未来20日收益) 训练集")
    print("  3. 滚动训练: 过去12月训练 → 预测下月 → 选Top-N")
    print("  4. 对比: ML选股 vs 等权因子排序")

    # ---- 1. 加载数据 ----
    print(f"\n[1] 加载数据 ({start_date} ~ {end_date})...")
    t0 = time.time()
    all_data = batch_load_daily(start_date, end_date, min_bars=80)
    print(f"    {len(all_data)} 只标的, 耗时 {time.time()-t0:.1f}s")

    if len(all_data) < 10:
        print("  标的不足, 请确保数据库有足够数据")
        exit()

    # ---- 2. 构建数据集 ----
    print(f"\n[2] 构建特征 + 标签数据集...")
    t0 = time.time()
    dataset = build_dataset(all_data, holding_days=20)
    print(f"    {len(dataset)} 条样本, {dataset['date'].nunique()} 个月, "
          f"{dataset['code'].nunique()} 只股票, 耗时 {time.time()-t0:.1f}s")

    if len(dataset) < 100:
        print("  样本不足")
        exit()

    # 数据概览
    print(f"\n    特征统计:")
    for col in FEATURE_NAMES[:5]:
        if col in dataset.columns:
            vals = dataset[col].dropna()
            print(f"    {col:<20} mean={vals.mean():>8.3f}  std={vals.std():>8.3f}")
    print(f"    未来收益: mean={dataset['forward_return'].mean()*100:+.2f}%  "
          f"std={dataset['forward_return'].std()*100:.2f}%")

    # ---- 3. ML滚动回测 ----
    print(f"\n{'=' * 70}")
    print("[3] Walk-Forward 滚动回测")
    print(f"{'=' * 70}")

    top_n = 10
    train_months = 12

    print(f"\n  训练: {train_months}个月滚动窗口, 预测: 未来20日收益, 选股: Top-{top_n}")

    print(f"\n  [ML选股 - LightGBM]")
    t0 = time.time()
    r_ml = walk_forward_backtest(dataset, train_months=train_months, top_n=top_n)
    ml_time = time.time() - t0

    if r_ml and r_ml['nav_log']:
        m_ml = calc_metrics(r_ml)
        print(f"  总收益: {m_ml['total_return']*100:+.2f}% | "
              f"年化: {m_ml['annual_return']*100:+.2f}% | "
              f"最大回撤: {m_ml['max_drawdown']*100:.2f}%")
        print(f"  夏普: {m_ml['sharpe']:.2f} | "
              f"卡玛: {m_ml['calmar']:.2f} | "
              f"月度胜率: {m_ml['win_rate']*100:.1f}% | "
              f"耗时: {ml_time:.1f}s")

        # 打印预测记录 (前5期)
        if r_ml['prediction_log']:
            print(f"\n  预测记录 (前5期):")
            for p in r_ml['prediction_log'][:5]:
                print(f"    {p['date'].strftime('%Y-%m-%d')} | "
                      f"训练={p['train_size']}  测试={p['test_size']} | "
                      f"收益={p['port_return']*100:+.1f}%")

        # 特征重要性 (用最后一轮模型展示)
        if HAS_LGB:
            dates = sorted(dataset['date'].unique())
            feature_cols = [c for c in FEATURE_NAMES if c in dataset.columns]
            last_train = dataset[dataset['date'].isin(dates[-train_months - 1:-1])]
            if len(last_train) > 20:
                X = last_train[feature_cols].values
                y = last_train['forward_return'].values
                model = lgb.LGBMRegressor(n_estimators=100, max_depth=5,
                                          learning_rate=0.05, verbose=-1)
                model.fit(X, y)
                importances = model.feature_importances_
                fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
                print(f"\n  特征重要性 (最近一轮模型):")
                for fname, imp in fi[:8]:
                    bar = '#' * int(imp / max(importances) * 20)
                    print(f"    {fname:<20} {imp:>6.0f} {bar}")

    # ---- 4. 基准: 等权动量排序 ----
    print(f"\n  [基准 - 动量排序]")
    r_mom = equal_weight_backtest(dataset, top_n=top_n, factor_name='momentum_20d')
    m_mom = calc_metrics(r_mom)
    print(f"  总收益: {m_mom['total_return']*100:+.2f}% | "
          f"年化: {m_mom['annual_return']*100:+.2f}% | "
          f"最大回撤: {m_mom['max_drawdown']*100:.2f}%")
    print(f"  夏普: {m_mom['sharpe']:.2f} | "
          f"卡玛: {m_mom['calmar']:.2f} | "
          f"月度胜率: {m_mom['win_rate']*100:.1f}%")

    # ---- 5. 汇总对比 ----
    print(f"\n{'=' * 70}")
    print("ML选股 vs 动量排序 对比")
    print(f"{'=' * 70}")

    if r_ml:
        print(f"\n  {'指标':<16} {'动量排序':>14} {'ML选股':>14}")
        print(f"  {'-' * 46}")
        print(f"  {'总收益':<16} {m_mom['total_return']*100:>+13.2f}% {m_ml['total_return']*100:>+13.2f}%")
        print(f"  {'年化收益':<16} {m_mom['annual_return']*100:>+13.2f}% {m_ml['annual_return']*100:>+13.2f}%")
        print(f"  {'最大回撤':<16} {m_mom['max_drawdown']*100:>13.2f}% {m_ml['max_drawdown']*100:>13.2f}%")
        print(f"  {'夏普比率':<16} {m_mom['sharpe']:>14.2f} {m_ml['sharpe']:>14.2f}")
        print(f"  {'卡玛比率':<16} {m_mom['calmar']:>14.2f} {m_ml['calmar']:>14.2f}")
        print(f"  {'月度胜率':<16} {m_mom['win_rate']*100:>13.1f}% {m_ml['win_rate']*100:>13.1f}%")

    print("\n关键发现:")
    print("  - ML模型能发现因子之间的非线性组合关系")
    print("  - 滚动训练避免了未来信息泄露 (Walk-Forward)")
    print("  - 交互特征(动量*波动率)往往排名靠前 → 因子组合比单因子更有效")
    print("  - 特征重要性排名帮助理解模型决策逻辑")
    print("  - 注意: 样本量不足时ML容易过拟合, 需要足够的标的和时间跨度")
    print("\n课程总结:")
    print("  网格线: 固定网格 → 中枢网格 → 趋势联动 → 融合策略")
    print("  选股线: 因子评价 → 多因子打分 → 小市值轮动 → ML增强")
    print("  两线融合: 因子选标的 + 中枢做网格 = 完整量化策略体系")
