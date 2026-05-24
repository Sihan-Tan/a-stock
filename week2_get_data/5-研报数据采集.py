# -*- coding: utf-8 -*-
"""
研报数据采集 - AkShare(东方财富+同花顺) -> MySQL
全量A股研报采集，多线程并行

数据源：
  1. 东方财富 - 机构评级明细 (ak.stock_institute_recommend_detail)
     提供每次券商发布的评级/目标价记录，包含券商名称、分析师、目标价、评级等
  2. 同花顺 - 盈利预测一致预期 (ak.stock_profit_forecast_ths)
     提供EPS/净利润一致预期，聚合多家券商的预测数据

去重策略（核心）：
  东方财富的原始数据包含券商每次发布研究笔记的记录（晨报/点评/行业周报等），
  同一券商对同一股票可能每隔几天发布一篇，但评级和目标价不变。
  去重方式：只保存"观点变化"，即同一券商对同一股票的评级或目标价发生变化时才写入。

  为什么这样设计？
    如果全部保存，数据量会非常庞大且冗余（每天都有大量重复评级的报告）。
    只保存观点变化可以大幅减少数据量，同时更准确地反映券商态度变化。

采集范围：trade_stock_daily 中所有股票
跳过逻辑：7天内已采集的股票跳过

运行：python 5-研报数据采集.py
"""
import sys
import os
import time
import pandas as pd
import akshare as ak
import pymysql
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import get_connection, execute_query

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ============================================================
# 配置
# ============================================================
# 测试模式开关
TEST_MODE = False
TEST_STOCK = '600519.SH'

# 单只股票最大采集的研报行数（控制内存和API返回量）
MAX_RECOMMEND_ROWS = 50
# 并行线程数（研报数据采集对QPS敏感，不宜设置过大）
NUM_WORKERS = 4

_print_lock = threading.Lock()


def safe_print(msg):
    with _print_lock:
        print(msg)


# ============================================================
# 采集函数
# ============================================================

def fetch_institute_recommend(stock_code):
    """
    从东方财富获取机构推荐（评级/目标价）明细。

    机构评级说明：
      - 评级通常为：买入 > 增持 > 中性 > 减持 > 卖出
      - 目标价是券商对该股票未来某个时间点的价格预测
      - 评级上调或目标价提高通常是正面信号

    Args:
        stock_code: 股票代码，如 "600519.SH"

    Returns:
        pd.DataFrame: 包含评级详细数据的DataFrame
    """
    code_num = stock_code.split('.')[0]
    try:
        # 增加线程安全的延迟，避免触发AkShare的限流
        time.sleep(0.2)
        df = ak.stock_institute_recommend_detail(symbol=code_num)
        if df is None or len(df) == 0:
            return pd.DataFrame()

        # 检查必要的列是否存在（AkShare版本变化可能导致列名变更）
        required_columns = ['股票代码', '股票名称', '目标价', '评级', '机构', '分析师', '行业', '研报日期']
        if not all(col in df.columns for col in required_columns):
            return pd.DataFrame()

        # 重命名列为英文，方便后续处理
        df.columns = ['stock_code_raw', 'stock_name', 'target_price',
                       'rating', 'broker', 'analyst', 'industry', 'report_date']
        df = df.head(MAX_RECOMMEND_ROWS)

        # 安全处理数据类型
        try:
            df['target_price'] = pd.to_numeric(df['target_price'], errors='coerce')
        except Exception:
            df['target_price'] = None

        try:
            df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
        except Exception:
            df['report_date'] = None

        df['stock_code'] = stock_code
        return df
    except Exception as e:
        safe_print(f"  采集研报失败 {stock_code}: {str(e)[:100]}")
        return pd.DataFrame()


def deduplicate_recommend(stock_code, df):
    """
    智能去重：同一券商仅保留"观点变化"的记录。

    详细逻辑：
      东方财富的推荐数据包含大量"例行报告"，如每周晨报都推荐某只股票，
      但评级和目标价没有变化。这些数据对投资决策的增量信息很少。

      本函数按时间从旧到新遍历，仅当评级或目标价与上一次记录不同时才保留。
      例如：
        2025-01-01  券商A  买入  目标价100元  -> 保留（首次记录）
        2025-01-08  券商A  买入  目标价100元  -> 跳过（无变化）
        2025-02-01  券商A  买入  目标价120元  -> 保留（目标价变了）
        2025-03-01  券商A  增持  目标价120元  -> 保留（评级变了）

    Args:
        stock_code: 股票代码（仅用于日志）
        df: 原始推荐数据DataFrame

    Returns:
        pd.DataFrame: 去重后的数据
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # 按日期升序排列，以便从旧到新遍历
    df_sorted = df.sort_values('report_date', ascending=True).reset_index(drop=True)

    # last_opinion 记录各券商上一条有效记录
    # key=券商名称, value={'rating': , 'target_price': }
    last_opinion = {}
    keep_indices = []

    for idx, row in df_sorted.iterrows():
        broker = str(row['broker'])
        rating = str(row['rating'])
        tp = row['target_price'] if pd.notna(row['target_price']) else None

        key = broker
        prev = last_opinion.get(key)

        # 如果该券商没有历史记录，或者评级/目标价发生了变化，则保留
        if prev is None or prev['rating'] != rating or prev['target_price'] != tp:
            keep_indices.append(idx)
            last_opinion[key] = {'rating': rating, 'target_price': tp}

    return df_sorted.loc[keep_indices]


def fetch_profit_forecast(stock_code):
    """
    从同花顺获取盈利预测一致预期。

    一致预期 = 多家券商预测值的平均值（或中位数），
    反映了市场对该股票未来盈利的集体看法。

    主要包括：
      - 预测年报每股收益(EPS)：未来1-2年的EPS预测
      - 预测年报净利润：未来1-2年的净利润预测

    一致预期的意义：
      - 上调一致预期通常推动股价上涨
      - 下调一致预期通常导致股价下跌
      - 实际业绩 vs 一致预期的偏差 = 业绩超预期/不及预期

    Args:
        stock_code: 股票代码

    Returns:
        dict: {'预测年报每股收益': DataFrame, '预测年报净利润': DataFrame}
    """
    code_num = stock_code.split('.')[0]
    forecasts = {}
    try:
        time.sleep(0.2)
        # 分别获取EPS和净利润的一致预期
        for indicator in ['预测年报每股收益', '预测年报净利润']:
            try:
                df = ak.stock_profit_forecast_ths(symbol=code_num, indicator=indicator)
                if df is not None and len(df) > 0:
                    # 检查必要的列是否存在
                    required_columns = ['年份', '分析师数量', '最低', '均值', '最高', '行业均值']
                    if all(col in df.columns for col in required_columns):
                        df.columns = ['year', 'analyst_count', 'min_val', 'mean_val', 'max_val', 'industry_avg']
                        forecasts[indicator] = df
            except Exception as e:
                safe_print(f"  采集盈利预测失败 {stock_code}({indicator}): {str(e)[:100]}")
                pass
    except Exception as e:
        safe_print(f"  采集盈利预测失败 {stock_code}: {str(e)[:100]}")
    return forecasts


def save_recommend_to_mysql(stock_code, recommend_df):
    """
    将去重后的机构评级写入 trade_report_consensus 表。

    增量去重说明：
      除了在上一步(deduplicate_recommend)中对新数据进行去重，
      还需要与数据库中已有的历史记录进行去重比较。
      如果数据库中该券商已有相同的评级和目标价，则跳过不写入。

    Args:
        stock_code: 股票代码
        recommend_df: 去重后的推荐数据DataFrame

    Returns:
        int: 写入数据库的记录数
    """
    if recommend_df is None or len(recommend_df) == 0:
        return 0

    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    sql_insert = """
        INSERT INTO trade_report_consensus
        (stock_code, broker, report_date, rating, target_price,
         eps_forecast_current, eps_forecast_next, revenue_forecast, source_file)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        rating=VALUES(rating), target_price=VALUES(target_price)
    """

    # 查询该股票已有的各券商最新观点（用于增量去重）
    existing = {}
    rows = execute_query("""
        SELECT broker, rating, target_price
        FROM trade_report_consensus
        WHERE stock_code=%s AND source_file='eastmoney'
        ORDER BY report_date DESC
    """, (stock_code,))
    for r in rows:
        b = r['broker']
        if b not in existing:
            existing[b] = {
                'rating': r['rating'],
                'target_price': r['target_price']
            }

    count = 0
    for _, row in recommend_df.iterrows():
        broker = str(row['broker'])[:50]
        rating = str(row['rating'])[:20]
        report_date = row['report_date'].strftime('%Y-%m-%d') if pd.notna(row['report_date']) else None
        target_price = float(row['target_price']) if pd.notna(row['target_price']) else None

        # 增量去重：如果数据库中该券商已有相同评级和目标价，跳过
        prev = existing.get(broker)
        if prev:
            prev_tp = float(prev['target_price']) if prev['target_price'] is not None else None
            if prev['rating'] == rating and prev_tp == target_price:
                continue

        cursor.execute(sql_insert, (
            row['stock_code'], broker, report_date, rating, target_price,
            None, None, None, 'eastmoney'
        ))
        count += 1
        # 更新内存中的记录，以便同一批次后续数据比较
        existing[broker] = {'rating': rating, 'target_price': target_price}

    conn.commit()
    cursor.close()
    conn.close()
    return count


def save_forecast_to_mysql(stock_code, forecasts):
    """
    将盈利预测一致预期写入MySQL。

    一致预期数据的典型结构：
      - 第一行：本年的一致预期（如果已有年报则为下一年）
      - 第二行：下一年的预期
      - EPS_current 和 EPS_next 分别对应这两行

    Args:
        stock_code: 股票代码
        forecasts: fetch_profit_forecast 返回的预测数据字典

    Returns:
        int: 写入记录数（成功=1，失败=0）
    """
    if not forecasts:
        return 0

    eps_df = forecasts.get('预测年报每股收益')
    if eps_df is None or len(eps_df) == 0:
        return 0

    profit_df = forecasts.get('预测年报净利润')

    conn = get_connection()
    cursor = conn.cursor()

    sql = """
        INSERT INTO trade_report_consensus
        (stock_code, broker, report_date, rating, target_price,
         eps_forecast_current, eps_forecast_next, revenue_forecast, source_file)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        eps_forecast_current=VALUES(eps_forecast_current),
        eps_forecast_next=VALUES(eps_forecast_next),
        revenue_forecast=VALUES(revenue_forecast)
    """

    today = datetime.now().strftime('%Y-%m-%d')
    # EPS本年预期 = 第一行的均值
    eps_current = float(eps_df.iloc[0]['mean_val']) if len(eps_df) > 0 else None
    # EPS下年预期 = 第二行的均值
    eps_next = float(eps_df.iloc[1]['mean_val']) if len(eps_df) > 1 else None
    profit_current = None
    if profit_df is not None and len(profit_df) > 0:
        profit_current = float(profit_df.iloc[0]['mean_val'])

    analyst_count = int(eps_df.iloc[0]['analyst_count']) if len(eps_df) > 0 else 0

    cursor.execute(sql, (
        stock_code,
        "一致预期({}家)".format(analyst_count),
        today, None, None,
        eps_current, eps_next, profit_current,
        'ths_consensus'
    ))

    conn.commit()
    cursor.close()
    conn.close()
    return 1


# ============================================================
# 主流程
# ============================================================

def get_all_stocks():
    """从行情数据表获取全量股票列表"""
    rows = execute_query("SELECT DISTINCT stock_code FROM trade_stock_daily")
    return [r['stock_code'] for r in rows]


def get_recently_collected():
    """
    获取7天内已采集研报的股票。

    为什么跳过7天内已采集的？
      券商研报具有一定的时效性，7天内同一家券商很少对同一只股票
      发布多份不同评级的报告。7天的冷却期可以在覆盖频率和API调用
      成本之间取得合理平衡。
    """
    rows = execute_query("""
        SELECT DISTINCT stock_code FROM trade_report_consensus
        WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)
    """)
    return {r['stock_code'] for r in rows}


def process_one_stock(stock_code):
    """
    采集并保存单只股票的研报数据。

    包含两个步骤：
      1. 采集机构推荐（评级+目标价）-> 去重 -> 保存
      2. 采集盈利预测一致预期 -> 保存

    Args:
        stock_code: 股票代码

    Returns:
        tuple: (stock_code, recommend新增数, forecast新增数)
    """
    recommend_count = 0
    forecast_count = 0

    try:
        # Step 1: 采集机构推荐数据
        raw_df = fetch_institute_recommend(stock_code)
        if len(raw_df) > 0:
            deduped_df = deduplicate_recommend(stock_code, raw_df)
            if len(deduped_df) > 0:
                recommend_count = save_recommend_to_mysql(stock_code, deduped_df)

        # 两个API调用之间增加延迟，避免限流
        time.sleep(0.3)

        # Step 2: 采集盈利预测数据
        forecasts = fetch_profit_forecast(stock_code)
        if forecasts:
            forecast_count = save_forecast_to_mysql(stock_code, forecasts)

        return stock_code, recommend_count, forecast_count
    except Exception as e:
        safe_print(f"  处理股票失败 {stock_code}: {str(e)[:100]}")
        return stock_code, 0, 0


def main():
    print("=" * 60)
    print("研报数据采集（AkShare - 全量A股）")
    print("=" * 60)

    if TEST_MODE:
        stock_list = [TEST_STOCK]
        print("[测试模式] 只采集 {}".format(TEST_STOCK))
    else:
        stock_list = get_all_stocks()
        print("全量股票: {} 只".format(len(stock_list)))

        collected = get_recently_collected()
        if collected:
            stock_list = [c for c in stock_list if c not in collected]
            print("跳过近7天已采集: {} 只, 待采集: {} 只".format(len(collected), len(stock_list)))

    if not stock_list:
        print("无股票需要采集")
        return

    total = len(stock_list)
    total_recommend = 0
    total_forecast = 0
    done = 0
    start_time = time.time()

    print("\n开始采集 ({} 线程)...".format(NUM_WORKERS))

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_one_stock, code): code
            for code in stock_list
        }

        for future in as_completed(futures):
            try:
                code, rec, fct = future.result()
                done += 1
                total_recommend += rec
                total_forecast += fct
            except Exception:
                done += 1

            if done % 100 == 0 or done == total:
                elapsed = time.time() - start_time
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / speed if speed > 0 else 0
                sys.stdout.write(
                    "\r  [{}/{}] {:.1f}% | {:.1f}/s | ETA {:.0f}s | "
                    "recommend {} forecast {}    ".format(
                        done, total, done * 100 / total,
                        speed, eta, total_recommend, total_forecast
                    )
                )
                sys.stdout.flush()

    print()

    elapsed = time.time() - start_time
    result = execute_query("SELECT COUNT(*) as cnt FROM trade_report_consensus")
    total_db = result[0]['cnt'] if result else 0

    print("\n" + "=" * 60)
    print("研报采集完成! 耗时 {:.1f} 秒".format(elapsed))
    print("  处理: {}/{} 只股票".format(done, total))
    print("  评级: {} 条(去重后新增), 一致预期: {} 条".format(total_recommend, total_forecast))
    print("  trade_report_consensus 总计 {} 条".format(total_db))
    print("=" * 60)


if __name__ == '__main__':
    main()
