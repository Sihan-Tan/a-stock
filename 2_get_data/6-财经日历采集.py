# -*- coding: utf-8 -*-
"""
财经日历采集 - AkShare(百度财经日历) -> MySQL

采集全球重要经济事件，为量化策略提供"事件驱动"的数据基础。

事件驱动策略（Event-Driven Strategy）是量化投资的重要流派：
  - FOMC利率决议 -> 影响全球资产定价
  - CPI/PPI公布日 -> 通胀数据超预期引发市场波动
  - 非农就业数据 -> 美元和美股的重要驱动因素
  - PMI数据发布 -> 经济景气度风向标

这些事件发生时间和市场预期值的偏差（实际值 vs 预期值）
本身就是重要的交易信号。

数据源：AkShare news_economic_baidu()
  返回列: 日期, 时间, 国家, 事件, 实际, 预期, 前值, 重要性

运行：python 6-财经日历采集.py
"""
import sys
import os
import math
import pandas as pd
import akshare as ak

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import get_connection

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 只保留这些国家的事件（其他国家对A股影响较小）
COUNTRIES = {'中国', '美国', '欧元区', '日本', '英国'}

# 事件类型分类关键词
# 将不同的事件名称归类到统一的类别，便于后续按类别分析
EVENT_TYPE_MAP = {
    'interest_rate': ['利率', 'FOMC', '加息', '降息', 'LPR', '基准利率', '联邦基金'],
    'inflation': ['CPI', 'PPI', '通胀', '物价'],
    'employment': ['就业', '非农', '失业率', 'ADP'],
    'pmi': ['PMI', '采购经理'],
    'gdp': ['GDP', '国内生产总值'],
    'trade': ['贸易', '进出口', '出口', '进口'],
    'monetary': ['M2', '货币供应', '社融', '信贷'],
    'housing': ['房价', '房屋'],
    'retail': ['零售', '消费'],
    'industry': ['工业', '产出', '产值'],
}


def classify_event(event_name):
    """
    根据事件名称自动分类，便于后续按类别分析。

    分类逻辑：
      - 关键词匹配（由长到短匹配避免误判）
      - 匹配到第一个关键词即返回对应类别
      - 未匹配到任何关键词返回 'other'

    Args:
        event_name: 事件名称字符串

    Returns:
        str: 事件类别（如 'interest_rate', 'inflation', 'employment' 等）
    """
    for etype, keywords in EVENT_TYPE_MAP.items():
        for kw in keywords:
            if kw in event_name:
                return etype
    return 'other'


def _to_str(val):
    """
    将数值转为字符串，NaN 返回 None。

    AkShare返回的数据中，缺失数据通常表示为NaN，
    需要统一转换为None后再写入MySQL。

    Args:
        val: 输入值

    Returns:
        str 或 None
    """
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    return s if s else None


def fetch_and_save():
    """
    采集财经日历并写入数据库。

    采集策略：
      - 覆盖前7天到后30天的事件（历史数据用于验证，未来数据用于预测）
      - 逐天调用 API（AkShare的限制，不支持批量日期查询）
      - 自动识别列名（兼容AkShare版本变化）
      - 过滤国家，只保留中/美/欧/日/英

    为什么覆盖前7天？
      事件的实际值往往在事件发生后几天才能获取到，覆盖前7天
      可以补充历史事件的实际值，这些值在首次采集时可能还是NULL。

    Returns:
        int: 写入/更新的事件总数
    """
    print("采集财经日历 (AkShare news_economic_baidu)...")

    today = pd.Timestamp.now().normalize()
    # 前7天 + 未来30天，共38天
    dates = pd.date_range(today - pd.Timedelta(days=7), today + pd.Timedelta(days=30))
    frames = []
    for d in dates:
        date_str = d.strftime('%Y%m%d')
        try:
            # 百度财经日历接口，按天查询
            part = ak.news_economic_baidu(date=date_str)
            if part is not None and len(part) > 0:
                frames.append(part)
        except Exception:
            pass
    if not frames:
        print("  接口返回空")
        return 0

    df = pd.concat(frames, ignore_index=True)
    print("  原始数据: {} 条".format(len(df)))

    # 列名映射：AkShare返回的列名可能随版本变化，
    # 通过关键词匹配自动识别各列
    col_map = {}
    for col in df.columns:
        if '日期' in col:
            col_map['date'] = col
        elif '时间' in col:
            col_map['time'] = col
        elif '国家' in col or '地区' in col:
            col_map['country'] = col
        elif '事件' in col:
            col_map['event'] = col
        elif '实际' in col:
            col_map['actual'] = col
        elif '预期' in col:
            col_map['forecast'] = col
        elif '前值' in col:
            col_map['previous'] = col
        elif '重要' in col:
            col_map['importance'] = col

    if 'date' not in col_map or 'event' not in col_map:
        print("  列名无法识别: {}".format(list(df.columns)))
        return 0

    # 过滤国家：只保留对A股影响较大的经济体
    if 'country' in col_map:
        df = df[df[col_map['country']].isin(COUNTRIES)]
        print("  过滤后({}): {} 条".format('/'.join(COUNTRIES), len(df)))

    conn = get_connection()
    cursor = conn.cursor()

    sql = """
        INSERT INTO trade_calendar_event
        (event_date, event_time, title, country, category,
         importance, forecast_value, actual_value, previous_value, source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        actual_value = COALESCE(VALUES(actual_value), actual_value),
        forecast_value = COALESCE(VALUES(forecast_value), forecast_value),
        previous_value = COALESCE(VALUES(previous_value), previous_value),
        importance = VALUES(importance)
    """

    count = 0
    for _, row in df.iterrows():
        event_date = row[col_map['date']]
        if pd.isna(event_date):
            continue
        if hasattr(event_date, 'strftime'):
            event_date_str = event_date.strftime('%Y-%m-%d')
        else:
            event_date_str = str(event_date)[:10]

        event_time = str(row.get(col_map.get('time', ''), '')).strip() or None
        title = str(row[col_map['event']]).strip()
        if not title:
            continue

        country = str(row.get(col_map.get('country', ''), '')).strip() or ''
        importance = int(row.get(col_map.get('importance', ''), 1) or 1)
        category = classify_event(title)
        actual = _to_str(row.get(col_map.get('actual', ''), None))
        forecast = _to_str(row.get(col_map.get('forecast', ''), None))
        previous = _to_str(row.get(col_map.get('previous', ''), None))

        cursor.execute(sql, (
            event_date_str, event_time, title, country, category,
            importance, forecast, actual, previous, 'baidu_economic'
        ))
        count += 1

    conn.commit()
    cursor.close()
    conn.close()
    return count


def main():
    print("=" * 60)
    print("财经日历采集 -> MySQL")
    print("=" * 60)

    count = fetch_and_save()
    print("\n写入/更新 {} 条事件".format(count))

    # 统计数据概况
    from db_config import execute_query
    rows = execute_query("""
        SELECT country, COUNT(*) as cnt,
               MIN(event_date) as min_d, MAX(event_date) as max_d
        FROM trade_calendar_event
        GROUP BY country ORDER BY cnt DESC
    """)
    print("\ntrade_calendar_event 概况:")
    for r in rows:
        print("  {}: {} 条 ({} ~ {})".format(r['country'], r['cnt'], r['min_d'], r['max_d']))

    total = execute_query("SELECT COUNT(*) as c FROM trade_calendar_event")
    print("  总计: {} 条".format(total[0]['c']))

    print("\n" + "=" * 60)
    print("财经日历采集完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
