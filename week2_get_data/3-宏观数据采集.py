# -*- coding: utf-8 -*-
"""
宏观经济数据采集 - 采集宏观经济指标存入MySQL

指标分组及其对A股投资的意义:
  通胀指标: CPI同比、PPI同比
    - CPI > 3% -> 可能加息紧缩 -> 利空股市
    - PPI > 0 -> 工业品涨价 -> 利好上游周期股
  景气指标: PMI(制造业)
    - PMI > 50 -> 经济扩张 -> 利好股市
    - PMI < 50 -> 经济收缩 -> 利空股市
  流动性: M2同比增速、社融规模增量
    - M2高增 -> 资金充裕 -> 利好股市
    - 社融大增 -> 实体融资活跃 -> 经济向好
  利率: LPR(1年/5年)、10年期国债收益率(中/美)
    - 利率下降 -> 估值提升(DCF模型折现率降低) -> 利好成长股
    - 中美利差 -> 影响汇率和外资流向

数据源(AkShare):
  - CPI: macro_china_cpi()     - 国家统计局月度CPI数据
  - PPI: macro_china_ppi()     - 国家统计局月度PPI数据
  - PMI: macro_china_pmi()     - 国家统计局月度PMI数据
  - M2:  macro_china_supply_of_money()  - 央行货币供应量
  - 社融: macro_china_shrzgm() - 社会融资规模增量
  - LPR: macro_china_lpr()     - 贷款市场报价利率
  - 国债: bond_zh_us_rate()    - 中美国债收益率(日频)

运行: python 3-宏观数据采集.py
"""
import re
import sys
import os
import time
import pandas as pd
import akshare as ak
import urllib3

# 忽略 SSL 警告（AkShare某些接口的SSL证书可能不完整）
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import get_connection, execute_query


def _parse_cn_date(series):
    """
    解析中文日期格式: '2026年01月份' / '202601' / '2026.01' -> datetime

    AkShare返回的日期格式不统一，不同接口使用不同格式：
      - '2026年01月份'  (CPI、PPI等国家统计局数据)
      - '202601'        (社融等央行数据)
      - '2026-01'       (LPR等)
    此函数统一解析为 pandas Timestamp。

    Args:
        series: pandas Series，包含日期字符串

    Returns:
        pandas Series of Timestamp
    """
    def _parse_one(s):
        if pd.isna(s):
            return pd.NaT
        s = str(s).strip()
        # 带分隔符: '2026年01月' / '2026.01'
        m = re.match(r'(\d{4})\D+(\d{1,2})', s)
        if m:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
        # 纯数字: '202601'
        m = re.match(r'^(\d{4})(\d{2})$', s)
        if m:
            return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
        return pd.NaT
    return series.apply(_parse_one)


def _find_col(columns, keywords):
    """
    按关键词列表在列名中查找匹配列。

    为什么需要这个函数？
      AkShare的数据列名可能随版本变化，如CPI数据列名可能是
      '全国-同比增长'、'同比增长'或'同比'。
      通过多关键词匹配提高兼容性。

    Args:
        columns: DataFrame的列名列表
        keywords: 要搜索的关键词列表

    Returns:
        str: 第一个匹配的列名，未找到返回None
    """
    for kw in keywords:
        for col in columns:
            if kw in col:
                return col
    return None


def fetch_cpi():
    """
    采集CPI(居民消费价格指数)同比数据。

    CPI衡量居民购买一篮子商品和服务的价格变化，
    是衡量通胀的核心指标。CPI同比 > 3%通常被视为通胀压力信号。

    Returns:
        pd.DataFrame: 包含 date 和 cpi_yoy 两列
    """
    print("  采集CPI...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)
            df = ak.macro_china_cpi()
            if df is None or len(df) == 0:
                print(f"    尝试 {attempt+1}/{max_retries}: 无数据")
                time.sleep(1)
                continue
            date_col = df.columns[0]
            # 查找同比数据列，不同版本的列名可能不同
            value_col = _find_col(df.columns, ['全国-同比增长', '同比增长', '同比'])
            if value_col is None:
                value_col = df.columns[2] if len(df.columns) > 2 else df.columns[1]
            result = pd.DataFrame({
                'date': _parse_cn_date(df[date_col]),
                'cpi_yoy': pd.to_numeric(df[value_col], errors='coerce')
            }).dropna()
            print(f"    CPI: {len(result)} 条")
            return result
        except Exception as e:
            print(f"    尝试 {attempt+1}/{max_retries}: 错误 - {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print("    CPI采集失败，返回空数据")
                return pd.DataFrame()


def fetch_ppi():
    """
    采集PPI(工业生产者出厂价格指数)同比数据。

    PPI衡量工业企业产品出厂价格变动趋势，
    与企业盈利高度相关。PPI上行利好上游周期行业（钢铁、有色、化工）。
    同时PPI与CPI的剪刀差（PPI-CPI）反映企业利润空间。

    Returns:
        pd.DataFrame: 包含 date 和 ppi_yoy 两列
    """
    print("  采集PPI...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)
            df = ak.macro_china_ppi()
            if df is None or len(df) == 0:
                print(f"    尝试 {attempt+1}/{max_retries}: 无数据")
                time.sleep(1)
                continue
            date_col = df.columns[0]
            value_col = _find_col(df.columns, ['当月同比增长', '同比增长', '同比'])
            if value_col is None:
                value_col = df.columns[2] if len(df.columns) > 2 else df.columns[1]
            result = pd.DataFrame({
                'date': _parse_cn_date(df[date_col]),
                'ppi_yoy': pd.to_numeric(df[value_col], errors='coerce')
            }).dropna()
            print(f"    PPI: {len(result)} 条")
            return result
        except Exception as e:
            print(f"    尝试 {attempt+1}/{max_retries}: 错误 - {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print("    PPI采集失败，返回空数据")
                return pd.DataFrame()


def fetch_pmi():
    """
    采集PMI(制造业采购经理指数)数据。

    PMI是经济景气度的先行指标：
      - > 50 表示制造业扩张（利好）
      - < 50 表示制造业收缩（利空）
      - 连续数月 > 50 通常意味着经济处于上行周期
      - 连续数月 < 50 可能预示经济衰退

    Returns:
        pd.DataFrame: 包含 date 和 pmi 两列
    """
    print("  采集PMI...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)
            df = ak.macro_china_pmi()
            if df is None or len(df) == 0:
                print(f"    尝试 {attempt+1}/{max_retries}: 无数据")
                time.sleep(1)
                continue
            date_col = df.columns[0]
            value_col = _find_col(df.columns, ['制造业-指标', '制造业', 'PMI'])
            if value_col is None:
                value_col = df.columns[1]
            result = pd.DataFrame({
                'date': _parse_cn_date(df[date_col]),
                'pmi': pd.to_numeric(df[value_col], errors='coerce')
            }).dropna()
            print(f"    PMI: {len(result)} 条")
            return result
        except Exception as e:
            print(f"    尝试 {attempt+1}/{max_retries}: 错误 - {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print("    PMI采集失败，返回空数据")
                return pd.DataFrame()


def fetch_m2():
    """
    采集M2(广义货币供应量)同比增速。

    M2 = 流通中的现金 + 企业活期存款 + 居民储蓄存款 + 企业定期存款
    M2增速反映货币政策宽松程度：
      - M2高增 -> 流动性充裕 -> 利好股市和房价
      - M2增速下降 -> 货币收紧 -> 利空资产价格
    通常M2增速与名义GDP增速之差反映了货币的超发程度。

    Returns:
        pd.DataFrame: 包含 date 和 m2_yoy 两列
    """
    print("  采集M2...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)
            df = ak.macro_china_supply_of_money()
            if df is None or len(df) == 0:
                print(f"    尝试 {attempt+1}/{max_retries}: 无数据")
                time.sleep(1)
                continue
            date_col = df.columns[0]
            value_col = _find_col(df.columns, ['M2）同比增长', 'M2)同比', 'M2同比'])
            if value_col is None:
                value_col = df.columns[2] if len(df.columns) > 2 else df.columns[1]
            result = pd.DataFrame({
                'date': _parse_cn_date(df[date_col]),
                'm2_yoy': pd.to_numeric(df[value_col], errors='coerce')
            }).dropna()
            print(f"    M2: {len(result)} 条")
            return result
        except Exception as e:
            print(f"    尝试 {attempt+1}/{max_retries}: 错误 - {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print("    M2采集失败，返回空数据")
                return pd.DataFrame()


def fetch_shrzgm():
    """
    采集社会融资规模增量(亿元)。

    社融是实体经济从金融体系获得的资金总量,
    包括人民币贷款、外币贷款、委托贷款、信托贷款、企业债券融资等。
    社融增量大说明企业融资需求旺盛，经济前景好。
    社融增量连续萎缩则可能预示经济下行风险。

    Returns:
        pd.DataFrame: 包含 date 和 shrzgm 两列
    """
    print("  采集社融...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 社融接口较慢，增加延迟
            time.sleep(1)
            df = ak.macro_china_shrzgm()
            if df is None or len(df) == 0:
                print(f"    尝试 {attempt+1}/{max_retries}: 无数据")
                time.sleep(2)
                continue
            date_col = df.columns[0]  # 月份 (格式: 202512)
            total_col = df.columns[1]  # 社会融资规模增量
            result = pd.DataFrame({
                'date': _parse_cn_date(df[date_col]),
                'shrzgm': pd.to_numeric(df[total_col], errors='coerce')
            }).dropna()
            print(f"    社融: {len(result)} 条")
            return result
        except Exception as e:
            print(f"    尝试 {attempt+1}/{max_retries}: 错误 - {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(3)
            else:
                print("    社融采集失败，返回空数据")
                return pd.DataFrame()


def fetch_lpr():
    """
    采集LPR利率(1年/5年)。

    LPR(贷款市场报价利率)是商业银行对其最优质客户的贷款利率，
    是央行货币政策传导的核心工具：
      - LPR下调 -> 降低企业融资成本 -> 利好股市
      - 1年期LPR主要影响企业短期贷款
      - 5年期LPR主要影响房贷利率，与地产板块高度相关
      - LPR下调对高负债行业（地产、基建）利好更显著

    Returns:
        pd.DataFrame: 包含 date, lpr_1y, lpr_5y 三列
    """
    print("  采集LPR...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)
            df = ak.macro_china_lpr()
            if df is None or len(df) == 0:
                print(f"    尝试 {attempt+1}/{max_retries}: 无数据")
                time.sleep(1)
                continue
            df['date'] = pd.to_datetime(df['TRADE_DATE'])
            df['lpr_1y'] = pd.to_numeric(df['LPR1Y'], errors='coerce')
            df['lpr_5y'] = pd.to_numeric(df['LPR5Y'], errors='coerce')
            result = df[['date', 'lpr_1y', 'lpr_5y']].dropna()
            print(f"    LPR: {len(result)} 条")
            return result
        except Exception as e:
            print(f"    尝试 {attempt+1}/{max_retries}: 错误 - {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print("    LPR采集失败，返回空数据")
                return pd.DataFrame()


def fetch_bond_yield():
    """
    采集中美10年期国债收益率(日频)，写入 trade_rate_daily 表。

    10年期国债收益率是无风险利率的基准，影响：
      - 中国10年期国债收益率：反映国内资金成本，下行利好股市
      - 美国10年期国债收益率：全球资产定价锚，上行压制高估值股票
      - 中美利差(中-美)：影响人民币汇率和外资流入
      - 利率上行时，高估值成长股承压（DCF模型敏感性高）
      - 利率下行时，金融、地产股受益（负债成本下降）

    Returns:
        int: 写入数据库的记录数
    """
    print("  采集国债收益率...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)
            # 近3年数据
            start = (pd.Timestamp.now() - pd.DateOffset(years=3)).strftime('%Y%m%d')
            df = ak.bond_zh_us_rate(start_date=start)
            if df is None or len(df) == 0:
                print(f"    尝试 {attempt+1}/{max_retries}: 无数据")
                time.sleep(1)
                continue

            # 根据列位置获取中美国债收益率
            # 列结构: 日期 | ... | 中国国债10年 | ... | 美国国债10年 | ...
            cols = df.columns.tolist()
            date_col = cols[0]
            cn10_col = cols[3]   # 中国国债收益率10年
            us10_col = cols[9]   # 美国国债收益率10年

            conn = get_connection()
            cursor = conn.cursor()
            sql = """
                INSERT INTO trade_rate_daily (rate_date, cn_bond_10y, us_bond_10y, data_source)
                VALUES (%s, %s, %s, 'akshare')
                ON DUPLICATE KEY UPDATE
                cn_bond_10y=COALESCE(VALUES(cn_bond_10y), cn_bond_10y),
                us_bond_10y=COALESCE(VALUES(us_bond_10y), us_bond_10y)
            """
            count = 0
            for _, row in df.iterrows():
                d = row[date_col]
                if pd.isna(d):
                    continue
                cn = float(row[cn10_col]) if pd.notna(row[cn10_col]) else None
                us = float(row[us10_col]) if pd.notna(row[us10_col]) else None
                if cn is None and us is None:
                    continue
                cursor.execute(sql, (pd.Timestamp(d).strftime('%Y-%m-%d'), cn, us))
                count += 1

            conn.commit()
            cursor.close()
            conn.close()
            print(f"    国债收益率: {count} 条写入 trade_rate_daily")
            return count
        except Exception as e:
            print(f"    尝试 {attempt+1}/{max_retries}: 错误 - {str(e)[:100]}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print("    国债收益率采集失败")
                return 0


# ==================== 月度数据合并写入 ====================

def merge_and_save(dfs):
    """
    合并月度宏观数据并写入MySQL。

    将CPI/PPI/PMI/M2/社融/LPR六个月度指标按月份合并成一张表，
    每条记录代表一个月份的所有宏观指标值。

    合并策略：
      - 每个指标取月份最后一条有效值
      - 用outer join确保不丢失任何指标的数据
      - 只保留近10年的数据

    Args:
        dfs: list of DataFrame，包含各指标的采集结果

    Returns:
        int: 写入的记录数
    """
    merged = None
    for df in dfs:
        if df is None or len(df) == 0:
            continue
        df = df.copy()
        # 按月份聚合，取每月最后一条数据
        df['month'] = df['date'].dt.to_period('M').dt.to_timestamp('M')
        df = df.drop(columns=['date']).groupby('month').last().reset_index()
        if merged is None:
            merged = df
        else:
            # outer join确保所有指标都保留
            merged = pd.merge(merged, df, on='month', how='outer')

    if merged is None or len(merged) == 0:
        print("  无数据可保存")
        return 0

    merged = merged.sort_values('month').reset_index(drop=True)

    # 只保留近10年
    cutoff = pd.Timestamp.now() - pd.DateOffset(years=10)
    merged = merged[merged['month'] >= cutoff]

    conn = get_connection()
    cursor = conn.cursor()

    sql = """
        INSERT INTO trade_macro_indicator
        (indicator_date, cpi_yoy, ppi_yoy, pmi, m2_yoy, shrzgm, lpr_1y, lpr_5y, data_source)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        cpi_yoy=COALESCE(VALUES(cpi_yoy), cpi_yoy),
        ppi_yoy=COALESCE(VALUES(ppi_yoy), ppi_yoy),
        pmi=COALESCE(VALUES(pmi), pmi),
        m2_yoy=COALESCE(VALUES(m2_yoy), m2_yoy),
        shrzgm=COALESCE(VALUES(shrzgm), shrzgm),
        lpr_1y=COALESCE(VALUES(lpr_1y), lpr_1y),
        lpr_5y=COALESCE(VALUES(lpr_5y), lpr_5y)
    """

    def _val(row, col):
        v = row.get(col)
        return float(v) if pd.notna(v) else None

    count = 0
    for _, row in merged.iterrows():
        cursor.execute(sql, (
            row['month'].strftime('%Y-%m-%d'),
            _val(row, 'cpi_yoy'), _val(row, 'ppi_yoy'),
            _val(row, 'pmi'), _val(row, 'm2_yoy'),
            _val(row, 'shrzgm'), _val(row, 'lpr_1y'), _val(row, 'lpr_5y'),
            'akshare'
        ))
        count += 1

    conn.commit()
    cursor.close()
    conn.close()
    return count


def main():
    print("=" * 60)
    print("宏观经济数据采集 -> MySQL")
    print("=" * 60)

    # Step 1: 采集月度指标
    print("\n[1/2] 采集月度宏观指标...")
    df_cpi = fetch_cpi()
    df_ppi = fetch_ppi()
    df_pmi = fetch_pmi()
    df_m2 = fetch_m2()
    df_shrzgm = fetch_shrzgm()
    df_lpr = fetch_lpr()

    all_dfs = [df_cpi, df_ppi, df_pmi, df_m2, df_shrzgm, df_lpr]
    names = ['CPI', 'PPI', 'PMI', 'M2', '社融', 'LPR']
    ok = sum(1 for df in all_dfs if len(df) > 0)
    print(f"\n采集结果: {ok}/{len(all_dfs)} 项成功")

    print("\n合并并写入MySQL...")
    count = merge_and_save(all_dfs)
    print(f"写入 {count} 条月度宏观指标")

    # Step 2: 采集日频利率指标
    print(f"\n[2/2] 采集日频利率指标...")
    fetch_bond_yield()

    # 打印数据库概况
    summary = execute_query("""
        SELECT COUNT(*) as cnt,
               MIN(indicator_date) as min_date, MAX(indicator_date) as max_date
        FROM trade_macro_indicator
    """)
    if summary:
        r = summary[0]
        print(f"\ntrade_macro_indicator: {r['cnt']} 期 ({r['min_date']} ~ {r['max_date']})")

    rate_summary = execute_query("""
        SELECT COUNT(*) as cnt,
               MIN(rate_date) as min_date, MAX(rate_date) as max_date
        FROM trade_rate_daily
    """)
    if rate_summary:
        r = rate_summary[0]
        print(f"trade_rate_daily: {r['cnt']} 日 ({r['min_date']} ~ {r['max_date']})")

    print("\n" + "=" * 60)
    print("宏观数据采集完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
