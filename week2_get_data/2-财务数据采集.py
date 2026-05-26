# -*- coding: utf-8 -*-
"""
财务数据采集 - 使用MiniQMT(xtquant)下载全量A股财务数据存入MySQL

从资产负债表、利润表、现金流量表、每股指标中提取常用财务指标，
写入 trade_stock_financial 表。

核心财务指标说明：
  - ROE (净资产收益率): 净利润/净资产，衡量股东权益的回报率，巴菲特最看重的指标
  - ROA (总资产回报率): 净利润/总资产，衡量公司全部资产的盈利能力
  - 毛利率: (营业收入-营业成本)/营业收入，反映产品定价能力和成本控制
  - 净利率: 净利润/营业收入，衡量最终盈利水平
  - 资产负债率: 总负债/总资产，衡量财务杠杆和偿债风险
  - 流动比率: 流动资产/流动负债，衡量短期偿债能力

功能：
  1. 获取沪深A股全量股票列表
  2. 跳过数据库中已有财务数据的股票（断点续传）
  3. 批量下载（每批50只），大幅减少QMT调用次数
  4. 提取ROE/毛利率/资产负债率等核心指标
  5. 写入MySQL（ON DUPLICATE KEY UPDATE）

优化（相比逐只下载）：
  - 重启后跳过已采集的股票，不重复下载
  - 50只/批 批量下载，比逐只下载快约10倍
  - 每批共用一个DB连接，减少连接开销

模式：
  - TEST_MODE = True  -> 只采集1只股票(贵州茅台)
  - TEST_MODE = False -> 采集沪深A股全量

运行：python 2-财务数据采集.py
环境：需安装QMT并配置好xtquant, pip install pymysql python-dotenv
"""
import sys
import os
import time
from datetime import datetime, date

import pandas as pd
from xtquant import xtdata

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
# 测试模式开关: True=只采贵州茅台, False=全量采集
TEST_MODE = False
# 测试用股票代码
TEST_STOCK = '600519.SH'

# 板块名称
SECTOR = '沪深A股'
# 每批处理的股票数量：在QMT调用次数和内存占用之间取平衡
BATCH_SIZE = 500
# 财务数据起始日期（建议至少2015年以后，太早的数据参考价值有限）
DATA_START = '20150101'
DATA_END = date.today().strftime('%Y%m%d')

# xtquant 财务数据表名列表
# Balance=资产负债表, Income=利润表, CashFlow=现金流量表
# PershareIndex=每股指标, Capital=股本结构
TABLE_LIST = ['Balance', 'Income', 'CashFlow', 'PershareIndex', 'Capital']

# UPSERT语句：如果主键(stock_code + report_date)冲突则更新指标值
INSERT_SQL = """
    INSERT INTO trade_stock_financial
    (stock_code, report_date, revenue, net_profit, eps, roe, roa,
     gross_margin, net_margin, debt_ratio, current_ratio,
     operating_cashflow, total_assets, total_equity, data_source)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE
    revenue=VALUES(revenue), net_profit=VALUES(net_profit), eps=VALUES(eps),
    roe=VALUES(roe), roa=VALUES(roa), gross_margin=VALUES(gross_margin),
    net_margin=VALUES(net_margin), debt_ratio=VALUES(debt_ratio),
    current_ratio=VALUES(current_ratio), operating_cashflow=VALUES(operating_cashflow),
    total_assets=VALUES(total_assets), total_equity=VALUES(total_equity)
"""


# ============================================================
# 工具函数
# ============================================================

def normalize_timetag(ts_val):
    """
    将xtquant的m_timetag转换为YYYYMMDD字符串。

    xtquant使用多种时间戳格式：
      - 8位数字字符串如 '20231231'
      - Unix时间戳（秒级10位或毫秒级13位）
      - 浮点数

    Args:
        ts_val: xtquant原始时间标签值

    Returns:
        str: YYYYMMDD格式日期字符串，转换失败返回None
    """
    if ts_val is None:
        return None
    s = str(ts_val).strip()
    # 已经是8位数字格式，直接返回
    if len(s) == 8 and s.isdigit():
        return s
    try:
        v = float(s)
        if v == 0:
            return None
        # 13位时间戳（毫秒级）需要除以1000转为秒级
        if v > 1e12:
            v = v / 1000
        return datetime.fromtimestamp(v).strftime('%Y%m%d')
    except (OSError, ValueError, TypeError):
        return None


def get_field(record, field_names, default=None):
    """
    从记录中获取字段值，支持多候选字段名。

    为什么需要这个函数？
      xtquant不同版本返回的财务数据字段名可能不同，
      通过提供多个候选名可以提高兼容性。

    Args:
        record: 财务数据记录字典
        field_names: 候选字段名列表，按优先级排列
        default: 所有字段都找不到时的默认值

    Returns:
        第一个非None的字段值，或default
    """
    for name in field_names:
        val = record.get(name)
        if val is not None:
            return val
    return default


def safe_float(val):
    """
    安全转换为浮点数，处理NaN和异常值。

    财务数据中经常出现None、NaN、字符串等异常值，
    统一转换为float或None。

    Args:
        val: 输入值

    Returns:
        float或None: 转换成功返回float，无效返回None
    """
    if val is None:
        return None
    try:
        v = float(val)
        # 检查是否为NaN（NaN不等于自身，这是Python的IEEE 754特性）
        return v if v == v else None
    except (ValueError, TypeError):
        return None


def safe_divide(a, b, pct=False):
    """
    安全除法，结果钳位到DECIMAL(10,4)范围（±999999.9999）。

    Args:
        a: 分子
        b: 分母
        pct: 是否将结果转为百分比（乘以100）

    Returns:
        float或None: 除零或无效输入返回None

    数值钳位说明：
      MySQL的DECIMAL(10,4)类型取值范围为 -999999.9999 ~ 999999.9999
      超出范围的数值会导致写入错误，因此需要钳位。
    """
    if a is None or b is None:
        return None
    a, b = float(a), float(b)
    if b == 0:
        return None
    result = a / b
    if pct:
        result *= 100
    result = round(result, 4)
    # 钳位，防止写入DECIMAL(10,4)溢出
    if result > 999999.9999:
        return 999999.9999
    if result < -999999.9999:
        return -999999.9999
    return result


def build_period_map(data_list):
    """
    将xtquant返回的财务数据转换为 {报告期: 记录} 映射。

    xtquant的财务数据可能以DataFrame或list of dict形式返回，
    此函数统一转换为以报告期为键的字典，方便按时期查找。

    Args:
        data_list: xtquant返回的财务数据（DataFrame或list）

    Returns:
        dict: {period_date: record_dict} 格式
    """
    period_map = {}
    if isinstance(data_list, pd.DataFrame):
        for _, row in data_list.iterrows():
            period_date = normalize_timetag(row.get('m_timetag'))
            if period_date:
                period_map[period_date] = row.to_dict()
    elif isinstance(data_list, list):
        for rec in data_list:
            if isinstance(rec, dict):
                period_date = normalize_timetag(rec.get('m_timetag'))
                if period_date:
                    period_map[period_date] = rec
    return period_map


def get_existing_stocks():
    """查询数据库中已有财务数据的股票集合，用于跳过已采集的股票"""
    rows = execute_query("SELECT DISTINCT stock_code FROM trade_stock_financial")
    return {r['stock_code'] for r in rows}


# ============================================================
# 核心逻辑
# ============================================================

def extract_periods(data, stock_code):
    """
    从xtquant财务数据中提取所有报告期的综合财务指标。

    这是本文件最核心的函数，组合多张财务报表的数据计算衍生指标：
      1. 从 PershareIndex 提取 EPS、ROE、毛利率
      2. 从 Balance 提取总资产、总负债、股东权益
      3. 从 Income 提取营业收入、净利润
      4. 从 CashFlow 提取经营性现金流
      5. 计算 ROA、净利率、资产负债率、流动比率

    Args:
        data: xtquant.get_financial_data() 的原始返回值
        stock_code: 股票代码

    Returns:
        list[dict]: 每个报告期的财务指标字典列表
    """
    stock_data = data.get(stock_code, {})
    if not stock_data:
        return []

    # 将各种财务表格数据转换为 {报告期: 记录} 映射
    pershare_map = build_period_map(stock_data.get('PershareIndex', []))
    balance_map = build_period_map(stock_data.get('Balance', []))
    income_map = build_period_map(stock_data.get('Income', []))
    cashflow_map = build_period_map(stock_data.get('CashFlow', []))

    # 取所有报告期的并集，确保覆盖各张报表的所有时期
    all_periods = sorted(set(
        list(pershare_map.keys()) +
        list(balance_map.keys()) +
        list(income_map.keys()) +
        list(cashflow_map.keys())
    ))

    records = []
    for period in all_periods:
        # 合并同一时期各张报表的数据
        ps = pershare_map.get(period, {})
        bal = balance_map.get(period, {})
        inc = income_map.get(period, {})
        cf = cashflow_map.get(period, {})

        # 基础指标
        # EPS(每股收益) = 净利润/总股本，衡量每股盈利能力
        eps = get_field(ps, ['s_fa_eps_basic'])
        # 营业收入 = 公司核心业务的总收入
        revenue = get_field(inc, ['revenue', 'operating_revenue'])
        # 净利润 = 扣除所有费用后的盈利
        net_profit = get_field(inc, ['net_profit_incl_min_int_inc', 'net_profit_excl_min_int_inc'])
        # 营业成本 = 与营收直接相关的成本
        operating_cost = get_field(inc, ['cost_of_goods_sold', 'total_operating_cost'])

        # ROE(净资产收益率) = 净利润/平均净资产
        # 这是巴菲特最看重的指标之一，反映公司为股东创造价值的能力
        roe = get_field(ps, ['du_return_on_equity', 'equity_roe', 'net_roe'])

        # 毛利率 = (营收-营业成本)/营收
        # 如果xtquant直接提供了毛利率则使用，否则手动计算
        gross_margin = get_field(ps, ['sales_gross_profit'])
        if gross_margin is None and revenue and operating_cost:
            r, c = float(revenue), float(operating_cost)
            if r > 0:
                gross_margin = round((r - c) / r * 100, 4)

        # 资产负债表数据
        total_assets = get_field(bal, ['tot_assets'])          # 总资产
        total_liab = get_field(bal, ['tot_liab'])               # 总负债
        total_equity = get_field(bal, ['total_equity', 'tot_shrhldr_eqy_incl_min_int'])  # 股东权益
        current_assets = get_field(bal, ['total_current_assets'])       # 流动资产
        current_liab = get_field(bal, ['total_current_liability'])      # 流动负债

        # 衍生指标计算
        # ROA(总资产回报率) = 净利润/总资产，衡量全部资产的盈利效率
        roa = safe_divide(net_profit, total_assets, pct=True)
        # 如果xtquant没有提供ROE，手动计算
        if roe is None and net_profit and total_equity:
            roe = safe_divide(net_profit, total_equity, pct=True)

        # 净利率(销售净利率) = 净利润/营业收入
        net_margin = safe_divide(net_profit, revenue, pct=True)
        # 资产负债率(杠杆率) = 总负债/总资产，通常40-60%较合理
        debt_ratio = safe_divide(total_liab, total_assets, pct=True)
        # 流动比率 = 流动资产/流动负债，>1表示短期偿债能力良好
        current_ratio = safe_divide(current_assets, current_liab)

        # 经营活动现金流净额（现金流量表中最重要的指标）
        # 反映公司日常经营产生的现金，持续为正才是健康的
        operating_cashflow = get_field(cf, ['net_cash_flows_oper_act'])

        records.append({
            'report_date': period,
            'revenue': safe_float(revenue),
            'net_profit': safe_float(net_profit),
            'eps': safe_float(eps),
            'roe': safe_float(roe),
            'roa': safe_float(roa),
            'gross_margin': safe_float(gross_margin),
            'net_margin': safe_float(net_margin),
            'debt_ratio': safe_float(debt_ratio),
            'current_ratio': safe_float(current_ratio),
            'operating_cashflow': safe_float(operating_cashflow),
            'total_assets': safe_float(total_assets),
            'total_equity': safe_float(total_equity),
        })

    return records


def process_batch(batch_codes):
    """
    批量下载 + 解析 + 写DB，返回 (写入总行数, 成功股票数)。

    这是性能优化的核心：一次下载50只股票的全部财务报表，
    然后批量解析、批量写入数据库。

    Args:
        batch_codes: 本批处理的股票代码列表

    Returns:
        tuple: (写入总行数, 成功处理的股票数)
    """
    # 批量下载到本地缓存（一次调用覆盖整批所有报表）
    done = [False]
    def on_done(data):
        """下载完成回调函数"""
        done[0] = True

    # download_financial_data2 是 xtquant 的异步批量下载接口
    # 它会将指定股票的所有财务表下载到本地缓存
    # 相比逐只下载，批量下载能大幅减少网络请求次数
    xtdata.download_financial_data2(
        stock_list=batch_codes,
        table_list=TABLE_LIST,
        start_time=DATA_START,
        end_time=DATA_END,
        callback=on_done
    )

    # 等待下载完成，最长120秒
    # download_financial_data2 是异步的，需要等待回调触发
    deadline = time.time() + 120
    while not done[0] and time.time() < deadline:
        time.sleep(0.5)
    # 下载完成后额外等待，确保缓存写入磁盘
    time.sleep(1)

    # 从本地缓存批量获取已下载的财务数据
    data = xtdata.get_financial_data(
        stock_list=batch_codes,
        table_list=TABLE_LIST,
        start_time=DATA_START,
        end_time=DATA_END,
        report_type='report_time'
    )

    if not data:
        return 0, 0

    # 逐只解析，统一写DB
    conn = get_connection()
    cursor = conn.cursor()
    batch_rows = 0
    batch_ok = 0

    for code in batch_codes:
        records = extract_periods(data, code)
        if records:
            rows = []
            for rec in records:
                p = rec['report_date']
                report_date = f"{p[:4]}-{p[4:6]}-{p[6:8]}"
                rows.append((
                    code, report_date,
                    rec['revenue'], rec['net_profit'], rec['eps'],
                    rec['roe'], rec['roa'], rec['gross_margin'], rec['net_margin'],
                    rec['debt_ratio'], rec['current_ratio'],
                    rec['operating_cashflow'], rec['total_assets'], rec['total_equity'],
                    'qmt'
                ))
            cursor.executemany(INSERT_SQL, rows)
            batch_rows += len(rows)
        batch_ok += 1

    conn.commit()
    cursor.close()
    conn.close()
    return batch_rows, batch_ok


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("财务数据采集 (MiniQMT -> MySQL)")
    if TEST_MODE:
        print("[测试模式] 只采集贵州茅台")
    else:
        print(f"[全量模式] 采集{SECTOR}, 每批{BATCH_SIZE}只")
    print("=" * 60)

    print("\n连接QMT数据服务...")
    xtdata.connect()
    print("  连接成功")

    # 获取股票列表
    if TEST_MODE:
        all_codes = [TEST_STOCK]
        print(f"\n[测试模式] 只采集 {TEST_STOCK}")
    else:
        print(f"\n获取 {SECTOR} 股票列表...")
        all_codes = xtdata.get_stock_list_in_sector(SECTOR)
        all_codes = [c for c in all_codes if '.' in str(c)]
        print(f"  共 {len(all_codes)} 只股票")

    # 查询已采集的股票，跳过已有数据的（断点续传）
    print("查询数据库已有数据...")
    existing = get_existing_stocks()
    pending = [c for c in all_codes if c not in existing]

    print(f"  已采集: {len(existing)} 只, 待采集: {len(pending)} 只")

    if not pending:
        print("\n全部已采集完成，无需下载")
        _print_summary()
        return

    # 分批处理：将待采集列表按 BATCH_SIZE 切分
    batches = [pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    total_batches = len(batches)
    total_pending = len(pending)

    print(f"\n开始批量下载（共 {total_batches} 批, 每批最多 {BATCH_SIZE} 只）...")

    total_rows = 0
    total_ok = 0
    total_done_stocks = 0
    start_time = time.time()

    for i, batch in enumerate(batches):
        # 先打印当前批次状态，让用户知道正在处理
        sys.stdout.write(
            f"\r  批次 {i + 1}/{total_batches} 下载中... "
            f"({total_done_stocks}/{total_pending})    "
        )
        sys.stdout.flush()

        batch_rows, batch_ok = process_batch(batch)
        total_rows += batch_rows
        total_ok += batch_ok
        total_done_stocks += len(batch)

        elapsed = time.time() - start_time
        speed = total_done_stocks / elapsed if elapsed > 0 else 0
        eta = (total_pending - total_done_stocks) / speed if speed > 0 else 0

        sys.stdout.write(
            f"\r  批次 {i + 1}/{total_batches} 完成 | "
            f"进度 {total_done_stocks}/{total_pending} ({total_done_stocks * 100 / total_pending:.1f}%) | "
            f"{speed:.1f} 只/秒 | 剩余约 {eta:.0f}秒 | "
            f"写入 {total_rows:,} 条    "
        )
        sys.stdout.flush()

    print()

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"财务数据采集完成! 耗时 {elapsed:.1f} 秒")
    print(f"  本次处理: {total_ok}/{total_pending} 只股票")
    print(f"  总写入: {total_rows:,} 条记录")

    _print_summary()


def _print_summary():
    """打印 trade_stock_financial 表的汇总统计"""
    summary = execute_query("""
        SELECT COUNT(DISTINCT stock_code) as stock_cnt,
               COUNT(*) as row_cnt,
               MIN(report_date) as min_date, MAX(report_date) as max_date
        FROM trade_stock_financial
    """)
    if summary:
        row = summary[0]
        print(f"\n数据库 trade_stock_financial 概况:")
        print(f"  {row['stock_cnt']} 只股票, {row['row_cnt']:,} 条记录")
        print(f"  日期范围: {row['min_date']} ~ {row['max_date']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
