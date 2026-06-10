# -*- coding: utf-8 -*-
"""
新闻事件采集 - AkShare -> MySQL

采集范围：全量A股（仅保留沪深京A股，过滤ETF/债券/基金）
数据源：AkShare stock_news_em() - 东方财富个股新闻
去重方式：内存去重（近期标题） + INSERT IGNORE（唯一键兜底）
跳过逻辑：当日已采集过的股票跳过（正确处理周末）

优化：
  - 批量缓冲：每50只股票的新闻汇总后 executemany 一次写入
  - 直接列访问替代 iterrows()
  - 仅加载近期标题去重，不限量全表加载
  - 线程仅负责拉取，主线程统一入库，无需锁

运行：python 4-新闻事件采集.py
"""
import sys
import os
import time
from datetime import date, timedelta, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import akshare as ak

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
NUM_WORKERS = 8                     # 并行采集线程数
FLUSH_STOCKS = 50                   # 每累积多少只股票的新闻就入库一次
NEWS_MAX_AGE = 3                    # 仅保留最近N天内发布的新闻
DB_RETENTION_DAYS = 7               # 超过N天的新闻从DB中删除

# 情感关键词
POSITIVE_WORDS = ['涨停', '大涨', '利好', '增长', '突破', '新高', '预增', '增持',
                  '盈利', '超预期', '重大突破', '战略合作', '中标']
NEGATIVE_WORDS = ['跌停', '大跌', '利空', '下降', '跌破', '新低', '预减', '减持',
                  '亏损', '违规', '处罚', '退市', '暴雷', '爆仓']
IMPORTANT_WORDS = ['资产重组', '业绩预增', '业绩预减', '高送转', '股权激励',
                   '定向增发', '股东减持', '股东增持', '重大合同', '中标',
                   '收购', '并购', '停牌', '复牌', '退市', '回购']

# A股代码前缀（与行情采集脚本保持一致）
_A_SH = set(list(range(600, 607)) + list(range(688, 691)))
_A_SZ = set(list(range(0, 5)) + list(range(300, 303)))
_A_BJ = set(range(800, 930))


def _is_a_stock(code):
    """判断是否为A股股票，排除ETF/债券/基金"""
    if not code or '.' not in code:
        return False
    num = code.split('.')[0]
    if len(num) != 6:
        return False
    try:
        p = int(num[:3])
    except ValueError:
        return False
    if code.endswith('.SH'):
        return p in _A_SH
    if code.endswith('.SZ'):
        return p in _A_SZ
    if code.endswith('.BJ'):
        return p in _A_BJ
    return False


def analyze_sentiment(title):
    """关键词情感分析：正面 > 负面 > 中性"""
    for word in POSITIVE_WORDS:
        if word in title:
            return 'positive'
    for word in NEGATIVE_WORDS:
        if word in title:
            return 'negative'
    return 'neutral'


def check_important(title):
    """判断是否重要事件新闻"""
    for word in IMPORTANT_WORDS:
        if word in title:
            return True
    return False


# ============================================================
# 数据库查询
# ============================================================

def _get_today_str():
    """返回今日日期字符串（周末回退到周五），避免 CURDATE() 的周末问题"""
    today = date.today()
    w = today.weekday()
    offset = w - 4 if w > 4 else 0
    return (today - timedelta(days=offset)).strftime('%Y-%m-%d')


def get_all_stocks():
    """从行情表获取A股列表（已过滤ETF/债券/基金）"""
    rows = execute_query("SELECT DISTINCT stock_code FROM trade_stock_daily")
    return [r['stock_code'] for r in rows if _is_a_stock(r['stock_code'])]


def get_today_collected():
    """获取今日已采集过新闻的股票（用推算的交易日，非CURDATE）"""
    today_str = _get_today_str()
    rows = execute_query(
        "SELECT DISTINCT stock_code FROM trade_stock_news WHERE DATE(created_at) = %s",
        (today_str,)
    )
    return {r['stock_code'] for r in rows}


def load_all_titles():
    """加载全部标题用于内存去重（由于仅保留近期数据，全量加载即可）"""
    rows = execute_query("SELECT title FROM trade_stock_news")
    return {r['title'] for r in rows}


# ============================================================
# 新闻采集
# ============================================================

INSERT_SQL = """
    INSERT IGNORE INTO trade_stock_news
    (stock_code, news_type, title, content, source, source_url,
     sentiment, is_important, published_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
"""


def fetch_news_akshare(stock_code):
    """
    通过 AkShare 采集个股新闻（来源：东方财富），仅保留最近 N 天内发布的。

    Returns:
        list[dict]: 新闻列表
    """
    code_num = stock_code.split('.')[0]
    try:
        df = ak.stock_news_em(symbol=code_num)
    except Exception:
        return []
    if df is None or len(df) == 0:
        return []

    try:
        titles = df['新闻标题'].values
        contents = df['新闻内容'].values
        urls = df['新闻链接'].values
        pub_times = df['发布时间'].values
        sources = df['文章来源'].values
    except Exception:
        return []

    cutoff = datetime.now() - timedelta(days=NEWS_MAX_AGE)

    news_list = []
    for i in range(len(df)):
        title = str(titles[i]).strip()
        if not title:
            continue
        pub_time_str = str(pub_times[i]).strip()
        # 过滤超过 NEWS_MAX_AGE 天的旧闻
        if pub_time_str:
            try:
                pub_dt = datetime.strptime(pub_time_str[:19], '%Y-%m-%d %H:%M:%S')
                if pub_dt < cutoff:
                    continue
            except ValueError:
                pass  # 解析失败不过滤，照常入库

        content = str(contents[i]).strip()[:2000]
        url = str(urls[i]).strip()
        source = str(sources[i]).strip() or 'eastmoney'

        news_list.append({
            'title': title,
            'content': content,
            'link': url,
            'published_at': pub_time_str if pub_time_str else None,
            'sentiment': analyze_sentiment(title),
            'is_important': check_important(title),
            'source': source,
            'news_type': 'news',
        })

    return news_list


def cleanup_old_news():
    """删除 DB_RETENTION_DAYS 天前的过期新闻"""
    deleted = execute_query(
        "SELECT COUNT(*) as cnt FROM trade_stock_news WHERE created_at < DATE_SUB(NOW(), INTERVAL %s DAY)",
        (DB_RETENTION_DAYS,)
    )
    count = deleted[0]['cnt'] if deleted else 0
    if count > 0:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM trade_stock_news WHERE created_at < DATE_SUB(NOW(), INTERVAL %s DAY)",
            (DB_RETENTION_DAYS,)
        )
        conn.commit()
        cursor.close()
        conn.close()
        print("清理过期新闻: {} 条 ({}天前)".format(count, DB_RETENTION_DAYS))
    return count


def _batch_save(news_list, existing_titles):
    """
    批量入库：内存去重 + executemany。

    Args:
        news_list: [(stock_code, news_dict), ...]
        existing_titles: 去重用标题集合（会被原地更新）

    Returns:
        int: 实际新增条数
    """
    if not news_list:
        return 0

    rows = []
    for stock_code, news in news_list:
        title = news['title']
        if title in existing_titles:
            continue
        existing_titles.add(title)
        rows.append((
            stock_code, news['news_type'], title,
            news['content'], news['source'], news['link'],
            news['sentiment'], 1 if news['is_important'] else 0,
            news['published_at']
        ))

    if not rows:
        return 0

    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.executemany(INSERT_SQL, rows)
        conn.commit()
        return len(rows)
    except Exception as e:
        conn.rollback()
        print(f"\n  批量入库失败: {e}")
        return 0
    finally:
        cursor.close()
        conn.close()


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("新闻事件采集（全量A股）")
    print("=" * 60)

    stock_list = get_all_stocks()
    print("全量A股: {} 只".format(len(stock_list)))

    # 跳过今日已采集过的股票（交易日推算，正确处理周末）
    collected = get_today_collected()
    if collected:
        stock_list = [c for c in stock_list if c not in collected]
        print("跳过今日已采集: {} 只, 待采集: {} 只".format(len(collected), len(stock_list)))

    if not stock_list:
        print("全部股票今日已采集，无需再跑")
        return

    # 加载全部标题用于去重（DB仅保留近期数据，全量加载即可）
    print("加载已有标题用于去重...")
    existing_titles = load_all_titles()
    print("  已有 {} 条标题".format(len(existing_titles)))

    total = len(stock_list)
    total_fetched = 0
    total_saved = 0
    done = 0
    start_time = time.time()

    # 缓冲：主线程收集 worker 结果，累积到阈值后批量入库
    buffer = []        # [(stock_code, news_dict), ...]
    buffer_stocks = 0

    print("\n开始采集 ({} 线程, 每{}只入库一次)...".format(NUM_WORKERS, FLUSH_STOCKS))

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(fetch_news_akshare, code): code for code in stock_list}

        for future in as_completed(futures):
            done += 1
            code = futures[future]
            try:
                news = future.result()
            except Exception:
                news = []

            total_fetched += len(news)
            for n in news:
                buffer.append((code, n))
            buffer_stocks += 1

            # 每 FLUSH_STOCKS 只股票入库一次
            if buffer_stocks >= FLUSH_STOCKS:
                saved = _batch_save(buffer, existing_titles)
                total_saved += saved
                elapsed = time.time() - start_time
                speed = done / elapsed if elapsed > 0 else 0
                print(f"\r    入库 {saved} 条 | 累计采集 {total_fetched} 新增 {total_saved} | "
                      f"{speed:.1f}只/秒    ")
                buffer.clear()
                buffer_stocks = 0

            # 每只股票都刷新进度
            elapsed = time.time() - start_time
            speed = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / speed if speed > 0 else 0
            sys.stdout.write(
                "\r  [{}/{}] {:.1f}% | {:.1f}只/秒 | ETA {:.0f}s | "
                "采集 {} 新增 {}    ".format(
                    done, total, done * 100 / total,
                    speed, eta, total_fetched, total_saved
                )
            )
            sys.stdout.flush()

    # 入库剩余
    if buffer:
        saved = _batch_save(buffer, existing_titles)
        total_saved += saved
        print(f"\n    最后入库 {saved} 条")

    print()

    elapsed = time.time() - start_time

    # 清理过期新闻
    print("\n清理过期新闻 ({}天前)...".format(DB_RETENTION_DAYS))
    cleanup_old_news()

    result = execute_query("SELECT COUNT(*) as cnt FROM trade_stock_news")
    total_db = result[0]['cnt'] if result else 0

    print("\n" + "=" * 60)
    print("新闻采集完成! 耗时 {:.1f} 秒".format(elapsed))
    print("  处理: {} 只股票".format(done))
    print("  采集: {} 条, 新增: {} 条".format(total_fetched, total_saved))
    print("  trade_stock_news 总计 {} 条".format(total_db))
    print("=" * 60)


if __name__ == '__main__':
    main()
