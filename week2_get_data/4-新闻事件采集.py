# -*- coding: utf-8 -*-
"""
新闻事件采集 - AkShare -> MySQL

采集范围：全量A股（trade_stock_daily 中所有股票）
数据源：AkShare stock_news_em() - 东方财富个股新闻
去重方式：按标题去重（批量预加载已有标题到内存，避免逐条查库）
跳过逻辑：当日已采集过的股票跳过

情感分析逻辑：
  使用关键词匹配进行简单的情感分类，分为正面(positive)、负面(negative)、中性(neutral)三类。
  注意：这种方法较为粗糙，但处理速度快，适合大规模数据。
  如果需要更高精度，后续可以接入NLP模型。

运行：python 4-新闻事件采集.py
"""
import sys
import os
import time
import pymysql
import akshare as ak
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import get_connection, execute_query

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 配置
# ============================================================
# 并行采集线程数
NUM_WORKERS = 8

# 正面关键词：当标题中出现这些词时标记为正面新闻
POSITIVE_WORDS = ['涨停', '大涨', '利好', '增长', '突破', '新高', '预增', '增持',
                  '盈利', '超预期', '重大突破', '战略合作', '中标']

# 负面关键词：当标题中出现这些词时标记为负面新闻
NEGATIVE_WORDS = ['跌停', '大跌', '利空', '下降', '跌破', '新低', '预减', '减持',
                  '亏损', '违规', '处罚', '退市', '暴雷', '爆仓']

# 重要事件关键词：涉及公司重大变化的新闻，需要特别关注
IMPORTANT_WORDS = ['资产重组', '业绩预增', '业绩预减', '高送转', '股权激励',
                   '定向增发', '股东减持', '股东增持', '重大合同', '中标',
                   '收购', '并购', '停牌', '复牌', '退市', '回购']

# 线程安全的打印锁，防止多线程同时输出导致内容混乱
_print_lock = threading.Lock()


def safe_print(msg):
    """线程安全的打印函数，保证多线程环境下输出不交错"""
    with _print_lock:
        print(msg)


def analyze_sentiment(title):
    """
    基于关键词的简单情感分析。

    匹配优先级：正面 > 负面 > 中性
    注意：标题中既包含正面词又包含负面词时，正面优先。
    这是因为负面词（如"减持"）可能出现在"股东增持"这样的正面语境中。

    Args:
        title: 新闻标题字符串

    Returns:
        str: 'positive', 'negative', 或 'neutral'
    """
    for word in POSITIVE_WORDS:
        if word in title:
            return 'positive'
    for word in NEGATIVE_WORDS:
        if word in title:
            return 'negative'
    return 'neutral'


def check_important(title):
    """
    判断新闻是否涉及公司重大变化。

    Args:
        title: 新闻标题字符串

    Returns:
        bool: True表示该新闻是重要事件
    """
    for word in IMPORTANT_WORDS:
        if word in title:
            return True
    return False


# ============================================================
# 获取需要采集的股票列表
# ============================================================

def get_all_stocks():
    """从行情数据表获取全量股票列表"""
    rows = execute_query("SELECT DISTINCT stock_code FROM trade_stock_daily")
    return [r['stock_code'] for r in rows]


def get_today_collected():
    """
    获取当日已采集过新闻的股票。

    目的是实现"断点续传"：如果某股票今日已经采集过新闻，
    则跳过它，避免重复采集。这在脚本中断重跑时非常有用。
    """
    rows = execute_query("""
        SELECT DISTINCT stock_code FROM trade_stock_news
        WHERE DATE(created_at) = CURDATE()
    """)
    return {r['stock_code'] for r in rows}


def load_existing_titles():
    """
    一次性加载所有已有新闻标题，用于内存去重。

    核心优化思路：
      如果不做预加载，每写入一条新闻前都要查一次数据库判断是否重复，
      这将产生O(n)次数据库查询。
      预加载到内存集合后，去重检查变为O(1)的集合查找操作，
      大幅减少数据库压力。

    Returns:
        set: 所有已有新闻标题的集合
    """
    rows = execute_query("SELECT title FROM trade_stock_news")
    return {r['title'] for r in rows}


# ============================================================
# 新闻采集
# ============================================================

def fetch_news_akshare(stock_code):
    """
    通过 AkShare 采集个股新闻（来源：东方财富）。

    Argstime:
        stock_code: 股票代码，如 "600519.SH"

    Returns:
        list[dict]: 新闻列表，每条包含标题、内容、链接、发布时间等
    """
    code_num = stock_code.split('.')[0]
    news_list = []
    try:
        # stock_news_em 从东方财富获取个股新闻
        df = ak.stock_news_em(symbol=code_num)
    except Exception:
        return news_list
    if df is None or len(df) == 0:
        return news_list

    for _, row in df.iterrows():
        title = str(row.get('新闻标题', '')).strip()
        if not title:
            continue
        content = str(row.get('新闻内容', '')).strip()
        url = str(row.get('新闻链接', '')).strip()
        pub_time = str(row.get('发布时间', '')).strip()
        source = str(row.get('文章来源', '')).strip()

        news_list.append({
            'title': title,
            'content': content[:2000] if content else '',  # 限制内容长度
            'link': url,
            'published_at': pub_time if pub_time else None,
            'sentiment': analyze_sentiment(title),
            'is_important': check_important(title),
            'source': source or 'eastmoney',
            'news_type': 'news',
        })

    return news_list


# ============================================================
# 写入数据库
# ============================================================

# 全局标题集合（用于内存去重）
# Python的 GIL(全局解释器锁) 保证了 dict/set 的单个操作是线程安全的
# 因此多个线程同时对 _existing_titles 进行 in 检查和 add 不会发生数据竞争
_existing_titles = set()


def save_news_to_db(stock_code, news_list):
    """
    新闻去重后写入MySQL。

    去重原理：
      1. 先在内存集合中检查标题是否存在（O(1)操作）
      2. 对不在集合中的新闻执行INSERT
      3. 如果INSERT遇到唯一键冲突(IntegrityError)，说明其他线程已写入
      4. 无论成功还是冲突，都将标题加入内存集合

    Args:
        stock_code: 股票代码
        news_list: 新闻列表

    Returns:
        int: 实际新增的新闻数量
    """
    global _existing_titles
    if not news_list:
        return 0

    # 内存去重：只保留标题在集合中不存在的新闻
    new_items = [n for n in news_list if n['title'] not in _existing_titles]
    if not new_items:
        return 0

    conn = get_connection()
    cursor = conn.cursor()
    saved = 0

    for news in new_items:
        try:
            cursor.execute("""
                INSERT INTO trade_stock_news
                (stock_code, news_type, title, content, source, source_url,
                 sentiment, is_important, published_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                stock_code, news['news_type'], news['title'],
                news['content'], news['source'], news['link'],
                news['sentiment'], 1 if news['is_important'] else 0,
                news['published_at']
            ))
            _existing_titles.add(news['title'])
            saved += 1
        except pymysql.err.IntegrityError:
            # 唯一键冲突（并发场景下其他线程已插入该新闻）
            # 将其标题加入内存集合，避免后续重复尝试
            _existing_titles.add(news['title'])

    conn.commit()
    cursor.close()
    conn.close()
    return saved


# ============================================================
# 单只股票处理
# ============================================================

def process_one_stock(stock_code):
    """
    采集单只股票新闻的完整流程。

    Args:
        stock_code: 股票代码

    Returns:
        tuple: (stock_code, 采集总数, 新增数量)
    """
    news = fetch_news_akshare(stock_code)
    saved = save_news_to_db(stock_code, news)
    return stock_code, len(news), saved


# ============================================================
# 主流程
# ============================================================

def main():
    global _existing_titles

    print("=" * 60)
    print("新闻事件采集（全量A股）")
    print("=" * 60)

    stock_list = get_all_stocks()
    print("全量股票: {} 只".format(len(stock_list)))

    # 跳过当日已采集过的股票
    collected = get_today_collected()
    if collected:
        stock_list = [c for c in stock_list if c not in collected]
        print("跳过当日已采集: {} 只, 待采集: {} 只".format(len(collected), len(stock_list)))

    if not stock_list:
        print("全部股票当日已采集，无需再跑")
        return

    # 预加载已有标题到内存（一次查询，后续不再逐条查库）
    # 这是核心性能优化：将所有已有标题加载到Python的set中
    print("加载已有标题用于去重...")
    _existing_titles = load_existing_titles()
    print("  已有 {} 条标题".format(len(_existing_titles)))

    total = len(stock_list)
    total_fetched = 0
    total_saved = 0
    done = 0
    start_time = time.time()

    print("\n开始采集 ({} 线程)...".format(NUM_WORKERS))

    # 使用线程池并行采集
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_one_stock, code): code
            for code in stock_list
        }

        for future in as_completed(futures):
            done += 1
            code, fetched, saved = None, 0, 0
            try:
                code, fetched, saved = future.result()
                total_fetched += fetched
                total_saved += saved
            except Exception:
                pass

            # 每200只或最后一只时打印进度
            if done % 200 == 0 or done == total:
                elapsed = time.time() - start_time
                speed = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / speed if speed > 0 else 0
                sys.stdout.write(
                    "\r  [{}/{}] {:.1f}% | {:.1f}/s | ETA {:.0f}s | "
                    "fetched {} saved {}    ".format(
                        done, total, done * 100 / total,
                        speed, eta, total_fetched, total_saved
                    )
                )
                sys.stdout.flush()

    print()

    elapsed = time.time() - start_time
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
