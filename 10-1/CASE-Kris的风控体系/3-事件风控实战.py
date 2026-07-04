# -*- coding: utf-8 -*-
"""
CASE: 事件风控实战

核心命题: 数字看不到的风险, 让 Kris 用 "读新闻" 的方式补上

技术面 (金额/价格/ATR/熔断) 都是基于数字的检查, 但有些风险藏在文字里:
    - 公司被立案调查 -- 财务报表上看不到, 但新闻标题第一时间出现
    - 大股东减持      -- 基本面正常, 但市场情绪会迅速反应
    - 财务造假        -- 报表本身就是假的, 数字检查无效

本脚本演示三件事 (全部基于 akshare 真实新闻):
    [Part 1] akshare 拉指定标的最近 N 条真实新闻
    [Part 2] 关键词 vs 大模型 双模式逐条对比
    [Part 3] 嵌入 Kris 主审批 (关键词模式 + LLM 模式)

调用:
    python 3-事件风控实战.py                  # 默认 600519 茅台 + 10 条新闻
    python 3-事件风控实战.py 002594           # 比亚迪
    python 3-事件风控实战.py 600519 20        # 茅台 + 20 条

依赖:
    pip install akshare dashscope python-dotenv
    在 .env 或系统环境变量中设置 DASHSCOPE_API_KEY
"""
import os
import sys
import time
from typing import List, Dict
from dotenv import load_dotenv

import akshare as ak
import pandas as pd

from importlib import import_module
risk_engine = import_module("1-风控引擎")
RiskManager = risk_engine.RiskManager
EventKeywordChecker = risk_engine.EventKeywordChecker
EventLLMChecker = risk_engine.EventLLMChecker
Order = risk_engine.Order
Decision = risk_engine.Decision

load_dotenv()


# ============================================================
# Part 1: akshare 真实新闻拉取
# ============================================================

def part1_fetch_real_news(stock_code: str, top_n: int = 10) -> List[Dict]:
    """[Part 1] 用 akshare 抓个股最近的真实新闻"""
    print("\n" + "=" * 100)
    print(f"  [Part 1] 真实新闻拉取 -- ak.stock_news_em(symbol='{stock_code}')")
    print("=" * 100)

    code = stock_code.split('.')[0] if '.' in stock_code else stock_code
    df = ak.stock_news_em(symbol=code)
    if df is None or len(df) == 0:
        return []

    df = df.head(top_n)
    news_list = []
    for _, row in df.iterrows():
        news_list.append({
            'title': str(row.get('新闻标题', '')),
            'content': str(row.get('新闻内容', ''))[:500],
            'time': str(row.get('发布时间', '')),
            'source': str(row.get('文章来源', '')),
        })
    print(f"\n  共抓到 {len(news_list)} 条最近新闻")
    return news_list


# ============================================================
# Part 2: 关键词 vs LLM 双模式逐条对比
# ============================================================

def part2_compare_modes(stock_code: str, news_list: List[Dict]) -> pd.DataFrame:
    """[Part 2] 把每条新闻分别喂给两种模式, 并排对比"""
    print("\n" + "=" * 100)
    print(f"  [Part 2] 关键词 vs LLM 双模式逐条对比 -- {stock_code}")
    print("=" * 100)
    print("    [关键词]  EventKeywordChecker -- 10 个关键词硬匹配, 0 成本秒级")
    print("    [LLM   ]  EventLLMChecker     -- 通义千问 qwen-turbo, 理解上下文")

    kw = EventKeywordChecker()
    llm = EventLLMChecker(model='qwen-turbo')

    rows = []
    for i, news in enumerate(news_list, 1):
        title = news['title']
        full_text = title + " " + news['content']

        d_kw = kw.check(stock_code, full_text)
        d_llm = llm.check(stock_code, full_text)
        time.sleep(0.3)  # 限流, 防打爆 API

        llm_reason = d_llm.reason.replace(f"{stock_code} LLM判定: ", "")

        rows.append({
            '序号': i,
            '时间': news['time'][:10],
            '标题': title[:40],
            '关键词': d_kw.decision.value,
            'LLM': d_llm.decision.value,
            'LLM理由': llm_reason[:40],
        })

        tag_kw = _icon(d_kw.decision.value)
        tag_llm = _icon(d_llm.decision.value)
        print(f"\n  [{i:>2}] {news['time'][:10]} | {title[:60]}")
        print(f"       关键词 {tag_kw} {d_kw.decision.value:<8} | "
              f"LLM {tag_llm} {d_llm.decision.value:<8} -- {llm_reason[:60]}")

    df = pd.DataFrame(rows)
    print("\n  --- 一致性统计 ---")
    consistent = (df['关键词'] == df['LLM']).sum()
    print(f"    一致条数:     {consistent}/{len(df)}  ({consistent/len(df):.0%})")
    print(f"    关键词 reject: {(df['关键词']=='reject').sum()}")
    print(f"    LLM    reject: {(df['LLM']=='reject').sum()}")
    print(f"    LLM    warn:   {(df['LLM']=='warn').sum()}")
    print(f"  --> LLM 多识别出来的中等风险信号 = "
          f"{((df['关键词']=='approve') & (df['LLM'].isin(['warn','reject']))).sum()} 条")

    return df


def _icon(verdict: str) -> str:
    return {
        'approve': '[PASS]', 'warn': '[WARN]',
        'reject':  '[REJ ]', 'halt': '[HALT]',
    }.get(verdict, '[??? ]')


# ============================================================
# Part 3: 嵌入 Kris 主审批
# ============================================================

def part3_kris_audit(stock_code: str, news_list: List[Dict]):
    """[Part 3] 把所有新闻拼成长文本, 喂给 Kris 主审批 (买入 10 万)"""
    print("\n" + "=" * 100)
    print(f"  [Part 3] 嵌入 Kris 主审批 -- {stock_code} 买入 100,000 元")
    print("=" * 100)

    full_text = " || ".join(
        n['title'] + ' ' + n['content'][:200] for n in news_list
    )

    portfolio = {
        'total_asset': 1_000_000,
        'prices': {stock_code: 100.0},
        'atr': {stock_code: 2.0},
    }
    order = Order(stock_code, 'buy', 100_000, 100.0)

    # 模式 A: 关键词
    print(f"\n  >>> 模式 A: 关键词审批")
    kris_kw = RiskManager()
    kris_kw.start_day(1_000_000)
    kris_kw.macro.update_vix(18.0)
    d_kw = kris_kw.approve(order, portfolio, {'news_text': full_text})
    print(f"      {d_kw}")

    # 模式 B: LLM
    print(f"\n  >>> 模式 B: 大模型审批 (qwen-turbo)")
    kris_llm = RiskManager(event_checker=EventLLMChecker(model='qwen-turbo'))
    kris_llm.start_day(1_000_000)
    kris_llm.macro.update_vix(18.0)
    d_llm = kris_llm.approve(order, portfolio, {'news_text': full_text})
    print(f"      {d_llm}")


# ============================================================
# 主程序
# ============================================================

def main():
    args = sys.argv[1:]
    stock_code = args[0] if len(args) > 0 else '600519'
    top_n = int(args[1]) if len(args) > 1 else 10

    print("=" * 100)
    print(f"  CASE: 事件风控实战")
    print(f"  数字看不到的风险, 让 Kris 用 '读新闻' 的方式补上")
    print(f"  股票代码: {stock_code}    新闻条数: {top_n}")
    print("=" * 100)

    if not os.getenv('DASHSCOPE_API_KEY'):
        raise RuntimeError(
            "未设置 DASHSCOPE_API_KEY 环境变量。\n"
            "请在 .env 中添加 DASHSCOPE_API_KEY=sk-xxx, 或在系统环境变量中设置。"
        )

    news_list = part1_fetch_real_news(stock_code, top_n=top_n)
    if not news_list:
        print(f"[警告] {stock_code} 没有抓到新闻")
        return

    part2_compare_modes(stock_code, news_list)
    part3_kris_audit(stock_code, news_list)


if __name__ == '__main__':
    main()
