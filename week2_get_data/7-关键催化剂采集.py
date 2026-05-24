# -*- coding: utf-8 -*-
"""
关键催化剂事件采集 - 使用 Qwen Max 联网搜索

什么是"催化剂事件"？
  催化剂事件是指能够显著影响A股市场走势的宏观/政策事件。
  与普通财经日历事件不同，催化剂事件更侧重于"对A股有重大影响"，包括：
    - 两会、中央经济工作会议等重大政策会议
    - FOMC利率决议
    - 中美关系相关事件
    - 重要产业政策发布
    - 重大监管政策变化

工作流程：
  1. 调用 Qwen Max 联网搜索未来6个月的关键催化剂事件
  2. 提取结构化JSON数据（日期、标题、国家、类别、重要性）
  3. 为每个事件生成AI提问prompt（用于后续深度分析）
  4. 写入 trade_calendar_event 表（importance >= 2）

数据流向：
  本脚本采集的事件 -> trade_calendar_event 表
  -> 后续AI分析脚本读取事件 + AI prompt -> 生成投资影响分析

运行：python 7-关键催化剂采集.py
"""
import sys
import os
import json
import yaml
from datetime import datetime, timedelta
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db_config import get_connection, DASHSCOPE_API_KEY, DASHSCOPE_BASE_URL

if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# prompts.yaml 文件路径，包含发送给LLM的提示词模板
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), 'prompts.yaml')


def load_prompts_config():
    """
    加载 prompts.yaml 配置文件。

    将提示词模板从代码中分离到YAML文件的好处：
      1. 可以独立修改提示词而不用改代码
      2. 不同类型任务的提示词可以统一管理
      3. 支持团队成员协作编辑提示词
    """
    with open(PROMPTS_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _call_qwen(client, prompt, enable_search=False):
    """
    通用 Qwen Max 调用函数。

    Args:
        client: OpenAI兼容客户端实例
        prompt: 发送给模型的提示词
        enable_search: 是否启用联网搜索功能

    Returns:
        str: 模型生成的文本响应

    联网搜索说明：
      当 enable_search=True 时，Qwen Max 会联网检索最新信息，
      这对于获取未来事件的准确日期至关重要（如FOMC会议日期、
      经济数据公布日等）。不启用搜索时模型只能依赖训练数据中的知识。
    """
    extra = {}
    if enable_search:
        extra = {"enable_search": True, "search_options": {"forced_search": True}}
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=[{"role": "user", "content": prompt}],
        extra_body=extra,
    )
    return completion.choices[0].message.content.strip()


def _parse_json_array(content):
    """
    从 LLM 输出中提取 JSON 数组。

    为什么需要专门解析？
      LLM可能会在JSON前后添加解释性文字，如：
      "根据搜索结果，以下是未来6个月的关键事件：
      [{"date": "2026-06-15", ...}]"
      需要从中提取出JSON数组部分。

    Args:
        content: LLM返回的文本

    Returns:
        list: 解析后的JSON数组，解析失败返回空列表
    """
    start = content.find('[')
    end = content.rfind(']')
    if start == -1 or end == -1:
        return []
    return json.loads(content[start:end + 1])


def _get_client():
    """
    创建 OpenAI 兼容客户端。

    使用 DashScope(阿里云)的API，与OpenAI SDK完全兼容，
    因此可以直接复用 openai Python 包。
    """
    return OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )


def search_catalysts():
    """
    调用 Qwen Max 联网搜索关键催化剂事件。

    搜索策略：
      1. 从 prompts.yaml 加载搜索提示词模板
      2. 将当前日期和未来6个月的日期范围填入模板
      3. 启用联网搜索获取最新、最准确的事件信息
      4. 解析返回的JSON数据

    Qwen Max联网搜索的能力：
      - 可以获取未来已确定的事件日期（FOMC会议、经济数据公布等）
      - 可以根据新闻推断可能发生的事件（政策预期等）
      - 可以对事件的重要性进行排序和分类

    Returns:
        list[dict]: 事件列表，每个事件包含 date, title, country, category, importance
    """
    cfg = load_prompts_config()
    prompt_tpl = cfg['calendar']['search_catalysts']
    today = datetime.now().date()
    start_date = today.isoformat()
    end_date = (today + timedelta(days=180)).isoformat()

    prompt = prompt_tpl.format(start_date=start_date, end_date=end_date)

    print(f"搜索范围: {start_date} ~ {end_date}")
    print("调用 Qwen Max 联网搜索...")

    client = _get_client()
    content = _call_qwen(client, prompt, enable_search=True)
    print(f"  原始响应长度: {len(content)} 字符")

    events = _parse_json_array(content)
    if not events:
        print("  未找到 JSON 数组，原始内容:")
        print(content[:500])
        return []

    print(f"  解析到 {len(events)} 个事件")
    return events


def generate_prompts(events):
    """
    为催化剂事件批量生成 AI 提问 prompt。

    作用：
      对每个事件生成一个专门的提问prompt，后续可以用于：
        - 让AI分析该事件对A股各板块的影响
        - 评估事件的超预期概率
        - 生成投资建议

    分批处理：
      为了防止LLM单次处理太多事件导致输出质量下降，
      每批最多20个事件。同时只传递标题和日期减少token消耗。

    Args:
        events: 事件列表

    Returns:
        dict: {event_title: ai_prompt_string} 的映射字典
    """
    cfg = load_prompts_config()
    prompt_tpl = cfg['calendar']['generate_prompts']

    # 只传标题和日期给LLM，减少token消耗
    events_brief = [{"date": e.get("date", ""), "title": e.get("title", ""), "country": e.get("country", "")} for e in events]

    # 每批最多20个事件
    all_prompts = {}
    client = _get_client()
    for i in range(0, len(events_brief), 20):
        batch = events_brief[i:i+20]
        prompt = prompt_tpl.format(events_json=json.dumps(batch, ensure_ascii=False, indent=2))
        print(f"  生成prompt第 {i//20+1} 批 ({len(batch)} 个事件)...")
        content = _call_qwen(client, prompt)
        results = _parse_json_array(content)
        for r in results:
            title = r.get('title', '').strip()
            ai_prompt = r.get('prompt', '').strip()
            if title and ai_prompt:
                all_prompts[title] = ai_prompt

    print(f"  共生成 {len(all_prompts)} 个prompt")
    return all_prompts


def save_events(events, prompts_map=None):
    """
    将催化剂事件写入数据库（先去重再写入）。

    去重逻辑：
      因为本脚本可能多次运行，已经写入的事件不应该重复写入。
      去重基于两个条件：
        1. source='qwen_search' 来源的事件
        2. 标题归一化后匹配 + 日期相差5天内

    Args:
        events: 事件列表
        prompts_map: {title: prompt} 映射，可选

    Returns:
        int: 新增写入的事件数
    """
    conn = get_connection()
    cursor = conn.cursor()

    # 先拿到已有的 qwen_search 事件，用于去重
    cursor.execute("""
        SELECT id, event_date, title FROM trade_calendar_event
        WHERE source = 'qwen_search'
    """)
    existing = cursor.fetchall()
    # 建立归一化标题 -> 已有事件的映射
    # 归一化方式：去除斜杠和空格
    existing_map = {}
    for row in existing:
        key = row[2].replace('/', '').replace(' ', '')
        existing_map.setdefault(key, []).append(row)

    if prompts_map is None:
        prompts_map = {}

    sql = """
        INSERT INTO trade_calendar_event
        (event_date, event_time, title, country, category,
         importance, source, ai_prompt)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
        importance = GREATEST(importance, VALUES(importance)),
        category = VALUES(category),
        source = VALUES(source),
        ai_prompt = COALESCE(VALUES(ai_prompt), ai_prompt)
    """

    count = 0
    for evt in events:
        date_str = evt.get('date', '')
        title = evt.get('title', '').strip()
        if not date_str or not title:
            continue

        try:
            evt_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            print(f"  跳过无效日期: {date_str} - {title}")
            continue

        # 检查是否已有类似事件（标题相同、日期相差5天内）
        norm_title = title.replace('/', '').replace(' ', '')
        skip = False
        for ex in existing_map.get(norm_title, []):
            diff = abs((evt_date - ex[1]).days)
            if diff <= 5 and diff > 0:
                skip = True
                break
        if skip:
            continue

        country = evt.get('country', '中国')
        category = evt.get('category', 'policy')
        # 重要性钳位到2~3之间，最低2星（重要），最高3星（非常重要）
        importance = max(2, min(3, evt.get('importance', 2)))

        ai_prompt = prompts_map.get(title)

        cursor.execute(sql, (
            date_str, None, title, country, category,
            importance, 'qwen_search', ai_prompt
        ))
        count += 1

    conn.commit()
    cursor.close()
    conn.close()
    return count


def main():
    print("=" * 60)
    print("关键催化剂事件采集 (Qwen Max 联网搜索)")
    print("=" * 60)

    events = search_catalysts()
    if not events:
        print("\n未获取到事件")
        return

    # 打印事件预览
    print(f"\n事件预览 ({len(events)} 个):")
    for evt in events:
        imp = evt.get('importance', 2)
        stars = '*' * imp
        print(f"  [{stars}] {evt.get('date', '?')} {evt.get('country', '?')} {evt.get('title', '?')}")

    # 为每个事件生成AI提问prompt
    print("\n生成事件提问prompt...")
    prompts_map = generate_prompts(events)

    count = save_events(events, prompts_map)
    print(f"\n写入/更新 {count} 条催化剂事件")

    # 统计
    from db_config import execute_query
    rows = execute_query("""
        SELECT importance, COUNT(*) as cnt
        FROM trade_calendar_event
        WHERE source = 'qwen_search'
        GROUP BY importance ORDER BY importance DESC
    """)
    print("\nqwen_search 来源统计:")
    for r in rows:
        print(f"  {r['importance']}星: {r['cnt']} 条")

    print("\n" + "=" * 60)
    print("催化剂采集完成!")
    print("=" * 60)


def _normalize_title(t):
    """
    标题归一化，用于模糊匹配。

    去除标题中的空格、斜杠、破折号、括号等干扰字符，
    使标题比较更加鲁棒。

    Args:
        t: 原始标题

    Returns:
        str: 归一化后的标题
    """
    import re
    return re.sub(r'[\s/\-—()（）]', '', t)


def backfill_prompts():
    """
    为已有事件补充缺失的 ai_prompt。

    使用场景：
      如果初次运行脚本时prompt生成失败（如API超时），
      已有事件可能缺少 ai_prompt 字段。
      通过 --backfill-prompts 参数运行本脚本可以补充这些缺失的prompt。

    使用方式：
      python 7-关键催化剂采集.py --backfill-prompts
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, event_date, title, country FROM trade_calendar_event
        WHERE source = 'qwen_search' AND (ai_prompt IS NULL OR ai_prompt = '')
          AND event_date >= CURDATE()
    """)
    rows = cursor.fetchall()
    if not rows:
        print("所有事件已有prompt，无需补充")
        cursor.close()
        conn.close()
        return

    print(f"需要补充prompt的事件: {len(rows)} 个")
    events = [{"date": str(r[1]), "title": r[2], "country": r[3]} for r in rows]

    # 建立归一化标题 -> id 的映射，支持模糊匹配
    norm_id_map = {}
    for r in rows:
        norm_id_map[_normalize_title(r[2])] = r[0]

    prompts_map = generate_prompts(events)

    updated = 0
    for title, prompt_text in prompts_map.items():
        norm = _normalize_title(title)
        eid = norm_id_map.get(norm)
        if eid:
            cursor.execute("UPDATE trade_calendar_event SET ai_prompt = %s WHERE id = %s", (prompt_text, eid))
            updated += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"已补充 {updated} 个事件的prompt")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--backfill-prompts', action='store_true', help='仅为缺失prompt的事件补充')
    args = parser.parse_args()
    if args.backfill_prompts:
        backfill_prompts()
    else:
        main()
