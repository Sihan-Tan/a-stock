#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
研报生成器

功能：将五步法分析结果组装为 Markdown 或 HTML 格式深度研报。
包含封面信息、目录、五步分析内容、风险提示和免责声明。
HTML 格式自带样式，浏览器打开即可阅读。

用法：
    python report_generator.py --analysis_file <分析结果JSON> --output_dir <输出目录>
    python report_generator.py --analysis_file output/贵州茅台_analysis.json --output_dir output/reports/ --format html
    python report_generator.py --content "# 标题\n正文内容" --title "研报标题" --output_dir reports/ --format html
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime


# 研报 Markdown 模板
REPORT_TEMPLATE = """# {stock_name} - 深度分析报告

> **分析框架**: 国泰君安"五步法"  
> **生成时间**: {analysis_date}  
> **分析模型**: {model}  
> **数据来源**: 公司财报 (来源页码: {source_pages})

---

## 目录

1. [信息差分析](#1-信息差分析)
2. [逻辑差分析](#2-逻辑差分析)
3. [预期差分析](#3-预期差分析)
4. [催化剂识别](#4-催化剂识别)
5. [投资结论](#5-投资结论)
6. [风险提示](#6-风险提示)
7. [免责声明](#7-免责声明)

---

{steps_content}

---

## 6. 风险提示

- 本报告基于公开财报数据和 AI 分析生成，分析结论可能存在偏差
- 财报数据具有滞后性，不代表公司当前经营状况
- 市场环境变化可能导致分析假设不再成立
- 行业政策调整可能对公司经营产生重大影响
- AI 模型的分析能力有限，无法完全替代专业分析师的判断

---

## 7. 免责声明

本报告由 AI 投研助手 Charles 自动生成，仅供学习和研究参考，**不构成任何投资建议**。

- 报告内容基于公开信息和 AI 分析，不保证信息的准确性和完整性
- 投资者据此操作，风险自担
- 报告作者不对任何投资损失承担责任
- 在做出投资决策前，建议咨询专业投资顾问

---

*报告生成于 {generation_time}*
"""

# 每个步骤的章节模板
STEP_SECTION_TEMPLATE = """## {step_num}. {step_title}

> 数据来源页码: {source_pages}

{analysis}

"""

# 步骤名称到章节标题的映射
STEP_TITLES = {
    "信息差": "信息差分析",
    "逻辑差": "逻辑差分析",
    "预期差": "预期差分析",
    "催化剂": "催化剂识别",
    "结论": "投资结论",
}


def load_analysis(analysis_file: str) -> dict:
    """加载五步法分析结果"""
    with open(analysis_file, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_report(analysis: dict) -> str:
    """
    将分析结果组装为 Markdown 研报。

    Args:
        analysis: 五步法分析结果（来自 five_step_analysis.py 的输出）

    Returns:
        完整的 Markdown 格式研报
    """
    stock_name = analysis["stock_name"]
    analysis_date = analysis["analysis_date"]
    model = analysis["model"]
    all_pages = analysis.get("all_source_pages", [])

    # 生成各步骤章节
    steps_content = ""
    for step_data in analysis["steps"]:
        step_num = step_data["step"]
        step_name = step_data["name"]
        step_title = STEP_TITLES.get(step_name, step_name)
        step_analysis = step_data["analysis"]
        step_pages = step_data.get("source_pages", [])

        section = STEP_SECTION_TEMPLATE.format(
            step_num=step_num,
            step_title=step_title,
            source_pages=", ".join(str(p) for p in step_pages) if step_pages else "未标注",
            analysis=step_analysis,
        )
        steps_content += section

    # 组装完整报告
    report = REPORT_TEMPLATE.format(
        stock_name=stock_name,
        analysis_date=analysis_date,
        model=model,
        source_pages=", ".join(str(p) for p in all_pages) if all_pages else "未标注",
        steps_content=steps_content,
        generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    return report


def save_report(report: str, stock_name: str, output_dir: str) -> str:
    """
    保存研报到 Markdown 文件。

    Returns:
        报告文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    filename = f"{stock_name}_深度研报_{date_str}.md"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    return filepath


# ============================================================
# HTML 格式支持
# ============================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary: #1a56db;
            --primary-light: #e8effc;
            --text: #1f2937;
            --text-secondary: #6b7280;
            --border: #e5e7eb;
            --bg: #ffffff;
            --bg-alt: #f9fafb;
            --success: #059669;
            --warning: #d97706;
            --danger: #dc2626;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                         "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
            color: var(--text);
            background: var(--bg-alt);
            line-height: 1.8;
        }}
        .container {{
            max-width: 900px;
            margin: 40px auto;
            background: var(--bg);
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            padding: 60px 80px;
        }}
        h1 {{
            font-size: 28px;
            color: var(--primary);
            border-bottom: 3px solid var(--primary);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        h2 {{
            font-size: 22px;
            color: var(--text);
            margin-top: 40px;
            margin-bottom: 16px;
            padding-left: 12px;
            border-left: 4px solid var(--primary);
        }}
        h3 {{
            font-size: 18px;
            color: var(--text);
            margin-top: 28px;
            margin-bottom: 12px;
        }}
        p {{ margin-bottom: 12px; }}
        blockquote {{
            background: var(--primary-light);
            border-left: 4px solid var(--primary);
            padding: 12px 20px;
            margin: 16px 0;
            border-radius: 0 8px 8px 0;
            color: var(--text-secondary);
        }}
        ul, ol {{
            margin: 12px 0;
            padding-left: 28px;
        }}
        li {{ margin-bottom: 6px; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 16px 0;
            font-size: 14px;
        }}
        th {{
            background: var(--primary);
            color: white;
            padding: 10px 16px;
            text-align: left;
        }}
        td {{
            padding: 10px 16px;
            border-bottom: 1px solid var(--border);
        }}
        tr:nth-child(even) td {{ background: var(--bg-alt); }}
        tr:hover td {{ background: var(--primary-light); }}
        code {{
            background: #f3f4f6;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 14px;
            color: #e11d48;
        }}
        pre {{
            background: #1e293b;
            color: #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 16px 0;
            font-size: 14px;
            line-height: 1.6;
        }}
        pre code {{
            background: none;
            color: inherit;
            padding: 0;
        }}
        hr {{
            border: none;
            height: 1px;
            background: var(--border);
            margin: 32px 0;
        }}
        strong {{ color: var(--text); }}
        .meta {{
            color: var(--text-secondary);
            font-size: 14px;
            margin-bottom: 32px;
        }}
        .risk-box {{
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 8px;
            padding: 20px;
            margin: 16px 0;
        }}
        .disclaimer {{
            background: var(--bg-alt);
            border-radius: 8px;
            padding: 20px;
            margin: 16px 0;
            font-size: 14px;
            color: var(--text-secondary);
        }}
        .footer {{
            text-align: center;
            color: var(--text-secondary);
            font-size: 13px;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }}
        @media (max-width: 768px) {{
            .container {{ padding: 30px 24px; margin: 16px; }}
            h1 {{ font-size: 22px; }}
        }}
    </style>
</head>
<body>
<div class="container">
{body}
<div class="footer">
    AI 投研助手 Charles 生成 | {generation_time}
</div>
</div>
</body>
</html>"""


def markdown_to_html(md_text):
    """
    简易 Markdown -> HTML 转换，覆盖研报中常用的语法。
    不依赖第三方库（markdown/mistune 等），保持技能自包含。
    """
    lines = md_text.split('\n')
    html_lines = []
    in_list = False
    in_ol = False
    in_code_block = False
    in_table = False
    in_blockquote = False

    i = 0
    while i < len(lines):
        line = lines[i]

        # 代码块
        if line.strip().startswith('```'):
            if in_code_block:
                html_lines.append('</code></pre>')
                in_code_block = False
            else:
                in_code_block = True
                html_lines.append('<pre><code>')
            i += 1
            continue

        if in_code_block:
            html_lines.append(_escape_html(line))
            i += 1
            continue

        # 结束之前的列表
        if in_list and not line.strip().startswith('- '):
            html_lines.append('</ul>')
            in_list = False
        if in_ol and not re.match(r'^\d+\.\s', line.strip()):
            html_lines.append('</ol>')
            in_ol = False
        if in_blockquote and not line.strip().startswith('>'):
            html_lines.append('</blockquote>')
            in_blockquote = False

        stripped = line.strip()

        # 空行
        if not stripped:
            i += 1
            continue

        # 水平线
        if stripped in ('---', '***', '___'):
            html_lines.append('<hr>')
            i += 1
            continue

        # 表格
        if '|' in stripped and stripped.startswith('|'):
            if not in_table:
                in_table = True
                html_lines.append('<table>')
                cells = [c.strip() for c in stripped.split('|')[1:-1]]
                html_lines.append('<tr>' + ''.join(f'<th>{_inline(c)}</th>' for c in cells) + '</tr>')
                # 跳过分隔行
                if i + 1 < len(lines) and re.match(r'^\|[\s\-:|]+\|$', lines[i + 1].strip()):
                    i += 1
            else:
                cells = [c.strip() for c in stripped.split('|')[1:-1]]
                html_lines.append('<tr>' + ''.join(f'<td>{_inline(c)}</td>' for c in cells) + '</tr>')

            if i + 1 >= len(lines) or '|' not in lines[i + 1]:
                html_lines.append('</table>')
                in_table = False
            i += 1
            continue

        # 标题
        m = re.match(r'^(#{1,6})\s+(.+)$', stripped)
        if m:
            level = len(m.group(1))
            text = _inline(m.group(2))
            html_lines.append(f'<h{level}>{text}</h{level}>')
            i += 1
            continue

        # 引用
        if stripped.startswith('>'):
            if not in_blockquote:
                in_blockquote = True
                html_lines.append('<blockquote>')
            content = stripped.lstrip('>').strip()
            html_lines.append(f'<p>{_inline(content)}</p>')
            i += 1
            continue

        # 无序列表
        if stripped.startswith('- '):
            if not in_list:
                in_list = True
                html_lines.append('<ul>')
            html_lines.append(f'<li>{_inline(stripped[2:])}</li>')
            i += 1
            continue

        # 有序列表
        m_ol = re.match(r'^(\d+)\.\s+(.+)$', stripped)
        if m_ol:
            if not in_ol:
                in_ol = True
                html_lines.append('<ol>')
            html_lines.append(f'<li>{_inline(m_ol.group(2))}</li>')
            i += 1
            continue

        # 普通段落
        html_lines.append(f'<p>{_inline(stripped)}</p>')
        i += 1

    # 关闭未关闭的标签
    if in_list:
        html_lines.append('</ul>')
    if in_ol:
        html_lines.append('</ol>')
    if in_blockquote:
        html_lines.append('</blockquote>')
    if in_table:
        html_lines.append('</table>')
    if in_code_block:
        html_lines.append('</code></pre>')

    return '\n'.join(html_lines)


def _escape_html(text):
    """转义 HTML 特殊字符"""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _inline(text):
    """处理行内 Markdown 语法: 粗体、斜体、行内代码、链接"""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    return text


def save_report_html(md_content, title, output_dir, filename=None):
    """
    将 Markdown 内容转换为 HTML 并保存。

    Args:
        md_content: Markdown 格式的研报内容
        title: 报告标题
        output_dir: 输出目录
        filename: 可选文件名（不含扩展名）

    Returns:
        HTML 文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    body_html = markdown_to_html(md_content)
    generation_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_html = HTML_TEMPLATE.format(
        title=title,
        body=body_html,
        generation_time=generation_time,
    )

    if not filename:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)
        filename = f"{safe_title}_{date_str}"

    filepath = os.path.join(output_dir, f"{filename}.html")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(full_html)

    return filepath


def main():
    parser = argparse.ArgumentParser(description="研报生成器")
    parser.add_argument("--analysis_file", default="", help="五步法分析结果 JSON 文件")
    parser.add_argument("--content", default="", help="直接传入 Markdown 文本内容")
    parser.add_argument("--title", default="", help="报告标题（--content 模式使用）")
    parser.add_argument("--output_dir", default="./reports", help="研报输出目录")
    parser.add_argument("--format", default="html", choices=["html", "md"],
                        help="输出格式: html（默认）或 md")
    args = parser.parse_args()

    # 模式1: 直接传入 Markdown 内容（Agent 输出的研报文本）
    if args.content:
        title = args.title or "AI 研报"
        md_content = args.content

        if args.format == "html":
            filepath = save_report_html(md_content, title, args.output_dir)
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)
            filepath = os.path.join(args.output_dir, f"{safe_title}_{date_str}.md")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(md_content)

        result = {
            "status": "success",
            "title": title,
            "format": args.format,
            "report_file": filepath,
            "report_length": len(md_content),
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 模式2: 从分析结果 JSON 生成（传统五步法流程）
    if not args.analysis_file:
        print(json.dumps({"error": "需要 --analysis_file 或 --content 参数"},
                          ensure_ascii=False))
        sys.exit(1)

    if not os.path.exists(args.analysis_file):
        print(f"[错误] 分析文件不存在: {args.analysis_file}")
        sys.exit(1)

    analysis = load_analysis(args.analysis_file)
    stock_name = analysis["stock_name"]

    report = generate_report(analysis)

    if args.format == "html":
        filepath = save_report_html(report, f"{stock_name} - 深度分析报告",
                                    args.output_dir, f"{stock_name}_深度研报")
    else:
        filepath = save_report(report, stock_name, args.output_dir)

    result = {
        "status": "success",
        "stock_name": stock_name,
        "format": args.format,
        "report_file": filepath,
        "report_length": len(report),
        "steps_included": len(analysis["steps"]),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
