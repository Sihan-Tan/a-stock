# -*- coding: utf-8 -*-
# 21-CASE-D: 投资晨会 LangGraph 工作流
"""
MorningBriefGraph -- 投资晨会工作流

每日 9:00 cron 触发, 跑完整 LangGraph 工作流, 9:20 推送钉钉/微信群

节点设计 (LangGraph StateGraph, 4 节点线性 DAG):

    START
      v
    industry_node     -- 申万二级板块强度 + 一二阶导拐点 (内嵌 lib/rotation_runner.py, 来自 CASE-B)
      v
    stock_picker_node -- 在 Top 板块成分股做多因子选股 (内嵌 lib/factor_runner.py, 来自 CASE-C)
                        成分股从 trade_stock_status.sector_2 反查, 不依赖 xtdata 在线
      v
    report_node       -- 拼装晨报 HTML / Markdown
      v
    push_node         -- 推送钉钉 / 企业微信
      v
    END

数据来源:
    DB (wucai_trade.*): trade_stock_status / trade_stock_daily / trade_sector_daily
    需要 CASE-A 已经跑过 (run_init.py), 表结构与 WucaiTrade 项目对齐

State (MorningState):
    - 输入:    trigger_time, top_n_industries, top_n_stocks, lookback_days
    - 中间产出: industry_rank, stock_pool, factor_rank, picked_stocks
    - 输出:    report_md, report_html, push_result

"""

from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, TypedDict
from operator import add

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(THIS_DIR / "lib"))

from langgraph.graph import END, START, StateGraph

from pusher import push_all


# ============================================================
# State 定义
# ============================================================

class MorningState(TypedDict, total=False):
    # === 输入 ===
    trigger_time: str        # ISO 时间
    industry_level: int      # 申万级别 1 / 2 (默认 2)
    top_n_industries: int    # 取多少个 Top 板块, 默认 5
    top_n_stocks: int        # 最终选多少只, 默认 5
    lookback_days: int       # 拉多少日 K 线
    sample_stocks: int       # 限制单板块选股池大小 (节省时间)

    # === 中间产出 ===
    industry_rank: list      # CASE-C 行业排名表
    stock_pool: list         # 候选股票池 (来自 Top 板块的成分股)
    factor_rank: list        # CASE-B 因子排名
    picked_stocks: list      # 最终选中的标的 (含因子分)

    # === 输出 ===
    report_md: str
    report_html: str
    push_result: dict

    # === 审计日志 ===
    messages: Annotated[list, add]


# ============================================================
# 节点 1: 行业强度排名 (调 CASE-C)
# ============================================================

def industry_node(state: dict) -> dict:
    print("\n" + "=" * 70)
    print("  [节点 1] industry_node -- 申万二级板块强度 + 一二阶导拐点")
    print("=" * 70)

    from rotation_runner import rank_industries_with_phase
    level   = state.get("industry_level", 2)
    top_n   = state.get("top_n_industries", 5)
    df = rank_industries_with_phase(level=level,
                                     lookback_days=state.get("lookback_days", 90),
                                     top_n=top_n)

    rank_list = []
    for ind_name, row in df.iterrows():
        rank_list.append({
            "industry":   ind_name,
            "rank":       int(row["composite_rank"]),
            "score":      round(float(row["composite_score"]), 3),
            "raw_score":  round(float(row["score"]), 3),
            "MOM_21":     round(float(row["MOM_21"]) * 100, 2),
            "RS_60":      round(float(row["RS_60"]) * 100, 2),
            "VOL_R":      round(float(row["VOL_RATIO"]), 2),
            "phase":      row.get("phase", "neutral"),
            "phase_desc": row.get("phase_desc", "中性"),
            "ROC_20":     round(float(row.get("ROC_20", 0)), 2),
            "members":    int(row["member_count"]),
        })

    print(f"  Top {top_n} 板块 (申万 {'一' if level == 1 else '二'} 级):")
    for r in rank_list:
        print(f"    [{r['rank']:>2}] {r['industry']:<14s} "
              f"score={r['score']:+.2f} ({r['phase_desc']:<6s})  "
              f"MOM21={r['MOM_21']:+5.2f}%  RS60={r['RS_60']:+5.2f}%  "
              f"ROC20={r['ROC_20']:+5.2f}%")

    return {
        "industry_rank": rank_list,
        "messages": [{"role": "industry", "time": datetime.now().strftime("%H:%M:%S"),
                      "content": f"Top {top_n} 板块: " + ", ".join(r["industry"] for r in rank_list)}],
    }


# ============================================================
# 节点 2: 多因子选股 (调 CASE-B)
# ============================================================

def stock_picker_node(state: dict) -> dict:
    print("\n" + "=" * 70)
    print("  [节点 2] stock_picker_node -- Top 板块成分股做多因子选股")
    print("=" * 70)

    from rotation_runner import get_sector_member_codes
    from factor_runner import filter_tradable, calc_factors_batch, preprocess_factors

    industry_rank = state.get("industry_rank", [])
    level = state.get("industry_level", 2)
    if not industry_rank:
        print("  [SKIP] 无行业排名, 跳过选股")
        return {"stock_pool": [], "factor_rank": [], "picked_stocks": []}

    # 从 trade_stock_status 反查 Top 板块的成分股 (走 DB, 不依赖 xtdata 在线)
    sample_per_industry = state.get("sample_stocks", 30)
    top_stocks: List[str] = []
    industry_to_codes: Dict[str, List[str]] = {}
    for r in industry_rank:
        ind_name = r["industry"]
        codes = get_sector_member_codes(ind_name, level=level)
        codes = filter_tradable(codes)[:sample_per_industry]
        industry_to_codes[ind_name] = codes
        top_stocks.extend(codes)
    top_stocks = sorted(set(top_stocks))
    print(f"  候选股票池: {len(top_stocks)} 只 (来自 {len(industry_rank)} 个 Top 板块)")

    # 算因子矩阵
    factor_df = calc_factors_batch(top_stocks)

    if len(factor_df) < 5:
        print("  [WARN] 因子计算结果太少, 跳过选股")
        return {"stock_pool": top_stocks, "factor_rank": [], "picked_stocks": []}

    # 反向行业映射
    ind_map = {c: ind for ind, codes in industry_to_codes.items() for c in codes}

    # 预处理
    factor_processed = preprocess_factors(factor_df, industry_map=ind_map, neutralize=True)

    # 等权合成 (实际项目可换成 IC 加权 -- 见 CASE-B docs)
    alpha = factor_processed.mean(axis=1).dropna().sort_values(ascending=False)

    top_n = state.get("top_n_stocks", 5)
    picked = []
    factor_rank_list = []
    for code in alpha.head(top_n * 3).index:    # 前 N×3 都展示
        factor_rank_list.append({
            "code":     code,
            "industry": ind_map.get(code, "未分类"),
            "alpha":    round(float(alpha[code]), 3),
            "raw_factors": {k: round(float(v), 3) for k, v in factor_df.loc[code].items()
                            if k in ("MOM_1M", "MOM_3M", "VOL_20", "RSI_14", "BIAS_20")},
        })
    picked = factor_rank_list[:top_n]

    print(f"  Top {top_n} 选中标的:")
    for p in picked:
        print(f"    {p['code']}  [{p['industry']:<6s}]  alpha={p['alpha']:+.3f}  "
              f"MOM_3M={p['raw_factors'].get('MOM_3M', 0):+.2%}")

    return {
        "stock_pool":    top_stocks,
        "factor_rank":   factor_rank_list,
        "picked_stocks": picked,
        "messages": [{"role": "stock_picker", "time": datetime.now().strftime("%H:%M:%S"),
                      "content": f"选中 {len(picked)} 只: " + ", ".join(p["code"] for p in picked)}],
    }


# ============================================================
# 节点 3: 拼装晨报
# ============================================================

def report_node(state: dict) -> dict:
    print("\n" + "=" * 70)
    print("  [节点 3] report_node -- 拼装晨报")
    print("=" * 70)

    today_str = datetime.now().strftime("%Y-%m-%d %A")
    industries = state.get("industry_rank", [])
    picked = state.get("picked_stocks", [])

    # ---- Markdown 版 (推送到钉钉/微信) ----
    md_lines = [
        f"# 投资晨会简报 -- {today_str}",
        "",
        f"## Top {len(industries)} 强势板块 (申万二级)",
        "",
        "| Rank | 板块 | 综合分 | 拐点信号 | 21日动量 | 60日相对强度 | 20日ROC |",
        "|------|------|--------|----------|----------|--------------|---------|",
    ]
    for r in industries:
        md_lines.append(
            f"| {r['rank']} | **{r['industry']}** | {r['score']:+.2f} | "
            f"{r.get('phase_desc', '中性')} | "
            f"{r['MOM_21']:+.2f}% | {r['RS_60']:+.2f}% | {r.get('ROC_20', 0):+.2f}% |"
        )
    md_lines += ["", f"## Top {len(picked)} 选中标的", ""]
    md_lines.append("| 代码 | 行业 | 综合alpha | 3M动量 |")
    md_lines.append("|------|------|-----------|--------|")
    for p in picked:
        md_lines.append(
            f"| `{p['code']}` | {p['industry']} | {p['alpha']:+.3f} | "
            f"{p['raw_factors'].get('MOM_3M', 0):+.2%} |"
        )

    md_lines += ["", "## 盘中应对建议", ""]
    if picked:
        for p in picked:
            md_lines.append(
                f"- `{p['code']}` ({p['industry']}): "
                f"alpha={p['alpha']:+.3f}, 关注开盘 30 分钟方向"
            )
    else:
        md_lines.append("- 无候选标的, 今日观望")

    md_lines += ["", "---", "",
                 "> 本简报由 AI 量化团队自动生成, 仅供参考, 不构成投资建议",
                 f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]

    report_md = "\n".join(md_lines)

    # ---- HTML 版 (邮件附件) ----
    report_html = _md_to_html(report_md)

    # 落盘
    output_dir = THIS_DIR / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    md_path = output_dir / f"morning_brief_{ts}.md"
    html_path = output_dir / f"morning_brief_{ts}.html"
    md_path.write_text(report_md, encoding="utf-8")
    html_path.write_text(report_html, encoding="utf-8")

    print(f"  晨报已落盘:")
    print(f"    Markdown: {md_path}")
    print(f"    HTML:     {html_path}")

    return {
        "report_md":   report_md,
        "report_html": str(html_path),
        "messages": [{"role": "report", "time": datetime.now().strftime("%H:%M:%S"),
                      "content": f"晨报生成 {len(report_md)} 字节"}],
    }


def _md_to_html(md: str) -> str:
    """简版 Markdown 渲染 (跟 20 章 charles_node 一致, 但更简化)"""
    import re
    from html import escape
    lines = md.splitlines()
    out = ["<!DOCTYPE html><html lang='zh-CN'><head><meta charset='UTF-8'>",
           "<title>投资晨会简报</title>",
           "<style>",
           "body{font-family:-apple-system,'Microsoft YaHei',sans-serif;max-width:900px;margin:30px auto;padding:0 24px;color:#2c3e50;line-height:1.7}",
           "h1{border-bottom:3px solid #3498db;padding-bottom:10px}",
           "h2{color:#3498db;margin-top:30px}",
           "table{border-collapse:collapse;width:100%;margin:14px 0}",
           "th{background:#34495e;color:#fff;padding:8px 12px;text-align:left}",
           "td{padding:8px 12px;border:1px solid #dee2e6}",
           "tr:nth-child(even){background:#f8f9fa}",
           "code{background:#e8ecef;padding:2px 6px;border-radius:4px;font-family:'Consolas',monospace}",
           "blockquote{border-left:3px solid #95a5a6;color:#555;padding-left:12px;background:#f1f3f5;padding-top:8px;padding-bottom:8px}",
           "</style></head><body>"]

    in_table = False
    table_rows = []
    for line in lines:
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            # 跳过分隔行 |---|---|
            if all(re.match(r"^-+$", c) for c in cells):
                continue
            if not in_table:
                in_table = True
                table_rows = ["<table><thead><tr>"]
                for c in cells:
                    table_rows.append(f"<th>{escape(c)}</th>")
                table_rows.append("</tr></thead><tbody>")
            else:
                table_rows.append("<tr>")
                for c in cells:
                    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", escape(c))
                    rendered = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", rendered)
                    table_rows.append(f"<td>{rendered}</td>")
                table_rows.append("</tr>")
            continue
        else:
            if in_table:
                table_rows.append("</tbody></table>")
                out.extend(table_rows)
                in_table = False
                table_rows = []

        if s.startswith("# "):
            out.append(f"<h1>{escape(s[2:])}</h1>")
        elif s.startswith("## "):
            out.append(f"<h2>{escape(s[3:])}</h2>")
        elif s.startswith("- "):
            li_html = re.sub(r"`([^`]+)`", r"<code>\1</code>", escape(s[2:]))
            out.append(f"<li>{li_html}</li>")
        elif s.startswith("> "):
            out.append(f"<blockquote>{escape(s[2:])}</blockquote>")
        elif s == "---":
            out.append("<hr>")
        elif s == "":
            continue
        else:
            p_html = re.sub(r"`([^`]+)`", r"<code>\1</code>", escape(s))
            out.append(f"<p>{p_html}</p>")

    if in_table:
        table_rows.append("</tbody></table>")
        out.extend(table_rows)

    out.append("</body></html>")
    return "\n".join(out)


# ============================================================
# 节点 4: 推送
# ============================================================

def push_node(state: dict) -> dict:
    print("\n" + "=" * 70)
    print("  [节点 4] push_node -- 推送钉钉 / 企业微信")
    print("=" * 70)

    title = f"投资晨会 {datetime.now().strftime('%m-%d')}"
    md = state.get("report_md", "")
    if not md:
        print("  [SKIP] 无内容可推送")
        return {"push_result": {}}

    result = push_all(title=title, content=md)
    return {
        "push_result": result,
        "messages": [{"role": "push", "time": datetime.now().strftime("%H:%M:%S"),
                      "content": f"推送结果: {result}"}],
    }


# ============================================================
# 编排图
# ============================================================

def build_graph():
    g = StateGraph(MorningState)
    g.add_node("industry",     industry_node)
    g.add_node("stock_picker", stock_picker_node)
    g.add_node("report",       report_node)
    g.add_node("push",         push_node)

    g.add_edge(START, "industry")
    g.add_edge("industry", "stock_picker")
    g.add_edge("stock_picker", "report")
    g.add_edge("report", "push")
    g.add_edge("push", END)

    return g.compile()


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="投资晨会工作流")
    parser.add_argument("--level", type=int, choices=[1, 2], default=2,
                        help="申万级别 (默认 2 二级板块)")
    parser.add_argument("--top-industries", type=int, default=5,
                        help="选 Top N 强势板块 (默认 5)")
    parser.add_argument("--top-stocks", type=int, default=5,
                        help="最终输出 Top N 选股 (默认 5)")
    parser.add_argument("--sample-per-industry", type=int, default=20,
                        help="每个板块选取多少只候选股 (默认 20)")
    parser.add_argument("--lookback", type=int, default=90,
                        help="拉多少日 K 线 (默认 90)")
    args = parser.parse_args()

    print()
    print("#" * 70)
    print("# 投资晨会工作流启动")
    print(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("#" * 70)

    graph = build_graph()
    result = graph.invoke({
        "trigger_time":     datetime.now().isoformat(timespec="seconds"),
        "industry_level":   args.level,
        "top_n_industries": args.top_industries,
        "top_n_stocks":     args.top_stocks,
        "lookback_days":    args.lookback,
        "sample_stocks":    args.sample_per_industry,
        "messages":         [],
    })

    print("\n" + "#" * 70)
    print("# 工作流执行完成")
    print("#" * 70)
    print()
    print("--- 节点对话历史 ---")
    for m in result.get("messages", []):
        print(f"  [{m['time']}] {m['role']:<14s} | {m['content']}")
    print()
    print(f"晨报路径 (HTML): {result.get('report_html', '')}")
    print(f"推送结果:        {result.get('push_result', {})}")


if __name__ == "__main__":
    main()
