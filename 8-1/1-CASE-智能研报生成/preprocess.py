#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本 - 批量处理 PDF 研报/财报，构建统一 FAISS 索引和 SQLite 元数据库

功能:
  1. 扫描 data/reports/ 中的所有 PDF 文件
  2. 使用 PyPDF2 解析 PDF 提取文本
  3. 从文件名和内容中提取元数据(股票名称/代码、报告类型、发布日期等)
  4. 按 chunk 切分文本，每个 chunk 携带完整元数据(所属文档、页码、股票等)
  5. 构建统一 FAISS 向量索引(支持按股票/文档过滤检索)
  6. 将文档元数据存入 SQLite 数据库

使用方式:
  python preprocess.py                     # 增量处理新 PDF，重建索引
  python preprocess.py --rebuild           # 清除旧记录，全部重新处理
  python preprocess.py --list              # 查看已处理文档列表
  python preprocess.py --skip-index        # 只解析 PDF 和提取元数据，不构建 FAISS 索引

处理完成后，Agent 可直接通过 query_report.py 查询统一索引，无需再手动解析 PDF。
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---- 路径配置 ----
PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store"
DB_PATH = PROJECT_ROOT / "data" / "documents.db"

# ---- 已知的股票名称 -> 代码映射 (可按需扩展) ----
STOCK_CODE_MAP = {
    "中芯国际": "688981",
    "贵州茅台": "600519",
    "比亚迪": "002594",
    "宁德时代": "300750",
    "隆基绿能": "601012",
    "招商银行": "600036",
    "中国平安": "601318",
    "五粮液": "000858",
    "海天味业": "603288",
    "恒瑞医药": "600276",
}

# ---- 报告类型识别规则 ----
REPORT_TYPE_PATTERNS = [
    (r"年度报告|年报", "年报"),
    (r"半年度报告|半年报", "半年报"),
    (r"第一季度报告|一季度报告|一季报", "一季报"),
    (r"第三季度报告|三季度报告|三季报", "三季报"),
    (r"季度报告|季报", "季报"),
    (r"业绩点评|业绩快报", "业绩点评"),
    (r"深度研究|深度报告|研究报告", "深度研报"),
    (r"调研纪要", "调研纪要"),
]


# ==================== 文件名解析 ====================

def parse_filename(filename: str) -> Dict:
    """
    从 PDF 文件名中提取元数据

    支持的文件名格式:
    - 财报_{股票名}：{标题}.pdf        (用户上传的财报)
    - 【财报】{股票名}：{标题}.pdf     (旧格式财报)
    - 【{券商名}】{标题}.pdf           (券商研报)
    - {通用标题}.pdf                   (其他文档)
    """
    name = os.path.splitext(filename)[0]

    result = {
        "filename": filename,
        "title": name,
        "stock_name": "",
        "source": "",
        "report_type": "",
    }

    # 格式: 财报_{股票}：{标题}
    m = re.match(r"财报[_](.+?)[:：](.+)", name)
    if m:
        result["stock_name"] = m.group(1).strip()
        result["title"] = m.group(2).strip()
        result["source"] = "官方财报"
        _detect_report_type(result)
        return result

    # 格式: 【财报】{股票}：{标题}
    m = re.match(r"【财报】(.+?)[:：](.+)", name)
    if m:
        result["stock_name"] = m.group(1).strip()
        result["title"] = m.group(2).strip()
        result["source"] = "官方财报"
        _detect_report_type(result)
        return result

    # 格式: 【{券商/来源}】{标题}
    m = re.match(r"【(.+?)】(.+)", name)
    if m:
        result["source"] = m.group(1).strip()
        result["title"] = m.group(2).strip()
        _detect_stock_from_title(result)
        _detect_report_type(result)
        if not result["report_type"]:
            result["report_type"] = "研报"
        return result

    # 通用格式: 尝试从标题中提取股票名
    _detect_stock_from_title(result)
    _detect_report_type(result)
    return result


def _detect_stock_from_title(result: Dict):
    """从标题中识别股票名称"""
    for stock_name in STOCK_CODE_MAP:
        if stock_name in result["title"]:
            result["stock_name"] = stock_name
            return

    # 尝试匹配标题中的股票代码 (如 "中芯国际(688981)")
    m = re.search(r"[（(](\d{6})[）)]", result["title"])
    if m:
        code = m.group(1)
        for name, c in STOCK_CODE_MAP.items():
            if c == code:
                result["stock_name"] = name
                return


def _detect_report_type(result: Dict):
    """从标题中识别报告类型"""
    for pattern, rtype in REPORT_TYPE_PATTERNS:
        if re.search(pattern, result["title"]):
            result["report_type"] = rtype
            return


# ==================== PDF 文本提取 ====================

def extract_pdf_text(pdf_path: str) -> Tuple[str, List[dict]]:
    """
    使用 PyPDF2 逐页提取 PDF 文本

    Returns:
        full_text: 完整文本
        pages: 每页信息列表 [{page, text, char_count}, ...]
    """
    reader = PdfReader(pdf_path)
    full_text = ""
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({
            "page": page_num,
            "text": text,
            "char_count": len(text),
        })
        full_text += text

    return full_text, pages


# ==================== 内容元数据提取 ====================

def extract_stock_code(text: str, stock_name: str = "") -> str:
    """
    提取股票代码

    优先使用已知映射，其次从 PDF 内容中匹配
    """
    if stock_name and stock_name in STOCK_CODE_MAP:
        return STOCK_CODE_MAP[stock_name]

    # 从文本前部搜索股票代码
    patterns = [
        r"(?:股票代码|证券代码|A\s*股.*?代码|代码)[：:\s]*(\d{6})",
        r"(?:Stock\s*Code)[：:\s]*(\d{6})",
        r"[（(](\d{6})[）)]",
    ]
    for pattern in patterns:
        m = re.search(pattern, text[:15000])
        if m:
            return m.group(1)

    return ""


def extract_stock_name_from_text(text: str) -> str:
    """从 PDF 正文中识别股票名称"""
    search_text = text[:8000]
    for stock_name in STOCK_CODE_MAP:
        if stock_name in search_text:
            return stock_name
    return ""


def extract_publish_date(text: str) -> str:
    """
    从 PDF 内容中提取发布/披露日期

    优先匹配带上下文关键词(披露日期、报告日期等)的日期，
    其次取文档前部出现的合理日期。
    """
    search_text = text[:15000]

    # 优先: 带上下文关键词的日期
    context_patterns = [
        r"(?:披露日期|公告日期|报告日期|发布日期|编制日期|批准报告日)[：:\s]*(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日",
        r"(?:披露日期|公告日期|报告日期|发布日期|编制日期)[：:\s]*(\d{4})-(\d{1,2})-(\d{1,2})",
    ]
    for pattern in context_patterns:
        m = re.search(pattern, search_text)
        if m:
            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if 2000 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                return f"{year:04d}-{month:02d}-{day:02d}"

    # 次选: 文档前3000字符中的日期，取最后一个(封面日期通常靠后)
    all_dates = []
    for m in re.finditer(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", search_text[:3000]):
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 2020 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
            all_dates.append(f"{year:04d}-{month:02d}-{day:02d}")

    if all_dates:
        return all_dates[-1]

    return ""


def extract_report_period(text: str) -> str:
    """提取报告期间，如 '2025年度'、'2025年第三季度'"""
    search_text = text[:8000]

    patterns = [
        r"(\d{4})\s*年\s*(第[一二三四]季度)",
        r"(\d{4})\s*年\s*(半年度)",
        r"(\d{4})\s*年[度]?\s*(年度报告|年报)",
    ]

    for pattern in patterns:
        m = re.search(pattern, search_text)
        if m:
            year = m.group(1)
            period = m.group(2)
            if "年度报告" in period or "年报" in period:
                return f"{year}年度"
            return f"{year}年{period}"

    # 尝试从报告期间范围推断
    m = re.search(
        r"报告期[间]?\s*[：:\s]*\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s*至\s*\d{4}\s*年\s*(\d{1,2})\s*月",
        search_text,
    )
    if m:
        end_month = int(m.group(1))
        year_m = re.search(r"(\d{4})\s*年", search_text)
        year = year_m.group(1) if year_m else ""
        if year:
            if end_month <= 3:
                return f"{year}年第一季度"
            elif end_month <= 6:
                return f"{year}年半年度"
            elif end_month <= 9:
                return f"{year}年第三季度"
            else:
                return f"{year}年度"

    return ""


# ==================== 文本切分 ====================

def chunk_with_metadata(
    full_text: str,
    pages: List[dict],
    doc_metadata: Dict,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    将文本切分为 chunks，每个 chunk 携带完整元数据

    metadata 字段:
    - doc_id, filename, title, stock_name, stock_code
    - source, report_type, publish_date
    - page (所在页码), chunk_index (切片序号)
    - total_chunks, total_pages
    """
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", ".", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = splitter.split_text(full_text)

    # 构建页码边界(字符偏移量 -> 页码)
    page_boundaries = []
    offset = 0
    for p in pages:
        page_boundaries.append({
            "page": p["page"],
            "start": offset,
            "end": offset + p["char_count"],
        })
        offset += p["char_count"]

    # 为每个 chunk 分配页码
    result = []
    for idx, chunk_text in enumerate(chunks):
        page_num = -1
        chunk_start = full_text.find(chunk_text[:100])
        if chunk_start >= 0:
            for pb in page_boundaries:
                if pb["start"] <= chunk_start < pb["end"]:
                    page_num = pb["page"]
                    break

        metadata = {
            "doc_id": doc_metadata.get("doc_id", 0),
            "filename": doc_metadata.get("filename", ""),
            "title": doc_metadata.get("title", ""),
            "stock_name": doc_metadata.get("stock_name", ""),
            "stock_code": doc_metadata.get("stock_code", ""),
            "source": doc_metadata.get("source", ""),
            "report_type": doc_metadata.get("report_type", ""),
            "publish_date": doc_metadata.get("publish_date", ""),
            "page": page_num,
            "chunk_index": idx,
            "total_chunks": len(chunks),
            "total_pages": doc_metadata.get("total_pages", 0),
        }

        result.append({"text": chunk_text, "metadata": metadata})

    return result


# ==================== SQLite 数据库 ====================

def init_db(db_path: str) -> sqlite3.Connection:
    """初始化 SQLite 数据库，创建 documents 表"""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            title TEXT,
            stock_name TEXT,
            stock_code TEXT,
            source TEXT,
            report_type TEXT,
            report_period TEXT,
            publish_date TEXT,
            total_pages INTEGER,
            total_chars INTEGER,
            total_chunks INTEGER,
            created_at TEXT DEFAULT (datetime('now', 'localtime'))
        )
    """)
    conn.commit()
    return conn


def get_processed_files(conn: sqlite3.Connection) -> set:
    """获取已处理的文件名集合"""
    cursor = conn.execute("SELECT filename FROM documents")
    return {row[0] for row in cursor.fetchall()}


def insert_document(conn: sqlite3.Connection, doc: Dict) -> int:
    """插入或更新文档元数据，返回 doc_id"""
    conn.execute("""
        INSERT OR REPLACE INTO documents
        (filename, title, stock_name, stock_code, source, report_type,
         report_period, publish_date, total_pages, total_chars, total_chunks)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        doc["filename"], doc["title"], doc["stock_name"], doc["stock_code"],
        doc["source"], doc["report_type"], doc.get("report_period", ""),
        doc["publish_date"], doc["total_pages"], doc["total_chars"],
        doc["total_chunks"],
    ))
    conn.commit()
    cursor = conn.execute(
        "SELECT id FROM documents WHERE filename = ?", (doc["filename"],)
    )
    return cursor.fetchone()[0]


def list_documents(conn: sqlite3.Connection):
    """打印所有已处理的文档信息"""
    cursor = conn.execute("""
        SELECT id, filename, stock_name, stock_code, report_type,
               publish_date, total_pages, total_chunks, created_at
        FROM documents ORDER BY id
    """)
    rows = cursor.fetchall()
    if not rows:
        print("  (暂无已处理文档)")
        return

    print(f"  {'ID':>3} | {'股票':>8} | {'代码':>6} | {'类型':>8} | "
          f"{'发布日期':>10} | {'页数':>4} | {'Chunks':>6} | 文件名")
    print("  " + "-" * 110)
    for row in rows:
        print(f"  {row[0]:>3} | {row[2] or '-':>8} | {row[3] or '-':>6} | "
              f"{row[4] or '-':>8} | {row[5] or '-':>10} | "
              f"{row[6]:>4} | {row[7]:>6} | {row[1]}")


# ==================== FAISS 索引构建 ====================

def build_unified_index(
    all_chunks: List[Dict],
    index_dir: str,
    embedding_model: str = "text-embedding-v4",
):
    """
    构建统一 FAISS 向量索引

    每个 chunk 的 metadata 会保存在 FAISS 的 docstore 中，
    查询时可通过 filter 参数按 stock_code 等字段过滤。
    同时保存 chunks.json 供 BM25 混合检索使用。
    """
    from langchain_community.embeddings import DashScopeEmbeddings
    from langchain_community.vectorstores import FAISS

    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("[错误] 请设置环境变量 DASHSCOPE_API_KEY")
        sys.exit(1)

    embeddings = DashScopeEmbeddings(
        model=embedding_model,
        dashscope_api_key=api_key,
    )

    texts = [c["text"] for c in all_chunks]
    metadatas = [c["metadata"] for c in all_chunks]

    print(f"[索引] 使用 {embedding_model} 为 {len(texts)} 个 chunks 生成向量...")

    # DashScope text-embedding-v4 每次最多 25 条
    batch_size = 25
    vectorstore = None

    max_retries = 6

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metas = metadatas[i:i + batch_size]

        for attempt in range(max_retries):
            try:
                if vectorstore is None:
                    vectorstore = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_metas)
                else:
                    vectorstore.add_texts(batch_texts, metadatas=batch_metas)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = (attempt + 1) * 5
                    print(f"  [重试] 第 {attempt + 1} 次失败({type(e).__name__}), {wait}s 后重试...")
                    time.sleep(wait)
                else:
                    print(f"  [错误] 第 {i // batch_size + 1} 批嵌入失败(已重试{max_retries}次): {e}")
                    raise

        done = min(i + batch_size, len(texts))
        print(f"  [{done}/{len(texts)}] 已完成")

        if i + batch_size < len(texts):
            time.sleep(1)

    # 保存 FAISS 索引
    # FAISS 底层 C++ fopen 不支持 Windows 非 ASCII 路径,
    # 先保存到临时目录再移动过来
    os.makedirs(index_dir, exist_ok=True)

    import tempfile
    import shutil

    tmp_dir = tempfile.mkdtemp(prefix="faiss_")
    try:
        vectorstore.save_local(tmp_dir)
        for fname in os.listdir(tmp_dir):
            src = os.path.join(tmp_dir, fname)
            dst = os.path.join(index_dir, fname)
            shutil.move(src, dst)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # 保存 chunks.json (供 BM25 混合检索和调试)
    chunks_data = [
        {"text": c["text"], "metadata": c["metadata"]}
        for c in all_chunks
    ]
    chunks_json_path = os.path.join(index_dir, "chunks.json")
    with open(chunks_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    # 收集索引统计信息
    doc_ids = set()
    stock_codes = set()
    for m in metadatas:
        doc_ids.add(m.get("doc_id", 0))
        if m.get("stock_code"):
            stock_codes.add(m["stock_code"])

    index_meta = {
        "total_chunks": len(texts),
        "total_documents": len(doc_ids),
        "embedding_model": embedding_model,
        "chunk_size": 800,
        "chunk_overlap": 150,
        "stock_codes": sorted(stock_codes),
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_path = os.path.join(index_dir, "index_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(index_meta, f, ensure_ascii=False, indent=2)

    print(f"[索引] 统一索引已保存: {index_dir}")
    print(f"  {len(texts)} 个 chunks, {len(doc_ids)} 个文档, "
          f"{len(stock_codes)} 只股票 {sorted(stock_codes)}")

    return vectorstore


# ==================== 单文档处理 ====================

def process_single_pdf(
    pdf_path: str,
    conn: sqlite3.Connection,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> Tuple[int, List[Dict]]:
    """
    处理单个 PDF 文件:
    解析文本 -> 提取元数据 -> 切分 chunks -> 保存到 SQLite 和 parsed 目录

    Returns:
        doc_id: 文档在 SQLite 中的 ID
        chunks: 切分后的 chunk 列表(每个带 metadata)
    """
    filename = os.path.basename(pdf_path)
    print(f"\n{'=' * 60}")
    print(f"[处理] {filename}")

    # 1. 解析文件名获取初步元数据
    file_meta = parse_filename(filename)
    print(f"  标题: {file_meta['title']}")
    print(f"  股票: {file_meta['stock_name'] or '(待从内容识别)'}")
    print(f"  来源: {file_meta['source'] or '(未知)'}")
    print(f"  类型: {file_meta['report_type'] or '(待识别)'}")

    # 2. 提取 PDF 文本
    full_text, pages = extract_pdf_text(pdf_path)
    total_pages = len(pages)
    non_empty = sum(1 for p in pages if p["char_count"] > 0)
    print(f"  页数: {total_pages} (有文本: {non_empty})")
    print(f"  字符: {len(full_text)}")

    if not full_text.strip():
        print("  [警告] 未提取到文本，可能是扫描件 PDF")
        return -1, []

    # 3. 从内容中补充元数据
    if not file_meta["stock_name"]:
        file_meta["stock_name"] = extract_stock_name_from_text(full_text)

    stock_code = extract_stock_code(full_text, file_meta["stock_name"])
    publish_date = extract_publish_date(full_text)
    report_period = extract_report_period(full_text)

    # 如果从内容中找到了股票代码但还没有名称，反查映射
    if not file_meta["stock_name"] and stock_code:
        for name, code in STOCK_CODE_MAP.items():
            if code == stock_code:
                file_meta["stock_name"] = name
                break

    print(f"  代码: {stock_code or '(未识别)'}")
    print(f"  发布: {publish_date or '(未识别)'}")
    print(f"  期间: {report_period or '(未识别)'}")

    # 4. 存入 SQLite
    doc_data = {
        "filename": filename,
        "title": file_meta["title"],
        "stock_name": file_meta["stock_name"],
        "stock_code": stock_code,
        "source": file_meta["source"],
        "report_type": file_meta["report_type"],
        "report_period": report_period,
        "publish_date": publish_date,
        "total_pages": total_pages,
        "total_chars": len(full_text),
        "total_chunks": 0,
    }
    doc_id = insert_document(conn, doc_data)

    # 5. 切分文本
    doc_metadata = {"doc_id": doc_id, **doc_data, "total_pages": total_pages}
    chunks = chunk_with_metadata(full_text, pages, doc_metadata, chunk_size, chunk_overlap)

    # 更新 SQLite 中的 chunk 数量
    conn.execute("UPDATE documents SET total_chunks = ? WHERE id = ?", (len(chunks), doc_id))
    conn.commit()
    print(f"  Chunks: {len(chunks)}")

    # 6. 保存解析后的纯文本(供调试和后续使用)
    parsed_dir = str(PARSED_DIR)
    os.makedirs(parsed_dir, exist_ok=True)
    base = os.path.splitext(filename)[0]

    full_text_path = os.path.join(parsed_dir, f"{base}_full.txt")
    with open(full_text_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    pages_json_path = os.path.join(parsed_dir, f"{base}_pages.json")
    with open(pages_json_path, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)

    return doc_id, chunks


# ==================== 主流程 ====================

def main():
    parser = argparse.ArgumentParser(
        description="数据预处理 - 批量处理 PDF 并构建统一 FAISS 索引",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--rebuild", action="store_true",
                        help="清除旧记录，全部重新处理")
    parser.add_argument("--list", action="store_true",
                        help="列出已处理的文档")
    parser.add_argument("--skip-index", action="store_true",
                        help="只解析 PDF 和提取元数据，不构建 FAISS 索引")
    parser.add_argument("--reports_dir", default=str(REPORTS_DIR),
                        help="PDF 文件目录")
    parser.add_argument("--chunk_size", type=int, default=800,
                        help="切片大小(默认 800)")
    parser.add_argument("--chunk_overlap", type=int, default=150,
                        help="切片重叠(默认 150)")
    parser.add_argument("--embedding_model", default="text-embedding-v4",
                        help="Embedding 模型(默认 text-embedding-v4)")
    args = parser.parse_args()

    print("=" * 60)
    print("  数据预处理 - PDF 批量处理 & 统一索引构建")
    print("=" * 60)

    # 初始化 SQLite 数据库
    os.makedirs(os.path.dirname(str(DB_PATH)), exist_ok=True)
    conn = init_db(str(DB_PATH))

    # --list 模式: 查看已处理文档
    if args.list:
        print("\n[已处理文档]")
        list_documents(conn)
        conn.close()
        return

    # 扫描 PDF 文件
    reports_dir = args.reports_dir
    if not os.path.exists(reports_dir):
        print(f"[错误] 报告目录不存在: {reports_dir}")
        sys.exit(1)

    pdf_files = sorted([
        f for f in os.listdir(reports_dir)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print("[提示] 未发现 PDF 文件")
        conn.close()
        return

    print(f"\n[扫描] 发现 {len(pdf_files)} 个 PDF 文件")

    # 重建模式: 清除旧记录
    if args.rebuild:
        conn.execute("DELETE FROM documents")
        conn.commit()
        print("[重建] 已清除旧的处理记录")
        processed = set()
    else:
        processed = get_processed_files(conn)

    new_files = [f for f in pdf_files if f not in processed]
    skip_files = [f for f in pdf_files if f in processed]

    # 检查统一索引是否存在
    index_exists = os.path.exists(os.path.join(str(VECTOR_STORE_DIR), "index.faiss"))

    if not new_files and index_exists:
        print("[提示] 所有文件已处理且索引完整。使用 --rebuild 可强制重建。")
        print("\n[已处理文档]")
        list_documents(conn)
        conn.close()
        return

    if not new_files and not index_exists:
        print("[提示] 所有文件已处理，但统一索引缺失，将从缓存重建索引...")
        skip_files = list(processed)

    print(f"[待处理] {len(new_files)} 个新文件")
    if skip_files:
        print(f"[已跳过] {len(skip_files)} 个已处理文件")

    # ---- 阶段1: 解析所有新 PDF ----
    all_chunks = []
    success_count = 0

    for filename in new_files:
        pdf_path = os.path.join(reports_dir, filename)
        try:
            doc_id, chunks = process_single_pdf(
                pdf_path, conn, args.chunk_size, args.chunk_overlap
            )
            if chunks:
                all_chunks.extend(chunks)
                success_count += 1
        except Exception as e:
            print(f"  [错误] 处理失败: {type(e).__name__}: {e}")
            continue

    # ---- 阶段2: 加载已处理文档的 chunks (用于重建统一索引) ----
    if skip_files:
        print(f"\n[加载] 正在加载 {len(skip_files)} 个已处理文档的切片...")
        for filename in skip_files:
            try:
                base = os.path.splitext(filename)[0]
                full_text_path = str(PARSED_DIR / f"{base}_full.txt")
                pages_json_path = str(PARSED_DIR / f"{base}_pages.json")

                if not os.path.exists(full_text_path):
                    # 缓存不存在，需要重新解析
                    pdf_path = os.path.join(reports_dir, filename)
                    if os.path.exists(pdf_path):
                        _, chunks = process_single_pdf(
                            pdf_path, conn, args.chunk_size, args.chunk_overlap
                        )
                        if chunks:
                            all_chunks.extend(chunks)
                    continue

                with open(full_text_path, "r", encoding="utf-8") as f:
                    full_text = f.read()

                pages = []
                if os.path.exists(pages_json_path):
                    with open(pages_json_path, "r", encoding="utf-8") as f:
                        pages = json.load(f)

                # 从 SQLite 获取元数据
                cursor = conn.execute(
                    "SELECT id, title, stock_name, stock_code, source, "
                    "report_type, report_period, publish_date, total_pages, total_chars "
                    "FROM documents WHERE filename = ?",
                    (filename,)
                )
                row = cursor.fetchone()
                if not row:
                    continue

                doc_metadata = {
                    "doc_id": row[0],
                    "filename": filename,
                    "title": row[1],
                    "stock_name": row[2],
                    "stock_code": row[3],
                    "source": row[4],
                    "report_type": row[5],
                    "report_period": row[6],
                    "publish_date": row[7],
                    "total_pages": row[8],
                    "total_chars": row[9],
                }
                chunks = chunk_with_metadata(
                    full_text, pages, doc_metadata,
                    args.chunk_size, args.chunk_overlap,
                )
                all_chunks.extend(chunks)
                print(f"  {filename} -> {len(chunks)} chunks")

            except Exception as e:
                print(f"  [警告] 加载 {filename} 失败: {e}")

    print(f"\n[统计] 共处理 {success_count} 个新文件, 总计 {len(all_chunks)} 个 chunks")

    # ---- 阶段3: 构建统一 FAISS 索引 ----
    if args.skip_index:
        print("\n[跳过] --skip-index 已指定，不构建 FAISS 索引")
    elif all_chunks:
        print(f"\n[构建索引] {len(all_chunks)} 个 chunks -> {VECTOR_STORE_DIR}")
        build_unified_index(all_chunks, str(VECTOR_STORE_DIR), args.embedding_model)
    else:
        print("\n[提示] 无可索引内容")

    # ---- 输出摘要 ----
    print(f"\n{'=' * 60}")
    print("[处理完成]")
    list_documents(conn)
    print(f"\n  索引目录: {VECTOR_STORE_DIR}")
    print(f"  元数据库: {DB_PATH}")
    print(f"  解析缓存: {PARSED_DIR}")
    print(f"{'=' * 60}")

    conn.close()


if __name__ == "__main__":
    main()
