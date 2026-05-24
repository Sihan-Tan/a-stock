# -*- coding: utf-8 -*-
"""
数据库连接配置模块。

本模块集中管理所有数据库和API的配置信息，位于 week2_get_data 目录下，
供该目录下所有采集脚本共享使用。

设计理念：
  1. 配置集中管理：所有数据库连接参数和API密钥集中于此，避免散落在各脚本中
  2. 环境变量驱动：敏感信息通过 .env 文件加载，不硬编码在代码中
  3. 统一数据库操作：提供 get_connection / execute_query / execute_update / execute_many 四个通用函数

环境变量文件 (.env) 配置示例：
  WUCAI_SQL_HOST=localhost
  WUCAI_SQL_USERNAME=root
  WUCAI_SQL_PASSWORD=your_password
  WUCAI_SQL_DB=stock
  WUCAI_SQL_PORT=3306
  DASHSCOPE_API_KEY=sk-xxx
  KIMI_API_KEY=sk-xxx
"""
import os
from pathlib import Path
import pymysql
from dotenv import dotenv_values

# ============================================================
# 数据库配置
# ============================================================
# 从 .env 文件读取配置（.env 文件位于本文件同级目录下）
_env_path = Path(__file__).parent / '.env'
_env = dotenv_values(_env_path)

# MySQL 数据库连接参数
# 这些参数通过 dotenv_values 从 .env 文件读取，如果 .env 中未设置则使用默认值
DB_CONFIG = {
    'host': _env.get('WUCAI_SQL_HOST', 'localhost'),        # 数据库主机地址
    'user': _env.get('WUCAI_SQL_USERNAME', 'root'),          # 数据库用户名
    'password': _env.get('WUCAI_SQL_PASSWORD', 'password'),  # 数据库密码
    'database': _env.get('WUCAI_SQL_DB', 'stock'),           # 数据库名
    'port': int(_env.get('WUCAI_SQL_PORT', '3306')),         # 端口，默认3306
    'charset': 'utf8mb4'                                     # 字符集（支持emoji等4字节UTF-8字符）
}

# ============================================================
# AI API 配置
# ============================================================

# Kimi (月之暗面) API 配置，主要用于研报分析和文本理解任务
KIMI_API_KEY = _env.get('KIMI_API_KEY', '')
KIMI_BASE_URL = _env.get('KIMI_BASE_URL', 'https://api.moonshot.cn/v1')
KIMI_MODEL = _env.get('KIMI_MODEL', 'kimi-latest')

# DashScope / Qwen (阿里通义千问) API 配置，主要用于联网搜索和催化剂事件采集
# DASHSCOPE_BASE_URL 使用兼容模式地址，兼容 OpenAI SDK 的调用格式
DASHSCOPE_API_KEY = _env.get('DASHSCOPE_API_KEY', '')
DASHSCOPE_BASE_URL = _env.get('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
QWEN_MODEL = _env.get('QWEN_MODEL', 'qwen-flash')


# ============================================================
# 数据库工具函数
# ============================================================

def get_connection():
    """
    获取数据库连接。

    每次调用创建一个新的 pymysql 连接。
    注意：调用方需要在使用完毕后关闭连接，否则会造成连接泄漏。
    推荐的使用模式：
      conn = get_connection()
      try:
          cursor = conn.cursor()
          # ... 执行数据库操作 ...
          conn.commit()
      finally:
          cursor.close()
          conn.close()

    Returns:
        pymysql.Connection: MySQL数据库连接对象
    """
    return pymysql.connect(**DB_CONFIG)


def execute_query(sql, params=None):
    """
    执行查询SQL，返回字典列表。

    这是最常用的查询函数，将查询结果以字典列表的形式返回，
    每条记录是一个字典，键为列名，值为列值。

    Args:
        sql: SQL查询语句（支持 %s 占位符）
        params: 参数元组或列表，用于预防SQL注入

    Returns:
        list[dict]: 查询结果，每行一个字典

    示例：
      execute_query("SELECT * FROM stocks WHERE code = %s", ("600519",))
      -> [{'code': '600519', 'name': '贵州茅台', ...}]
    """
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute(sql, params or ())
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result


def execute_update(sql, params=None):
    """
    执行单条更新/插入SQL。

    适用于 INSERT、UPDATE、DELETE 等非查询操作。
    返回受影响的行数，可以用来判断操作是否成功。

    Args:
        sql: SQL语句
        params: 参数元组或列表

    Returns:
        int: 受影响的行数

    示例：
      affected = execute_update(
          "UPDATE stocks SET name = %s WHERE code = %s",
          ("贵州茅台", "600519")
      )
      print(f"更新了 {affected} 行")
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(sql, params or ())
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    return affected


def execute_many(sql, data_list):
    """
    批量执行插入/更新。

    使用 executemany 批量执行相同的SQL，比逐条执行效率高得多。
    适用于批量数据导入场景。

    Args:
        sql: SQL语句模板（使用 %s 作为占位符）
        data_list: 数据列表，每个元素是一条记录的参数元组

    Returns:
        int: 受影响的行数

    示例：
      data = [("600519", 100), ("000858", 200)]
      execute_many("INSERT INTO test VALUES (%s, %s)", data)
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.executemany(sql, data_list)
    conn.commit()
    affected = cursor.rowcount
    cursor.close()
    conn.close()
    return affected
