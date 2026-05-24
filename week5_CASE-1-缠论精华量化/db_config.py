# -*- coding: utf-8 -*-
"""
配置文件 - 从 .env 文件读取数据库连接配置和回测参数

为什么用 .env 而不是直接写在代码里:
  1. 敏感信息（数据库密码）不应提交到版本控制系统
  2. 不同环境（开发/生产）使用不同的配置
  3. 回测参数可在不修改代码的情况下调整

使用方法:
  from db_config import DB_CONFIG, get_connection, execute_query

  conn = get_connection()
  rows = execute_query("SELECT * FROM table WHERE id = %s", [123])
"""
from pathlib import Path
import pymysql
from dotenv import dotenv_values

# --- 加载 .env 配置文件 ---
# Path(__file__).parent 定位到当前 py 文件所在目录, 然后拼接 .env 文件路径
# dotenv_values() 将 .env 中的 K=V 键值对读取为一个字典
_env_path = Path(__file__).parent / '.env'
_env = dotenv_values(_env_path)

# 数据库连接配置字典
# 使用 .env.get('KEY', '默认值') 方式: 如果 .env 中未定义该变量, 使用默认值兜底
# 这样即使 .env 文件不完整, 程序也能在开发环境中正常运行
DB_CONFIG = {
    'host': _env.get('WUCAI_SQL_HOST', 'localhost'),       # 数据库主机地址
    'user': _env.get('WUCAI_SQL_USERNAME', 'root'),        # 数据库用户名
    'password': _env.get('WUCAI_SQL_PASSWORD', ''),        # 数据库密码
    'database': _env.get('WUCAI_SQL_DB', 'wucai_trade'),   # 数据库名
    'port': int(_env.get('WUCAI_SQL_PORT', '3306')),       # 端口号 (注意转int)
    'charset': 'utf8mb4'  # 使用 utf8mb4 而非 utf8, 支持完整的 Unicode (如表情符号)
}

# --- 回测参数（可在 .env 中修改, 无需改动代码） ---
INITIAL_CASH = int(_env.get('BACKTEST_INITIAL_CASH', '1000000'))   # 初始资金, 默认 100 万
COMMISSION = float(_env.get('BACKTEST_COMMISSION', '0.0002'))      # 手续费率, 默认万2
POSITION_PCT = int(_env.get('BACKTEST_POSITION_PCT', '95'))        # 仓位百分比, 默认 95%


def get_connection():
    """
    获取数据库连接

    使用 DB_CONFIG 字典解包为关键字参数传递给 pymysql.connect(),
    每次调用创建一个新的连接。生产环境中建议使用连接池而非每次都创建新连接,
    但在教学/回测场景中, 连接频率不高, 简化处理即可。

    返回:
        pymysql.connection.Connection 对象
    """
    return pymysql.connect(**DB_CONFIG)


def execute_query(sql, params=None):
    """
    执行 SQL 查询并返回结果

    使用步骤:
      1. 获取数据库连接
      2. 创建 DictCursor (返回字典列表, 每行是一个字段名->值的字典)
      3. 执行 SQL (参数化查询防 SQL 注入)
      4. 获取所有结果
      5. 关闭游标和连接 (防止连接泄漏)

    参数:
        sql: SQL 查询语句, 使用 %s 作为占位符 (如 "SELECT * FROM table WHERE id = %s")
        params: 参数元组或列表, 与 sql 中的占位符一一对应

    返回:
        list[dict]: 查询结果, 每行一个字典, 字段名为 key

    示例:
        rows = execute_query(
            "SELECT * FROM trade_stock_daily WHERE stock_code = %s AND trade_date >= %s",
            ('600519.SH', '2024-01-01')
        )
    """
    conn = get_connection()
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute(sql, params or ())
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result
