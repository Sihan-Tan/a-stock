# -*- coding: utf-8 -*-
"""
配置文件 - 从 .env 读取数据库连接和回测参数

本模块是整个项目的数据层基础配置，负责：
  1. 从 .env 文件读取 MySQL 数据库连接参数（主机、端口、用户名、密码、数据库名）
  2. 读取回测引擎的全局参数（初始资金、手续费率、仓位比例）
  3. 提供数据库连接和查询的工具函数

为什么用 .env 而不是硬编码？
  数据库密码等敏感信息不应提交到 git 仓库，.env 文件已在 .gitignore 中排除。
  dotenv_values() 是纯读取函数，不会修改系统环境变量，比 load_dotenv() 更安全。

使用方式:
  from db_config import DB_CONFIG, get_connection, execute_query
"""
from pathlib import Path      # 跨平台路径处理，比 os.path 更现代
import pymysql                # MySQL 数据库驱动
from dotenv import dotenv_values  # 纯读取 .env 文件（不污染环境变量）


# ============================================================
# 读取 .env 配置文件
# ============================================================
# Path(__file__).parent 获取当前脚本所在目录，保证无论从哪个目录执行都能找到 .env
_env_path = Path(__file__).parent / '.env'
_env = dotenv_values(_env_path)  # 返回 dict，key 不存在时返回 None

# ============================================================
# 数据库连接配置
# ============================================================
# 每个字段都有默认值，即使 .env 缺失也能使用默认配置连接到本地数据库
DB_CONFIG = {
    'host': _env.get('WUCAI_SQL_HOST', 'localhost'),       # 数据库主机地址
    'user': _env.get('WUCAI_SQL_USERNAME', 'root'),        # 数据库用户名
    'password': _env.get('WUCAI_SQL_PASSWORD', ''),        # 数据库密码
    'database': _env.get('WUCAI_SQL_DB', 'wucai_trade'),   # 数据库名，存储A股日K线数据
    'port': int(_env.get('WUCAI_SQL_PORT', '3306')),       # MySQL 端口号，默认 3306
    'charset': 'utf8mb4'    # 使用 utf8mb4 而非 utf8，支持存储 emoji 和生僻汉字
}

# ============================================================
# 回测引擎全局参数
# ============================================================
# 这些参数会被 data_loader.py 中的 setup_cerebro() 使用
INITIAL_CASH = int(_env.get('BACKTEST_INITIAL_CASH', '1000000'))   # 初始资金，默认100万
COMMISSION = float(_env.get('BACKTEST_COMMISSION', '0.0002'))      # 手续费率，默认万分之二
POSITION_PCT = int(_env.get('BACKTEST_POSITION_PCT', '95'))       # 单次开仓仓位比例，默认95%

# DeepSeek API 配置（用于AI二次筛选候选池）
DEEPSEEK_API_KEY = _env.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_BASE_URL = _env.get('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')
DEEPSEEK_MODEL = _env.get('DEEPSEEK_MODEL', 'deepseek-chat')


def get_connection():
    """
    获取 MySQL 数据库连接

    pymysql.connect() 返回的连接对象支持上下文管理器（with 语句），
    但本模块统一使用 execute_query 进行查询，不直接暴露连接对象。

    返回:
        pymysql.Connection 数据库连接对象
    """
    return pymysql.connect(**DB_CONFIG)  # ** 解包字典为关键字参数


def execute_query(sql, params=None):
    """
    执行 SQL 查询，以字典列表形式返回结果

    为什么使用 DictCursor？
      DictCursor 让每行结果以字典形式返回（column_name -> value），
      比默认的元组形式更易读，可直接用列名访问。

    参数:
        sql: SQL 查询语句，支持 %s 占位符
        params: 参数元组或列表，用于防止 SQL 注入（切勿用 f-string 拼接 SQL！）

    返回:
        list[dict] 查询结果，每行为一个字典，字段名 -> 值

    注意:
        本函数仅适用于查询（SELECT），不适用于 INSERT/UPDATE/DELETE。
        execute() 会自动处理参数转义，防止 SQL 注入攻击。
    """
    conn = get_connection()                          # 获取新连接
    cursor = conn.cursor(pymysql.cursors.DictCursor)  # 使用字典游标，结果以列名索引
    cursor.execute(sql, params or ())                 # 执行 SQL，params 防止 SQL 注入
    result = cursor.fetchall()                        # 获取所有结果行
    cursor.close()                                    # 关闭游标
    conn.close()                                      # 关闭连接
    return result
