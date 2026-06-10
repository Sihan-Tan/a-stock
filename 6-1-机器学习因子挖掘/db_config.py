# -*- coding: utf-8 -*-
"""
数据库配置文件 - 从 .env 环境变量文件读取数据库连接参数与回测参数。

为什么需要这个文件？
  在量化回测中，数据库连接信息（主机、端口、用户名、密码）属于敏感信息，
  不应硬编码在代码中。通过 .env 文件统一管理，既安全又便于在不同环境间切换。

依赖关系：
  - data_loader.py 通过本模块获取数据库连接，加载K线和财务数据
  - 回测参数（初始资金、手续费率、仓位比例）也在这里统一配置
"""
from pathlib import Path          # 跨平台路径处理，自动适配 Windows/Linux 路径分隔符
import pymysql                    # Python MySQL 驱动，用于连接数据库
from dotenv import dotenv_values  # 读取 .env 文件的轻量级库，比 python-dotenv 更简洁


# ============================================================
# 读取 .env 文件
# ============================================================
# Path(__file__) 获取当前文件所在目录，parent 是 week6 目录
# dotenv_values() 将 .env 文件解析为字典，返回 None 的键用默认值替代
_env_path = Path(__file__).parent / '.env'
_env = dotenv_values(_env_path)


# ============================================================
# 数据库连接配置
# ============================================================
# _env.get('KEY', 'default')：从 .env 取值，若未设置则使用默认值
# 这样做的好处是：即使没有 .env 文件，程序也能用默认参数运行开发环境
DB_CONFIG = {
    'host': _env.get('WUCAI_SQL_HOST', 'localhost'),       # 数据库主机地址，默认本地
    'user': _env.get('WUCAI_SQL_USERNAME', 'root'),         # 数据库用户名
    'password': _env.get('WUCAI_SQL_PASSWORD', ''),         # 数据库密码
    'database': _env.get('WUCAI_SQL_DB', 'wucai_trade'),    # 数据库名，存储A股日K线
    'port': int(_env.get('WUCAI_SQL_PORT', '3306')),        # MySQL 默认端口 3306
    'charset': 'utf8mb4'                                    # 使用 utf8mb4 支持完整 Unicode（含emoji）
}


# ============================================================
# 回测参数（可在 .env 中修改，无需改动代码）
# ============================================================
# INITIAL_CASH: 回测初始资金，默认 100 万（A股回测常用起始资金）
# COMMISSION: 手续费率，万2（0.02%），A股实际佣金通常在万1.5~万3之间
# POSITION_PCT: 仓位比例 95%，满仓操作但留 5% 现金应对冲击成本
INITIAL_CASH = int(_env.get('BACKTEST_INITIAL_CASH', '1000000'))
COMMISSION = float(_env.get('BACKTEST_COMMISSION', '0.0002'))
POSITION_PCT = int(_env.get('BACKTEST_POSITION_PCT', '95'))


def get_connection():
    """
    获取数据库连接。

    为什么封装成函数而不是全局变量？
      每次调用都创建新连接，避免连接长时间闲置被数据库服务器断开。
      对短查询场景（如量化数据加载）更安全。

    返回:
        pymysql.connections.Connection 对象
    """
    return pymysql.connect(**DB_CONFIG)


def execute_query(sql, params=None):
    """
    执行 SQL 查询并以字典列表形式返回结果。

    参数:
        sql: SQL 查询语句，支持 %s 占位符（pymysql 的参数化查询方式）
        params: 参数元组或列表，用于安全填充 SQL 中的 %s 占位符

    为什么使用参数化查询？
      1. 防止 SQL 注入攻击
      2. pymysql 会自动处理参数的类型转换和转义

    返回:
        list[dict]：每行数据为一个字典，键为列名，值为列值

    使用示例:
        execute_query("SELECT * FROM table WHERE id = %s", [123])
    """
    conn = get_connection()
    # DictCursor 让查询结果以字典形式返回，比默认的元组更易读
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    cursor.execute(sql, params or ())
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result
