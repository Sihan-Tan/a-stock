# -*- coding: utf-8 -*-
"""
数据库配置模块 - 从 .env 文件读取数据库连接信息和回测参数

本模块是整个回测系统的数据基础层:
  - 统一管理数据库连接配置，避免在多个策略文件中重复写连接信息
  - 所有数据库配置和回测参数均来自 .env 文件，便于环境切换而不改代码
  - 提供基础的数据库查询工具函数

使用方式:
  from db_config import DB_CONFIG, get_connection, execute_query
"""
from pathlib import Path          # 跨平台路径处理，自动适配 Windows/Linux 路径分隔符
import pymysql                    # MySQL 数据库驱动 (纯 Python 实现，无需安装 C 扩展)
from dotenv import dotenv_values  # 解析 .env 文件的轻量级工具，比 python-dotenv 的 load_dotenv 更可控


# ============================================================
# .env 文件定位：自动寻找与本文件同目录下的 .env 文件
# ============================================================
# Path(__file__) 获取当前文件路径，.parent 取所在目录
# 这样无论从哪个目录启动脚本，都能找到正确的 .env 文件
_env_path = Path(__file__).parent / '.env'

# dotenv_values() 将 .env 文件解析为字典，不会污染 os.environ
# 这样做的好处是：只读取我们需要的变量，不会意外覆盖系统环境变量
_env = dotenv_values(_env_path)


# ============================================================
# 数据库连接配置
# ============================================================
# 每个配置项都提供了默认值，即使 .env 文件缺失或字段不完整，系统也能以默认值运行
# 默认连接本地的 wucai_trade 数据库，这是数据采集模块写入的目标库
DB_CONFIG = {
    # 数据库主机地址：本地开发用 localhost，生产环境改为远程 IP
    'host': _env.get('WUCAI_SQL_HOST', 'localhost'),
    # 数据库用户名：生产环境建议使用专用账户而非 root
    'user': _env.get('WUCAI_SQL_USERNAME', 'root'),
    # 数据库密码：默认为空，生产环境必须设置强密码
    'password': _env.get('WUCAI_SQL_PASSWORD', ''),
    # 数据库名：wucai_trade 是数据采集模块 (CASE-股票数据) 写入的目标库
    'database': _env.get('WUCAI_SQL_DB', 'wucai_trade'),
    # 端口号：MySQL 默认端口 3306
    'port': int(_env.get('WUCAI_SQL_PORT', '3306')),
    # 字符集：utf8mb4 支持完整的 Unicode（包括 emoji 和生僻汉字）
    'charset': 'utf8mb4'
}


# ============================================================
# 回测参数（可在 .env 中修改，无需修改代码）
# ============================================================
# 为什么把回测参数放 .env 而不是 hardcode？
#   方便在不改代码的情况下进行参数敏感性分析、团队协作时各自使用不同参数

# 初始资金：默认 100 万元，适合 A 股实盘模拟
INITIAL_CASH = int(_env.get('BACKTEST_INITIAL_CASH', '1000000'))

# 手续费率：默认万二 (0.02%)，包含券商佣金和规费
# A 股实际佣金约为万1.5~万3，这里取中间值万二
# 注意：印花税（卖出时千分之一）不在该费率中，需要 Backtrader 的 commission 模式支持
COMMISSION = float(_env.get('BACKTEST_COMMISSION', '0.0002'))

# 默认仓位比例：95%（预留 5% 现金应对滑点和意外情况）
# 主要用于固定仓位策略（如 SimpleTurtle），海龟策略使用 ATR 动态仓位，不依赖此参数
POSITION_PCT = int(_env.get('BACKTEST_POSITION_PCT', '95'))


def get_connection():
    """
    获取数据库连接

    每次调用都会创建一个新的连接，调用方负责在使用完毕后关闭连接。
    对于回测场景（查询次数有限），这种"短连接"模式简单可靠；
    如果需要大量高频查询，建议改用连接池（如 pymysql.pooling 或 SQLAlchemy 连接池）。

    返回:
        pymysql.connections.Connection 对象
    """
    return pymysql.connect(**DB_CONFIG)


def execute_query(sql, params=None):
    """
    执行 SQL 查询并返回结果（字典列表格式）

    这个函数封装了"连接-查询-关闭"的完整生命周期，调用方无需手动管理连接。
    每次查询都独立创建和销毁连接，避免长连接导致的连接泄漏问题。

    参数:
        sql: SQL 查询语句，支持 %s 占位符
        params: 参数元组或列表，用于填充 SQL 中的占位符
                使用参数化查询可以防止 SQL 注入攻击

    返回:
        list[dict] 查询结果列表，每行数据为一个字典（字段名: 值）
        如果查询结果为空，返回空列表 []

    使用示例:
        execute_query("SELECT * FROM trade_stock_daily WHERE stock_code = %s AND trade_date >= %s",
                      ('600519.SH', '2024-01-01'))
    """
    # 创建数据库连接
    conn = get_connection()

    # 使用 DictCursor：查询结果以字典形式返回，key 为字段名，value 为字段值
    # 相比默认的元组格式，字典格式更具可读性，可以直接通过字段名访问数据
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # 执行 SQL 查询
    # params or () 确保即使 params 为 None 也能正常执行
    cursor.execute(sql, params or ())

    # 获取所有查询结果
    result = cursor.fetchall()

    # 关闭游标和连接，释放数据库资源
    # 如果不关闭连接，会导致数据库连接数耗尽
    cursor.close()
    conn.close()

    return result
