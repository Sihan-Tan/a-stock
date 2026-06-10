# -*- coding: utf-8 -*-
"""
LLM 报告引擎 -- 调用大模型生成分析结论

功能:
  1. generate_metrics_summary()  - 根据QuantStats指标生成绩效总结
  2. generate_trade_summary()    - 根据交易记录生成交易分析
  3. generate_svd_summary()      - 根据SVD诊断结果生成市场状态分析
  4. generate_report_conclusion() - 综合所有分析生成报告结论

使用:
  需要设置环境变量 DASHSCOPE_API_KEY
  模型: qwen-max (通义千问)
"""
import os
import json


def _call_qwen(prompt, system_prompt=None, max_tokens=1500):
    """
    调用通义千问 qwen-max 生成文本

    参数:
        prompt: str, 用户提示
        system_prompt: str, 系统提示 (可选)
        max_tokens: int, 最大生成长度

    返回:
        str, 生成的文本; 失败时返回 None
    """
    api_key = os.environ.get('DASHSCOPE_API_KEY', '')
    if not api_key:
        print('  [LLM] DASHSCOPE_API_KEY 未设置, 跳过LLM分析')
        return None

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=api_key,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})

        response = client.chat.completions.create(
            model='qwen-max',
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.3,
        )

        content = response.choices[0].message.content
        return content.strip()

    except ImportError:
        print('  [LLM] openai 库未安装, 请运行: pip install openai')
        return None
    except Exception as e:
        print(f'  [LLM] 调用失败: {e}')
        return None


SYSTEM_PROMPT = """你是一位专业的量化投资分析师, 擅长解读投资绩效数据。
请用简洁、专业的中文撰写分析结论。
要求:
- 数据驱动, 引用具体数值
- 给出明确的判断(好/差/一般)
- 提出可操作的建议
- 不要使用 emoji
- 控制在200字以内"""


def generate_metrics_summary(metrics, strategy_name='策略'):
    """
    根据 QuantStats 指标生成绩效总结

    参数:
        metrics: dict, calc_quantstats_metrics() 返回的指标字典
        strategy_name: str, 策略名称

    返回:
        str, 分析结论文本
    """
    if not metrics:
        return None

    prompt = f"""请分析以下量化策略的绩效指标, 给出专业的总结和建议:

策略名称: {strategy_name}
总收益率: {metrics.get('total_return', 0):.2%}
年化收益率(CAGR): {metrics.get('cagr', 0):.2%}
年化波动率: {metrics.get('volatility', 0):.2%}
最大回撤: {metrics.get('max_drawdown', 0):.2%}
夏普比率: {metrics.get('sharpe', 0):.4f}
索提诺比率: {metrics.get('sortino', 0):.4f}
卡玛比率: {metrics.get('calmar', 0):.4f}
日胜率: {metrics.get('win_rate', 0):.2%}
盈亏比: {metrics.get('gain_to_pain', 0):.4f}
VaR(95%): {metrics.get('var_95', 0):.2%}

请从以下几个维度分析:
1. 收益表现 (与年化10%的基准比较)
2. 风险控制 (回撤、波动率)
3. 风险调整后收益 (夏普、索提诺)
4. 综合评价和改进建议"""

    print(f'  [LLM] 正在生成 {strategy_name} 绩效分析...')
    result = _call_qwen(prompt, system_prompt=SYSTEM_PROMPT)
    if result:
        print(f'  [LLM] 绩效分析已生成 ({len(result)} 字)')
    return result


def generate_comparison_summary(all_metrics):
    """
    根据多策略指标对比生成比较分析

    参数:
        all_metrics: dict, {策略名: metrics_dict}

    返回:
        str, 对比分析文本
    """
    if not all_metrics or len(all_metrics) < 2:
        return None

    comparison_text = '以下是多个策略的绩效指标对比:\n\n'
    comparison_text += f'{"策略":<10s} {"总收益":>10s} {"年化":>10s} {"最大回撤":>10s} '
    comparison_text += f'{"夏普":>8s} {"索提诺":>8s} {"日胜率":>8s}\n'

    for name, m in all_metrics.items():
        comparison_text += (f'{name:<10s} {m.get("total_return",0):>9.2%} '
                           f'{m.get("cagr",0):>9.2%} {m.get("max_drawdown",0):>9.2%} '
                           f'{m.get("sharpe",0):>8.4f} {m.get("sortino",0):>8.4f} '
                           f'{m.get("win_rate",0):>7.2%}\n')

    prompt = f"""{comparison_text}
请分析:
1. 哪个策略综合表现最优? 为什么?
2. 各策略的优劣势分别是什么?
3. 在不同市场环境下, 推荐使用哪个策略?
4. 是否有策略组合的可能性?"""

    print(f'  [LLM] 正在生成多策略对比分析...')
    result = _call_qwen(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=2000)
    if result:
        print(f'  [LLM] 对比分析已生成 ({len(result)} 字)')
    return result


def generate_trade_summary(stock_df, cost_info, nav_info=None):
    """
    根据交易记录生成实盘交易分析

    参数:
        stock_df: DataFrame, 个股盈亏分析结果
        cost_info: dict, 成本分析结果
        nav_info: dict, 净值相关信息 (可选)
    """
    stocks_text = ''
    for _, row in stock_df.iterrows():
        pnl = f'{row["已实现盈亏"]:+,.0f}' if row.get('已实现盈亏') is not None else '持仓中'
        stocks_text += f'  {row["证券名称"]}: 买入{row["买入金额"]:,.0f}元, '
        stocks_text += f'卖出{row["卖出金额"]:,.0f}元, 盈亏:{pnl}\n'

    prompt = f"""请分析以下实盘交易记录:

个股交易明细:
{stocks_text}
交易成本:
  总成交额: {cost_info['total_turnover']:,.0f}元
  总成本: {cost_info['total_cost']:,.0f}元 (费率: {cost_info['cost_ratio']:.3f}%)"""

    if nav_info:
        prompt += f"""

组合绩效:
  总收益: {nav_info.get('total_return', 'N/A')}
  夏普比率: {nav_info.get('sharpe', 'N/A')}
  最大回撤: {nav_info.get('max_drawdown', 'N/A')}"""

    prompt += """

请从以下角度分析:
1. 选股特点和盈亏分布
2. 交易成本是否合理
3. 持仓策略的优劣
4. 改进建议"""

    print(f'  [LLM] 正在生成实盘交易分析...')
    result = _call_qwen(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=2000)
    if result:
        print(f'  [LLM] 交易分析已生成 ({len(result)} 字)')
    return result


def generate_svd_summary(svd_result):
    """根据SVD诊断结果生成市场状态分析"""
    if not svd_result:
        return None

    prompt = f"""请分析以下SVD市场状态诊断结果:

分析维度: {svd_result['stock_count']} 只股票, {svd_result['data_days']} 个交易日
当前市场状态: {svd_result['current_state']}
第一因子方差占比: {svd_result['current_f1_ratio']:.1%}

(Factor1方差占比 >50% 表示齐涨齐跌/beta主导,
 35%-50% 表示板块分化, <35% 表示个股行情/alpha机会多)

请分析:
1. 当前市场状态对投资策略的影响
2. 在这种市场环境下的最优策略选择
3. 需要注意的风险点"""

    print(f'  [LLM] 正在生成SVD市场分析...')
    result = _call_qwen(prompt, system_prompt=SYSTEM_PROMPT)
    if result:
        print(f'  [LLM] SVD分析已生成 ({len(result)} 字)')
    return result


def summary_to_html(summary_text, title='分析结论'):
    """将LLM生成的文本转为HTML片段"""
    if not summary_text:
        return ''

    # 简单的 Markdown 转 HTML
    lines = summary_text.split('\n')
    html_lines = []
    in_list = False

    for line in lines:
        line = line.strip()
        if not line:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append('<br>')
            continue

        # 标题
        if line.startswith('###'):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h4>{line.lstrip("#").strip()}</h4>')
        elif line.startswith('##'):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h3>{line.lstrip("#").strip()}</h3>')
        elif line.startswith('#'):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h3>{line.lstrip("#").strip()}</h3>')
        # 列表
        elif line.startswith(('-', '*')) and len(line) > 2:
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{line[1:].strip()}</li>')
        elif len(line) > 2 and line[0].isdigit() and line[1] in '.、':
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{line[2:].strip()}</li>')
        elif len(line) > 3 and line[:2].isdigit() and line[2] in '.、':
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{line[3:].strip()}</li>')
        else:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            # 加粗处理
            import re
            line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', line)
            html_lines.append(f'<p>{line}</p>')

    if in_list:
        html_lines.append('</ul>')

    content = '\n'.join(html_lines)

    return f'''
    <div style="background:linear-gradient(135deg,#f8f9fa,#e9ecef);
                border-left:4px solid #2980b9;border-radius:8px;
                padding:20px;margin:16px 0;">
      <h3 style="color:#2980b9;margin-top:0;margin-bottom:12px;">
        {title}
      </h3>
      <div style="font-size:14px;line-height:1.8;color:#2c3e50;">
        {content}
      </div>
      <p style="font-size:11px;color:#95a5a6;margin-top:12px;margin-bottom:0;">
        * 以上分析由 AI 大模型(qwen-max)基于数据自动生成, 仅供参考
      </p>
    </div>'''
