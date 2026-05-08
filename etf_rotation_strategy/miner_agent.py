#!/usr/bin/env python3
"""
因子挖掘主循环 (LLM Factor Miner Agent)

多轮迭代: Generation -> Evaluation -> Reflection -> Generation
让大模型自主生成因子公式字符串，经由 factor_engine 测试后，
将诊断结果反馈给大模型进行反思改进。
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from etf_rotation_strategy.data_loader import load_sample_data
from etf_rotation_strategy.factor_engine import (
    parse_and_calculate,
    evaluate_factor,
    FactorRegistry,
    IS_START,
    IS_END,
)
from etf_rotation_strategy.llm_client import generate_response, parse_json_response

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

TARGET_IC = 0.02
TARGET_ICIR = 0.5


# ===================== System Prompt =====================

SYSTEM_PROMPT = """你是一个顶尖的量化因子挖掘AI，专精于美股行业ETF截面轮动策略。

【你的任务】
利用提供的算子库，组合基础价量特征，生成具有截面区分度的因子公式。
每个公式必须是一个合法的 Python 表达式字符串，最终输出一个 (T, 11) 的 DataFrame。

【可用算子库】
算术类 (输入 x, y 均为 DataFrame):
  - add(x, y):     加法
  - sub(x, y):     减法
  - mul(x, y):     乘法
  - div(x, y):     除法 (自动处理除零)
  - log(x):        自然对数
  - sign(x):       符号函数
  - abs_val(x):    绝对值

时序类 (d 为回溯天数):
  - delay(x, d):         滞后 d 期
  - delta(x, d):         x - delay(x, d)
  - ts_returns(x, d):    d 期收益率
  - ts_mean(x, d):       d 期滚动均值
  - ts_max(x, d):        d 期滚动最大值
  - ts_min(x, d):        d 期滚动最小值
  - ts_std(x, d):        d 期滚动标准差
  - ts_rank(x, d):       过去 d 天的时序百分位排名

截面类:
  - cs_rank(x):    每日横截面百分位排名 (0~1)
  - cs_zscore(x):  每日横截面标准化

交互类:
  - correlation(x, y, d): x 与 y 的 d 日滚动相关系数

【数据输入】
可用基础特征: open, high, low, close, volume
每个特征均为 (T, 11) 的 DataFrame，列为 11 只 SPDR 行业 ETF。

【关键约束】
1. 公式最终必须使用 cs_rank() 或 cs_zscore() 进行截面标准化。
2. 严禁使用任何未来函数 (如 shift(-1))，所有时序算子仅回溯历史。
3. 回溯天数 d 建议取 5, 10, 20, 60, 120 等标准周期。
4. 优先寻找逻辑清晰、经济含义明确的因子，避免过度数据挖掘。
5. 部分ETF早期含有NaN (如XLRE 2015年上市、XLC 2018年上市)，算子已内置NaN处理。

【输出格式】
你的回复必须是一个纯 JSON 格式，包含一个公式列表。例如:
{"formulas": ["cs_rank(ts_mean(close, 20))", "cs_rank(correlation(close, volume, 10))"]}

每次生成 formulas_per_round 个不同的公式。确保公式之间逻辑正交、风格多样。"""


# ===================== 主循环 =====================

def run_mining_loop(
    iterations: int = 5,
    formulas_per_round: int = 3,
    forward_period: int = 5,
) -> None:
    """
    因子挖掘主循环

    Args:
        iterations:         迭代轮数
        formulas_per_round: 每轮生成公式数
        forward_period:     前瞻收益率天数
    """
    logger.info("=" * 60)
    logger.info("ETF 轮动策略 - LLM 因子挖掘机启动")
    logger.info(f"  迭代轮数: {iterations}")
    logger.info(f"  每轮公式数: {formulas_per_round}")
    logger.info(f"  IS 评估区间: {IS_START} ~ {IS_END}")
    logger.info("=" * 60)

    logger.info("[初始化] 加载数据...")
    data_dict = load_sample_data()
    close_df = data_dict['close']
    logger.info(f"  数据维度: {close_df.shape}, 日期范围: {close_df.index[0].date()} ~ {close_df.index[-1].date()}")

    forward_returns = close_df.shift(-forward_period) / close_df - 1

    registry = FactorRegistry()
    logger.info(f"  已有注册因子: {registry.count}")

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    total_generated = 0
    total_accepted = 0

    for iteration in range(1, iterations + 1):
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"第 {iteration}/{iterations} 轮迭代")
        logger.info("=" * 60)

        user_prompt = f"请生成 {formulas_per_round} 个不同的因子公式。注意逻辑多样性和正交性。"
        messages.append({"role": "user", "content": user_prompt})

        logger.info(f"[生成] 调用 LLM 生成 {formulas_per_round} 个公式...")
        try:
            raw_response = generate_response(messages)
        except RuntimeError as e:
            logger.error(f"[生成] LLM 调用失败: {e}")
            messages.pop()
            continue

        logger.info(f"[生成] LLM 原始回复:\n{raw_response[:500]}...")

        try:
            parsed = parse_json_response(raw_response)
        except ValueError as e:
            logger.error(f"[解析] JSON 解析失败: {e}")
            feedback = f"你的回复无法被解析为 JSON。错误: {e}。请严格按要求输出 JSON 格式。"
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "user", "content": feedback})
            continue

        formulas = parsed.get('formulas', [])
        if not formulas or not isinstance(formulas, list):
            logger.warning(f"[解析] 未找到 formulas 列表: {parsed}")
            feedback = "你的 JSON 中没有有效的 formulas 列表。请确保格式为 {\"formulas\": [\"公式1\", \"公式2\"]}"
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "user", "content": feedback})
            continue

        logger.info(f"[解析] 提取到 {len(formulas)} 个公式")

        messages.append({"role": "assistant", "content": raw_response})

        reflection_parts: List[str] = []

        for i, formula in enumerate(formulas, 1):
            total_generated += 1
            formula = formula.strip()
            logger.info(f"")
            logger.info(f"--- 公式 {i}/{len(formulas)}: {formula} ---")

            try:
                factor_df = parse_and_calculate(formula, data_dict)
                logger.info(f"  计算成功, shape={factor_df.shape}, "
                            f"有效值占比={factor_df.notna().mean().mean():.2%}")
            except ValueError as e:
                error_msg = str(e)
                logger.warning(f"  计算失败: {error_msg}")
                reflection_parts.append(
                    f"公式 '{formula}' 报错: {error_msg}。"
                    f"请严格使用提供的算子库，检查算子名称拼写和参数格式。"
                )
                continue

            metrics = evaluate_factor(factor_df, close_df, forward_returns)

            rank_ic = metrics['RankIC']
            icir = metrics['ICIR']
            turnover = metrics['Turnover']

            logger.info(f"  RankIC={rank_ic:+.6f}, ICIR={icir:+.4f}, Turnover={turnover:.6f}")
            logger.info(f"[DEBUG] 当前 RankIC 入库阈值已临时调低至 {TARGET_IC} 以测试存储流。")

            is_correlated, corr_with, corr_value = registry.check_correlation(factor_df)

            accepted = False
            reject_reason = ""

            if np.isnan(rank_ic):
                reject_reason = "RankIC 为 NaN (可能是截面有效样本不足)"
            elif abs(rank_ic) < TARGET_IC:
                reject_reason = (
                    f"IC 绝对值 {abs(rank_ic):.4f} 低于 {TARGET_IC} 阈值，信号太弱。"
                    f"建议: 尝试更长周期的平滑、引入成交量确认、或寻找全新逻辑维度。"
                )
            elif abs(icir) < TARGET_ICIR:
                reject_reason = (
                    f"ICIR 绝对值 {abs(icir):.4f} 低于 {TARGET_ICIR} 阈值，信号不稳定。"
                    f"建议: 增加滚动窗口长度以降低噪音，或结合多个互补指标。"
                )
            elif is_correlated:
                reject_reason = (
                    f"与已有因子 {corr_with} 截面相关性 {corr_value:.4f} 过高 (>0.6)，"
                    f"缺乏正交增量。建议: 基于该公式取残差、或寻找完全不同的逻辑维度。"
                )
            else:
                accepted = True

            if accepted:
                success = registry.register(
                    formula=formula, metrics=metrics, factor_df=factor_df,
                    mining_round=iteration,
                )
                if success:
                    total_accepted += 1
                    reflection_parts.append(
                        f"公式 '{formula}' 测试结果: RankIC={rank_ic:+.4f}, ICIR={icir:+.4f} (达标，已入库)。"
                        f"该测试结果基于 {IS_START}~{IS_END} 年 In-Sample 数据区间。"
                    )
                    logger.info(f"★★★ 因子入库成功: {formula} ★★★")
                else:
                    reflection_parts.append(
                        f"公式 '{formula}' 测试结果: RankIC={rank_ic:+.4f}, ICIR={icir:+.4f} "
                        f"(指标达标但入库失败，可能因相关性)。"
                        f"该测试结果基于 {IS_START}~{IS_END} 年 In-Sample 数据区间。"
                    )
            else:
                reflection_parts.append(
                    f"公式 '{formula}' 测试结果: RankIC={rank_ic:+.4f}, ICIR={icir:+.4f}, "
                    f"Turnover={turnover:.4f} (不达标)。拒绝原因: {reject_reason} "
                    f"该测试结果基于 {IS_START}~{IS_END} 年 In-Sample 数据区间。"
                )

        feedback_text = "\n\n".join(reflection_parts)
        feedback_message = (
            f"【第 {iteration} 轮评估反馈】\n\n"
            f"{feedback_text}\n\n"
            f"请根据以上反馈反思并改进，在下一轮生成更好的因子公式。"
            f"记住: 目标是 RankIC 绝对值 >= {TARGET_IC} 且 ICIR 绝对值 >= {TARGET_ICIR}，同时与已有因子保持正交。"
        )

        logger.info(f"\n[反思] 本轮反馈摘要:")
        for part in reflection_parts:
            logger.info(f"  - {part[:100]}...")

        messages.append({"role": "user", "content": feedback_message})

    logger.info("")
    logger.info("=" * 60)
    logger.info("因子挖掘完成!")
    logger.info(f"  总生成公式数: {total_generated}")
    logger.info(f"  总入库因子数: {total_accepted}")
    logger.info(f"  数据库路径: {registry.db_path}")
    logger.info("=" * 60)


if __name__ == '__main__':
    run_mining_loop(iterations=5, formulas_per_round=3)
