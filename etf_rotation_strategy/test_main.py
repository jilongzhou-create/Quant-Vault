#!/usr/bin/env python3
"""
独立测试脚本 - 验证因子层闭环运行

流程:
  1. 从数据库加载行业 ETF 数据 (2017年起，含 warm-up)
  2. 计算未来5日收益率 (forward_returns)
  3. 解析计算两个因子公式
  4. 评测因子质量 (仅 IS 区间 2018-2022)
  5. 存入 FactorRegistry (SQLite)
  6. 测试相关性去重
"""

import sys
import os
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import numpy as np

from etf_rotation_strategy.data_loader import load_sample_data
from etf_rotation_strategy.factor_engine import (
    parse_and_calculate,
    evaluate_factor,
    FactorRegistry,
    IS_START,
    IS_END,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def main() -> None:
    print("=" * 60)
    print("ETF 轮动策略 - 因子层闭环测试")
    print(f"  IS 评估区间: {IS_START} ~ {IS_END}")
    print("=" * 60)

    # ---- 1. 加载数据 ----
    print("\n[步骤1] 加载数据 (2017年起，含 warm-up)...")
    data_dict = load_sample_data()
    close_df = data_dict['close']
    print(f"  数据维度: {close_df.shape}")
    print(f"  日期范围: {close_df.index[0].date()} ~ {close_df.index[-1].date()}")
    print(f"  资产池:   {list(close_df.columns)}")

    # ---- 2. 计算 forward_returns ----
    print("\n[步骤2] 计算未来5日收益率 (forward_returns)...")
    forward_period = 5
    forward_returns = close_df.shift(-forward_period) / close_df - 1
    print(f"  forward_returns 维度: {forward_returns.shape}")
    print(f"  有效样本数: {forward_returns.notna().any(axis=1).sum()}")

    # ---- 3. 解析计算因子 ----
    formula_1 = "cs_rank(ts_mean(close, 20))"
    formula_2 = "cs_rank(correlation(close, volume, 10))"

    formulas = [formula_1, formula_2]
    factor_results = {}

    for formula in formulas:
        print(f"\n[步骤3] 解析因子: {formula}")
        try:
            factor_df = parse_and_calculate(formula, data_dict)
            factor_results[formula] = factor_df
            print(f"  计算成功! 维度: {factor_df.shape}")
            print(f"  有效值占比: {factor_df.notna().mean().mean():.2%}")
            print(f"  均值: {factor_df.mean().mean():.6f}")
            print(f"  标准差: {factor_df.std().mean():.6f}")
        except ValueError as e:
            print(f"  计算失败: {e}")

    # ---- 4. 评测因子 (仅 IS 区间) ----
    print("\n" + "=" * 60)
    print(f"[步骤4] 因子评测报告 (IS: {IS_START} ~ {IS_END})")
    print("=" * 60)

    for formula, factor_df in factor_results.items():
        metrics = evaluate_factor(factor_df, close_df, forward_returns)
        print(f"\n  因子: {formula}")
        print(f"    RankIC:   {metrics['RankIC']:+.6f}")
        print(f"    ICIR:     {metrics['ICIR']:+.6f}")
        print(f"    Turnover: {metrics['Turnover']:.6f}")

    # ---- 5. 存入 FactorRegistry (SQLite) ----
    print("\n" + "=" * 60)
    print("[步骤5] 因子入库 (SQLite)")
    print("=" * 60)

    registry = FactorRegistry()

    for formula, factor_df in factor_results.items():
        metrics = evaluate_factor(factor_df, close_df, forward_returns)
        success = registry.register(formula=formula, metrics=metrics, factor_df=factor_df)
        status = "[OK] 入库成功" if success else "[REJ] 入库被拒"
        print(f"  {formula} -> {status}")

    print(f"\n  当前注册因子总数: {registry.count}")
    print(f"  数据库路径: {registry.db_path}")

    # ---- 6. 测试相关性拒绝 ----
    print("\n" + "=" * 60)
    print("[步骤6] 相关性去重测试")
    print("=" * 60)

    dup_formula = "cs_rank(ts_mean(close, 20))"
    if dup_formula in factor_results:
        dup_metrics = evaluate_factor(factor_results[dup_formula], close_df, forward_returns)
        is_corr, corr_with, corr_val = registry.check_correlation(factor_results[dup_formula])
        print(f"  重复因子 '{dup_formula}' 与已有因子高度相关: {is_corr}")
        if is_corr:
            print(f"  最大相关因子: {corr_with}, 相关系数: {corr_val:.4f}")
        success = registry.register(
            formula=dup_formula + " #duplicate",
            metrics=dup_metrics,
            factor_df=factor_results[dup_formula],
        )
        print(f"  重复入库结果: {'[OK] 成功' if success else '[REJ] 被拒绝 (符合预期)'}")

    # ---- 7. 查看数据库内容 ----
    print("\n" + "=" * 60)
    print("[步骤7] 数据库内容")
    print("=" * 60)

    all_factors = registry.list_factors()
    for f in all_factors:
        print(f"  {f['factor_id']}: status={f['status']}, "
              f"RankIC={f['rank_ic']:+.4f}, ICIR={f['icir']:+.4f}")

    print("\n" + "=" * 60)
    print("闭环测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
