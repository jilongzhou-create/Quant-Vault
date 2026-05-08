#!/usr/bin/env python3
"""
因子解析、评测与存储引擎

功能1: parse_and_calculate - 将公式字符串解析为因子 DataFrame
功能2: evaluate_factor     - 计算因子的 RankIC / ICIR / Turnover (仅 IS 区间)
功能3: FactorRegistry      - 因子持久化存储 (SQLite) 与相关性去重
"""

import hashlib
import logging
import os
import sqlite3
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from etf_rotation_strategy import operators as ops
from etf_rotation_strategy.db.schema import get_connection, init_tables, DB_PATH

logger = logging.getLogger(__name__)

IS_START = '2018-01-01'
IS_END = '2022-12-31'


# ===================== 功能1: 因子解析器 =====================

def parse_and_calculate(
    formula_str: str,
    data_dict: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    将公式字符串解析计算为因子 DataFrame

    将 data_dict 中的键 (open/high/low/close/volume) 和 operators.py
    中的全部函数注入命名空间，然后使用 eval(formula_str, namespace) 计算。

    Args:
        formula_str: 因子公式，如 "cs_rank(ts_mean(close, 20))"
        data_dict:   数据字典，键为字段名，值为 (T, 11) DataFrame

    Returns:
        pd.DataFrame: (T, 11) 因子值矩阵

    Raises:
        ValueError: 公式语法错误或算子名称拼写错误
    """
    namespace: Dict = {}

    namespace.update({
        'add': ops.add, 'sub': ops.sub, 'mul': ops.mul, 'div': ops.div,
        'log': ops.log, 'sign': ops.sign, 'abs_val': ops.abs_val,
        'delay': ops.delay, 'delta': ops.delta, 'ts_returns': ops.ts_returns,
        'ts_mean': ops.ts_mean, 'ts_max': ops.ts_max, 'ts_min': ops.ts_min,
        'ts_std': ops.ts_std, 'ts_rank': ops.ts_rank,
        'cs_rank': ops.cs_rank, 'cs_zscore': ops.cs_zscore,
        'correlation': ops.correlation,
    })

    namespace.update(data_dict)

    namespace['__builtins__'] = {}

    try:
        result = eval(formula_str, namespace)
    except NameError as e:
        raise ValueError(f"因子公式中存在未定义的名称: {e}") from e
    except SyntaxError as e:
        raise ValueError(f"因子公式语法错误: {e}") from e
    except Exception as e:
        raise ValueError(f"因子公式计算失败: {e}") from e

    if not isinstance(result, pd.DataFrame):
        raise ValueError(
            f"因子公式计算结果类型错误: 期望 pd.DataFrame, 实际 {type(result)}"
        )

    result = result.replace([np.inf, -np.inf], np.nan)
    return result


# ===================== 功能2: 因子统计评测 =====================

def evaluate_factor(
    factor_df: pd.DataFrame,
    close_df: pd.DataFrame,
    forward_returns: pd.DataFrame,
    is_start: str = IS_START,
    is_end: str = IS_END,
) -> Dict[str, float]:
    """
    评测因子质量，返回 RankIC / ICIR / Turnover 指标

    仅使用 IS 区间 (默认 2018-01-01 ~ 2022-12-31) 的数据进行统计评估，
    防止数据穿越 (Data Leakage)。

    仅在有足够截面样本 (至少5只有效ETF) 的日期计算 IC 均值。

    Args:
        factor_df:        (T, 11) 因子值矩阵 (全量数据，含 warm-up)
        close_df:         (T, 11) 收盘价矩阵 (用于对齐)
        forward_returns:  (T, 11) 未来N日收益率矩阵
        is_start:         IS 区间起始日期
        is_end:           IS 区间结束日期

    Returns:
        dict: {'RankIC': float, 'ICIR': float, 'Turnover': float}
    """
    is_start_dt = pd.Timestamp(is_start)
    is_end_dt = pd.Timestamp(is_end)

    common_idx = factor_df.index.intersection(forward_returns.index)
    common_idx = common_idx[(common_idx >= is_start_dt) & (common_idx <= is_end_dt)]
    common_cols = factor_df.columns.intersection(forward_returns.columns)

    if len(common_idx) == 0:
        logger.warning(f"[EVAL] IS 区间内无有效日期: {is_start} ~ {is_end}")
        return {'RankIC': np.nan, 'ICIR': np.nan, 'Turnover': np.nan}

    f = factor_df.loc[common_idx, common_cols]
    fr = forward_returns.loc[common_idx, common_cols]

    valid_mask = f.notna() & fr.notna()
    f_clean = f.where(valid_mask)
    fr_clean = fr.where(valid_mask)

    min_cross_section = 5
    valid_count = valid_mask.sum(axis=1)

    f_rank = f_clean.rank(axis=1, pct=True)
    fr_rank = fr_clean.rank(axis=1, pct=True)

    f_mean = f_rank.mean(axis=1, skipna=True)
    fr_mean = fr_rank.mean(axis=1, skipna=True)
    f_std = f_rank.std(axis=1, skipna=True).replace(0, np.nan)
    fr_std = fr_rank.std(axis=1, skipna=True).replace(0, np.nan)

    f_z = f_rank.sub(f_mean, axis=0).div(f_std, axis=0)
    fr_z = fr_rank.sub(fr_mean, axis=0).div(fr_std, axis=0)

    daily_ic = (f_z * fr_z).mean(axis=1, skipna=True)
    daily_ic = daily_ic[valid_count >= min_cross_section]
    daily_ic = daily_ic.dropna()

    if len(daily_ic) == 0:
        logger.warning(f"[EVAL] IS 区间内无有效 IC (截面样本不足 {min_cross_section})")
        return {'RankIC': np.nan, 'ICIR': np.nan, 'Turnover': np.nan}

    rank_ic_mean = float(daily_ic.mean())
    rank_ic_std = float(daily_ic.std())
    icir = rank_ic_mean / rank_ic_std if rank_ic_std > 1e-8 else np.nan

    f_ranked = f.rank(axis=1, pct=True)
    daily_turnover = f_ranked.diff().abs().mean(axis=1, skipna=True)
    daily_turnover = daily_turnover[valid_count >= min_cross_section]
    daily_turnover = daily_turnover.dropna()
    turnover = float(daily_turnover.mean()) if len(daily_turnover) > 0 else np.nan

    logger.info(
        f"[EVAL] IS={is_start}~{is_end}, "
        f"RankIC={rank_ic_mean:+.6f}, ICIR={icir:+.4f}, "
        f"Turnover={turnover:.6f}, 有效日数={len(daily_ic)}"
    )

    return {
        'RankIC': rank_ic_mean,
        'ICIR': icir,
        'Turnover': turnover,
    }


# ===================== 功能3: 存储管理器 =====================

def _generate_factor_id(formula: str) -> str:
    h = hashlib.md5(formula.encode()).hexdigest()[:8]
    safe = formula.replace('(', '_').replace(')', '_').replace(',', '_').replace(' ', '')[:40]
    return f"{safe}_{h}"


class FactorRegistry:
    """
    因子持久化存储 (SQLite) 与相关性去重

    因子生命周期 (当前简化版):
      draft → is_passed → accepted

    预留完整生命周期 (后续 Phase 扩展):
      draft → is_passed → accepted → dead / dead_oos / superseded
    """

    def __init__(self, db_path: str = DB_PATH) -> None:
        self.db_path = db_path
        self._factor_dfs: Dict[str, pd.DataFrame] = {}
        init_tables(db_path)
        self._load_cache()

    def _load_cache(self) -> None:
        conn = get_connection(self.db_path)
        try:
            rows = conn.execute(
                "SELECT factor_id, formula, status FROM etf_factor_registry "
                "WHERE status IN ('is_passed', 'accepted')"
            ).fetchall()
            for row in rows:
                logger.debug(f"[REGISTRY] 已加载因子: {row[0]} (status={row[2]})")
        finally:
            conn.close()

    def _register_to_db(
        self,
        factor_id: str,
        formula: str,
        metrics: Dict[str, float],
        status: str = 'is_passed',
        mining_round: Optional[int] = None,
        max_corr_with: Optional[str] = None,
        max_corr_value: Optional[float] = None,
    ) -> None:
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                """INSERT OR REPLACE INTO etf_factor_registry
                   (factor_id, formula, status, mining_round, rank_ic, icir, turnover,
                    max_corr_with, max_corr_value, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
                (
                    factor_id, formula, status, mining_round,
                    metrics.get('RankIC'), metrics.get('ICIR'), metrics.get('Turnover'),
                    max_corr_with, max_corr_value,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def _save_audit(
        self,
        factor_id: str,
        metrics: Dict[str, float],
        verdict: str,
        reject_reason: Optional[str] = None,
        is_start: str = IS_START,
        is_end: str = IS_END,
        max_corr_with: Optional[str] = None,
        max_corr_value: Optional[float] = None,
    ) -> None:
        conn = get_connection(self.db_path)
        try:
            conn.execute(
                """INSERT INTO etf_factor_audit
                   (factor_id, is_start, is_end, rank_ic, icir, turnover,
                    max_corr_with, max_corr_value, verdict, reject_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    factor_id, is_start, is_end,
                    metrics.get('RankIC'), metrics.get('ICIR'), metrics.get('Turnover'),
                    max_corr_with, max_corr_value, verdict, reject_reason,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def check_correlation(
        self,
        new_factor_df: pd.DataFrame,
        threshold: float = 0.6,
    ) -> tuple:
        """
        检查新因子与已有因子的截面相关系数

        Returns:
            (is_highly_correlated: bool, max_corr_with: str or None, max_corr_value: float or None)
        """
        if not self._factor_dfs:
            return False, None, None

        new_rank = new_factor_df.rank(axis=1, pct=True)
        max_corr_with = None
        max_corr_value = 0.0

        for old_id, old_df in self._factor_dfs.items():
            old_rank = old_df.rank(axis=1, pct=True)
            common_idx = new_rank.index.intersection(old_rank.index)
            if len(common_idx) == 0:
                continue

            nr = new_rank.loc[common_idx]
            or_ = old_rank.loc[common_idx]

            valid = nr.notna() & or_.notna()
            nr = nr.where(valid)
            or_ = or_.where(valid)

            nr_mean = nr.mean(axis=1, skipna=True)
            or_mean = or_.mean(axis=1, skipna=True)
            nr_std = nr.std(axis=1, skipna=True).replace(0, np.nan)
            or_std = or_.std(axis=1, skipna=True).replace(0, np.nan)

            nr_z = nr.sub(nr_mean, axis=0).div(nr_std, axis=0)
            or_z = or_.sub(or_mean, axis=0).div(or_std, axis=0)

            daily_corr = (nr_z * or_z).mean(axis=1, skipna=True)
            daily_corr = daily_corr.dropna()
            if len(daily_corr) > 0:
                corr_val = float(daily_corr.mean())
                if abs(corr_val) > abs(max_corr_value):
                    max_corr_value = corr_val
                    max_corr_with = old_id
                if corr_val > threshold:
                    return True, old_id, corr_val

        return False, max_corr_with, max_corr_value

    def register(
        self,
        formula: str,
        metrics: Dict[str, float],
        factor_df: pd.DataFrame,
        mining_round: Optional[int] = None,
        force: bool = False,
    ) -> bool:
        """
        将因子注册入库

        Args:
            formula:       因子公式字符串
            metrics:       评测指标字典
            factor_df:     (T, 11) 因子值矩阵
            mining_round:  挖掘轮次
            force:         是否强制入库 (跳过相关性检查)

        Returns:
            bool: True = 入库成功，False = 被拒绝
        """
        factor_id = _generate_factor_id(formula)

        is_correlated, corr_with, corr_value = self.check_correlation(factor_df)

        if not force and is_correlated:
            logger.info(f"[REGISTRY] 因子被拒绝 (与 {corr_with} 相关性 {corr_value:.4f}): {formula}")
            self._register_to_db(
                factor_id, formula, metrics,
                status='dead', mining_round=mining_round,
                max_corr_with=corr_with, max_corr_value=corr_value,
            )
            self._save_audit(
                factor_id, metrics, verdict='DEAD',
                reject_reason=f'与已有因子 {corr_with} 截面相关性 {corr_value:.4f} 过高',
                max_corr_with=corr_with, max_corr_value=corr_value,
            )
            return False

        self._register_to_db(
            factor_id, formula, metrics,
            status='is_passed', mining_round=mining_round,
            max_corr_with=corr_with, max_corr_value=corr_value,
        )
        self._save_audit(
            factor_id, metrics, verdict='ACCEPTED',
            max_corr_with=corr_with, max_corr_value=corr_value,
        )
        self._factor_dfs[factor_id] = factor_df.copy()
        logger.info(f"[REGISTRY] 因子已入库: {factor_id} ({formula})")
        return True

    def list_factors(self, status: Optional[str] = None) -> List[Dict]:
        conn = get_connection(self.db_path)
        try:
            if status:
                rows = conn.execute(
                    "SELECT factor_id, formula, status, rank_ic, icir, turnover "
                    "FROM etf_factor_registry WHERE status = ? ORDER BY created_at",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT factor_id, formula, status, rank_ic, icir, turnover "
                    "FROM etf_factor_registry ORDER BY created_at",
                ).fetchall()
            return [
                {
                    'factor_id': r[0], 'formula': r[1], 'status': r[2],
                    'rank_ic': r[3], 'icir': r[4], 'turnover': r[5],
                }
                for r in rows
            ]
        finally:
            conn.close()

    @property
    def count(self) -> int:
        conn = get_connection(self.db_path)
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM etf_factor_registry "
                "WHERE status IN ('is_passed', 'accepted')"
            ).fetchone()
            return row[0]
        finally:
            conn.close()
