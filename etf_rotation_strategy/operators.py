#!/usr/bin/env python3
"""
因子算子库 - 纯数学计算算子

所有输入 x, y 均为 2D pd.DataFrame:
  - Index: 日期 (升序, Datetime类型)
  - Columns: 11只ETF代码

全部使用 pandas/numpy 向量化运算，严禁 for 循环。
min_periods 统一设为 d // 2，处理好 NaN 与无穷大。
"""

import numpy as np
import pandas as pd


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace([np.inf, -np.inf], np.nan)


# ===================== 算术类 =====================

def add(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return _clean(x + y)


def sub(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return _clean(x - y)


def mul(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return _clean(x * y)


def div(x: pd.DataFrame, y: pd.DataFrame) -> pd.DataFrame:
    return _clean(x / y.replace(0, np.nan))


def log(x: pd.DataFrame) -> pd.DataFrame:
    return _clean(np.log(x.replace(0, np.nan)))


def sign(x: pd.DataFrame) -> pd.DataFrame:
    return _clean(np.sign(x))


def abs_val(x: pd.DataFrame) -> pd.DataFrame:
    return _clean(np.abs(x))


# ===================== 时序类 =====================

def delay(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.shift(d)


def delta(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return _clean(x - x.shift(d))


def ts_returns(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return _clean(x / x.shift(d) - 1)


def ts_mean(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=d // 2).mean()


def ts_max(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=d // 2).max()


def ts_min(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=d // 2).min()


def ts_std(x: pd.DataFrame, d: int) -> pd.DataFrame:
    return x.rolling(window=d, min_periods=d // 2).std()


def ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
    min_p = max(1, d // 2)
    return x.rolling(window=d, min_periods=min_p).rank(pct=True)


# ===================== 截面类 =====================

def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    return x.rank(axis=1, pct=True)


def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    std = std.replace(0, np.nan)
    return _clean(x.sub(mean, axis=0).div(std, axis=0))


# ===================== 交互类 =====================

def correlation(x: pd.DataFrame, y: pd.DataFrame, d: int) -> pd.DataFrame:
    return _clean(x.rolling(window=d, min_periods=d // 2).corr(y))
