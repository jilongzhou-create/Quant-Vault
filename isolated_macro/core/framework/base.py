#!/usr/bin/env python3
"""
Base Interfaces - MacroTrendFramework 基础因子接口

核心设计原则:
  1. 所有因子输出 [-1.0, 1.0] 连续平滑信号 (禁止阶跃 if-else)
  2. 使用 Z-Score + tanh 映射实现平滑压缩
  3. 统一接口: calculate_signal(data) -> pd.Series
  4. AI Agent 可自动实现 BaseFactor 接口编写新因子
"""

import numpy as np
import pandas as pd


def smooth_signal(raw_series: pd.Series, zscore_window: int = 252,
                  cap: float = 2.0, ema_span: int = 10) -> pd.Series:
    """
    将原始序列转化为 [-1.0, 1.0] 连续平滑信号

    步骤:
      1. Rolling Z-Score 标准化 (消除量纲和均值漂移)
      2. clip 到 [-cap, cap] (防止极端值主导)
      3. tanh 压缩到 [-1, 1] (S 形平滑, 非阶跃)
      4. EMA 平滑 (减少信号抖动)

    Args:
        raw_series: 原始因子序列 (如差分、ROC等)
        zscore_window: Z-Score 滚动窗口 (default 252 = 1年)
        cap: Z-Score 截断阈值 (default 2.0)
        ema_span: EMA 平滑跨度 (default 10天)

    Returns:
        pd.Series: [-1.0, 1.0] 范围的连续信号
    """
    rolling_mean = raw_series.rolling(window=zscore_window, min_periods=63).mean()
    rolling_std = raw_series.rolling(window=zscore_window, min_periods=63).std()
    z = (raw_series - rolling_mean) / (rolling_std + 1e-9)
    z = z.clip(-cap, cap)
    signal = np.tanh(z)
    signal = signal.ewm(span=ema_span, min_periods=1).mean()
    return signal


class BaseFactor:
    """
    因子基类 - 所有宏观因子的标准接口

    子类必须实现:
      calculate_signal(data: pd.DataFrame) -> pd.Series
        输入: 包含所需列的宽表 DataFrame
        输出: [-1.0, 1.0] 范围的连续信号 Series

    属性:
      name: 因子唯一标识符 (如 'tips_momentum', 'credit_spread')
      direction: 信号方向, +1 表示正方向 (值越大越看多), -1 表示反方向
      zscore_window: Z-Score 标准化窗口
      ema_span: EMA 平滑跨度
    """

    def __init__(self, name: str, direction: int = 1,
                 zscore_window: int = 252, ema_span: int = 10):
        if direction not in (-1, 1):
            raise ValueError(f"direction must be -1 or 1, got {direction}")
        self.name = name
        self.direction = direction
        self.zscore_window = zscore_window
        self.ema_span = ema_span

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        """
        计算因子信号 (子类必须覆写)

        Args:
            data: DataFrame, 至少包含因子所需的原始数据列

        Returns:
            pd.Series: index 与 data 相同, 值域 [-1.0, 1.0]
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement calculate_signal()"
        )

    def _apply_smooth(self, raw_series: pd.Series, cap: float = 2.0) -> pd.Series:
        """
        标准平滑管道: Z-Score -> clip -> tanh -> EMA -> 方向调整

        子类在 calculate_signal 中计算出原始序列后,
        调用此方法完成标准化和方向调整。
        """
        signal = smooth_signal(
            raw_series,
            zscore_window=self.zscore_window,
            cap=cap,
            ema_span=self.ema_span,
        )
        return signal * self.direction

    def __repr__(self):
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"direction={self.direction})")
