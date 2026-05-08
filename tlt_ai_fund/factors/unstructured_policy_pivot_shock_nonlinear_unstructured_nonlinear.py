import numpy as np
import pandas as pd

class UnstructuredPolicyPivotShockNonlinearFactor:
    """Unstructured Policy Pivot Shock (unstructured/nonlinear)

    逻辑: 捕捉美联储政策预期的极端突变(狙击手级别脉冲)。结合非结构化NLP情绪(fomc_sentiment)的边际跳跃与短端利率预期(dgs2)/曲线形态(t10y2y)的非线性交叉。平时严格休眠，仅在预期突变及随后极短几天内释放信号。
    数据: fomc_sentiment, dgs2, t10y2y
    触发: 
      多头(鸽派): FOMC情绪5日变化Z-Score > 2.5，或 短端利率极度下行(Z < -2.5)且曲线陡峭(Z > 1.5)且伴随下行动量衰竭(Second Derivative > 0)。
      空头(鹰派): FOMC情绪5日变化Z-Score < -2.5，或 短端利率极度飙升(Z > 2.5)且曲线平坦(Z < -1.5)且伴随上行动量衰竭(Second Derivative < 0)。
    输出: +1.0 看多美债 (降息预期激增), -1.0 看空美债 (加息预期激增), 脉冲触发后保持5日，确保 Trigger Rate 控制在 5%-15%。
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_shock_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 检查数据完整性，若缺少则直接返回 0.0 的 Series
        required_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        # 提取并前向填充数据，处理低频数据的稀疏性
        fomc = data['fomc_sentiment'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # --- 三大铁律 3：边际变化铁律 (绝对禁止使用低频数据绝对值) ---
        # 计算 5 日边际动量变化
        fomc_diff = fomc.diff(5)
        dgs2_diff = dgs2.diff(5)
        t10y2y_diff = t10y2y.diff(5)

        # 计算滚动 Z-Score (252个交易日窗口)
        window = 252
        fomc_z = (fomc_diff - fomc_diff.rolling(window).mean()) / (fomc_diff.rolling(window).std() + 1e-5)
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(window).mean()) / (dgs2_diff.rolling(window).std() + 1e-5)
        t10y2y_z = (t10y2y_diff - t10y2y_diff.rolling(window).mean()) / (t10y2y_diff.rolling(window).std() + 1e-5)

        # --- 三大铁律 2：二阶导数铁律 (极值 + 衰竭) ---
        # DGS2_diff 的二阶导：diff(2) 用于判断当前极值的动量是否开始放缓(衰竭)
        
        # 鸽派突变 (看多美债 TLT: +1.0)
        # 纯粹的 FOMC NLP 鸽派情绪突变 (阶梯数据自身带脉冲属性)
        dovish_nlp = fomc_z > 2.5
        
        # 非线性交叉验证：短端利率极度暴跌(Z < -2.5) + 曲线牛陡(Z > 1.5) + 下行速率衰竭(不再加速)
        bull_steepening = (dgs2_z < -2.5) & (t10y2y_z > 1.5) & (dgs2_diff.diff(2) > 0)

        # 鹰派突变 (看空美债 TLT: -1.0)
        hawkish_nlp = fomc_z < -2.5
        
        # 非线性交叉验证：短端利率极度飙升(Z > 2.5) + 曲线熊平(Z < -1.5) + 上行速率衰竭(不再加速)
        bear_flattening = (dgs2_z > 2.5) & (t10y2y_z < -1.5) & (dgs2_diff.diff(2) < 0)

        # 获取当天激发的脉冲触发点
        pulse_long = dovish_nlp | bull_steepening
        pulse_short = hawkish_nlp | bear_flattening

        # --- 三大铁律 1：零值休眠铁律 (狙击手脉冲) ---
        # 极端事件 Z>2.5 发动频率极低。为达到 5% - 15% 的目标 Trigger Rate，
        # 我们将高能脉冲的余波在随后 5 个交易日内保持有效，随后因子重新陷入深度休眠 (0.0)
        signal_long_active = pulse_long.rolling(window=5, min_periods=1).max() > 0
        signal_short_active = pulse_short.rolling(window=5, min_periods=1).max() > 0

        # 初始化输出信号 (默认严格为 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 赋值并过滤多空同时触发的无效混沌期
        signal[signal_long_active & ~signal_short_active] = 1.0
        signal[signal_short_active & ~signal_long_active] = -1.0

        # 抹除前置计算窗口内产生的噪音
        signal[:window] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"