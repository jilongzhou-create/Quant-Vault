import numpy as np
import pandas as pd

class UnstructuredPolicyUncertaintyShockFactor:
    """Unstructured Policy Uncertainty Curve Shock (unstructured/nonlinear)

    逻辑: 当非结构化经济政策不确定性新闻指数(USEPU)极度飙升并开始衰竭时，如果此时收益率曲线因短端利率剧烈变化而发生动量突变(Bull Steepening看多/Bear Flattening看空)，说明新闻事件直接促成了宏观政策预期的根本反转。利用不确定性消散的瞬间脉冲顺势狙击美债。
    数据: usepuindxd (经济政策新闻不确定性), t10y2y (长短期利差), dgs2 (2年期收益率)
    触发: USEPU Z-Score > 1.5 且开始回落 (恐慌衰竭) + T10Y2Y 5日动量极端变陡/变平 + DGS2 5日大幅变动 > 10bps
    输出: 降息共振时输出 +1.0 (看多), 加息共振时输出 -1.0 (看空), 其余时间严格为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_policy_uncertainty_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['usepuindxd', 't10y2y', 'dgs2']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        df = data[required_cols].ffill()
        
        # 1. 非结构化新闻政策不确定性指数的极值与衰竭 (遵守二阶导数铁律)
        usepu = df['usepuindxd']
        # 使用 63 个交易日 (约一季度) 作为局部宏观环境的基准
        usepu_roll_mean = usepu.rolling(window=63, min_periods=21).mean()
        usepu_roll_std = usepu.rolling(window=63, min_periods=21).std()
        usepu_z = (usepu - usepu_roll_mean) / (usepu_roll_std + 1e-8)
        
        # 衰竭条件：不确定性极高，但已停止恶化并开始回落 (当前值小于3日均值)
        usepu_exhaustion = (usepu_z > 1.5) & (usepu < usepu.rolling(3).mean())
        
        # 2. 收益率曲线动量 (遵守边际变化铁律, 禁止直接使用倒挂水位)
        t10y2y_diff5 = df['t10y2y'].diff(5)
        t10y2y_diff_mean = t10y2y_diff5.rolling(window=63, min_periods=21).mean()
        t10y2y_diff_std = t10y2y_diff5.rolling(window=63, min_periods=21).std()
        t10y2y_z = (t10y2y_diff5 - t10y2y_diff_mean) / (t10y2y_diff_std + 1e-8)
        
        # 3. 短端利率预期突变 (作为曲线变形的驱动力印证)
        dgs2_diff5 = df['dgs2'].diff(5)
        
        # 4. 非线性特征交叉触发逻辑
        # Bull Steepening (强降息预期发酵): 利差剧烈扩宽 (变陡) + 纯由短端暴跌主导 (5天内收益率下行 > 10 bps)
        bull_steepening = (t10y2y_z > 1.5) & (dgs2_diff5 < -0.10)
        
        # Bear Flattening (强加息预期发酵): 利差剧烈收窄 (变平) + 纯由短端暴涨主导 (5天内收益率上行 > 10 bps)
        bear_flattening = (t10y2y_z < -1.5) & (dgs2_diff5 > 0.10)
        
        # 5. 生成狙击手脉冲信号 (遵守零值休眠铁律)
        long_cond = usepu_exhaustion & bull_steepening
        short_cond = usepu_exhaustion & bear_flattening
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"