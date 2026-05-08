import numpy as np
import pandas as pd

class UnstructuredMacroPivotNonlinearFactor:
    """宏观政策转向非线性交叉因子 (unstructured/nonlinear)

    逻辑: 捕捉"恐慌极端+降息预期骤升"的多维共振脉冲。将VIX水位、短端利率(DGS2)边际变动和收益率曲线(T10Y2Y)边际变动合成一个多维共振强度指数。当该指数突破极值，且VIX开始回落(恐慌见顶衰竭，避免流动性枯竭导致的现金为王抛售)时，代表美联储政策转鸽被市场多维度定价，触发强烈看多美债的买入脉冲。反之亦然。
    数据: vixcls, dgs2, t10y2y
    触发: 多维共振指数滚动 Z-Score > 2.0，且叠加二阶衰竭条件 (看多: vixcls.diff(3) < 0; 看空: vixcls.diff(3) > 0)
    输出: 满足极端脉冲及衰竭条件时瞬间输出 +1.0 (看多) / -1.0 (看空)，常态下保持零值休眠(0.0)以控制Trigger Rate在狙击手级别
    """

    def __init__(self):
        self.name = 'unstructured_macro_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠, 初始信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 填充缺失值并剥离所需数据
        vix = data['vixcls'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 使用 252 交易日计算动态滚动宏观分布
        window = 252
        
        # 1. 绝对水位特征: VIX 波动率恐慌水位 (仅供交叉过滤使用)
        vix_z = (vix - vix.rolling(window).mean()) / vix.rolling(window).std()
        
        # 2. 铁律3 边际变化特征: DGS2 短端利率剧烈变化 (极度敏感的政策预期前瞻)
        dgs2_diff = dgs2.diff(5)
        dgs2_diff_z = (dgs2_diff - dgs2_diff.rolling(window).mean()) / dgs2_diff.rolling(window).std()
        
        # 3. 铁律3 边际变化特征: T10Y2Y 收益率曲线动量 (捕捉短端下行引发的突然牛陡/熊平)
        t10y2y_diff = t10y2y.diff(5)
        t10y2y_diff_z = (t10y2y_diff - t10y2y_diff.rolling(window).mean()) / t10y2y_diff.rolling(window).std()
        
        # 4. 非线性高维交叉 - 策略转折共振合成指数 (Macro Pivot Shock Index)
        # 看多方向: VIX处于高位(恐慌) + 短端收益率暴降(负DGS2突变) + 曲线变陡(正T10Y2Y突变)
        bull_index = (vix_z * 0.5) - dgs2_diff_z + (t10y2y_diff_z * 0.5)
        bull_index_z = (bull_index - bull_index.rolling(window).mean()) / bull_index.rolling(window).std()
        
        # 看空方向: VIX处于极低位(贪婪) + 短端收益率暴涨(加息预期升温) + 曲线变平(衰退预期滞后)
        bear_index = -(vix_z * 0.5) + dgs2_diff_z - (t10y2y_diff_z * 0.5)
        bear_index_z = (bear_index - bear_index.rolling(window).mean()) / bear_index.rolling(window).std()
        
        # 5. 铁律2 二阶导数防飞刀: 涉及流动性恐慌/极端的环境必须确认指标边际收敛
        bull_exhaustion = vix.diff(3) < 0
        bear_exhaustion = vix.diff(3) > 0
        
        # 6. 生成脉冲触发条件: 综合强度极大 + 二阶衰竭反转
        buy_cond = (bull_index_z > 2.0) & bull_exhaustion
        sell_cond = (bear_index_z > 2.0) & bear_exhaustion
        
        # 赋值狙击手信号
        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"