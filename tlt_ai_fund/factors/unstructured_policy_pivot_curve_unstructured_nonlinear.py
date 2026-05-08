import numpy as np
import pandas as pd

class UnstructuredPolicyPivotCurveFactor:
    """政策转向与恐慌衰竭交叉因子 (unstructured/nonlinear)

    逻辑: 结合非结构化数据(FOMC情绪变化与经济政策不确定性EPU)与美债短端利率及曲线形态(dgs2, t10y2y)的非线性交叉。当FOMC鸽派突变或政策恐慌极值叠加短端利率下行(Bull Steepening)且利率下行动量开始衰竭时，输出看多脉冲。避免单边主跌浪接飞刀。
    数据: fomc_sentiment, usepuindxd, dgs2, t10y2y
    触发: (FOMC/EPU极值+曲线顺向 OR 曲线极值+FOMC/EPU顺向) + dgs2动量衰竭(二阶导反转) -> 脉冲
    输出: +1.0 看多美债(降息预期骤升/避险脉冲), -1.0 看空美债(加息预期骤升/风险偏好修复), 常态 0.0
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_curve'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须为 0.0 (三大铁律：零值休眠)
        signal = pd.Series(0.0, index=data.index)
        
        # 缺失列检查
        required_cols = ['fomc_sentiment', 'usepuindxd', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        df = data[required_cols].ffill()
        
        # 1. 边际变化计算 (三大铁律：边际变化，严禁使用绝对水位)
        # FOMC为低频阶梯数据，使用10日差分捕捉近期会议或讲话的边际预期突变
        fomc_diff = df['fomc_sentiment'].diff(10)
        fomc_std = fomc_diff.rolling(252).std().replace(0, np.nan)
        fomc_z = (fomc_diff - fomc_diff.rolling(252).mean()) / fomc_std
        
        # EPU (经济政策不确定性) 5日动量变化
        epu_diff = df['usepuindxd'].diff(5)
        epu_std = epu_diff.rolling(252).std().replace(0, np.nan)
        epu_z = (epu_diff - epu_diff.rolling(252).mean()) / epu_std
        
        # dgs2 (对政策最敏感的2年期美债) 5日动量变化
        dgs2_diff = df['dgs2'].diff(5)
        dgs2_std = dgs2_diff.rolling(252).std().replace(0, np.nan)
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(252).mean()) / dgs2_std
        
        # t10y2y (收益率曲线形态) 5日动量变化
        curve_diff = df['t10y2y'].diff(5)
        
        # 2. 基础方向判断 (Directional Agreement)
        # 看多美债宏观环境：短端急降且曲线变陡 (Bull Steepening)；非结构化指标转向鸽派或政策恐慌上升
        curve_bull_dir = (dgs2_diff < 0) & (curve_diff > 0)
        curve_bear_dir = (dgs2_diff > 0) & (curve_diff < 0)
        
        unstruct_bull_dir = (fomc_diff > 0) | (epu_diff > 0)
        unstruct_bear_dir = (fomc_diff < 0) | (epu_diff < 0)
        
        # 3. 极值条件 (Extreme Triggers Z-Score > 1.5 确保突发性)
        unstruct_bull_ext = (fomc_z > 1.5) | (epu_z > 1.5)
        unstruct_bear_ext = (fomc_z < -1.5) | (epu_z < -1.5)
        
        curve_bull_ext = (dgs2_z < -1.5) & (curve_diff > 0)
        curve_bear_ext = (dgs2_z > 1.5) & (curve_diff < 0)
        
        # 4. 非线性交叉验证 (Nonlinear Cross-Validation)
        # 至少一侧域(非结构化或曲线)发生极端冲击，且另一侧域在方向上予以印证
        # 这样组合可以把目标 Trigger Rate 稳定控制在 5%~15% 的狙击频次
        bull_trigger = (unstruct_bull_ext & curve_bull_dir) | (curve_bull_ext & unstruct_bull_dir)
        bear_trigger = (unstruct_bear_ext & curve_bear_dir) | (curve_bear_ext & unstruct_bear_dir)
        
        # 5. 二阶导数衰竭条件 (三大铁律：二阶导数 Anti Catch-Falling-Knife)
        # 短端利率急剧波动的动量必须放缓，表示市场第一波无脑Pricing已经结束，避免接飞刀
        # dgs2_diff.diff(1) > 0 表示今天的短端收益率跌幅已经小于昨天，下行动量衰竭
        dgs2_exhaustion_bull = dgs2_diff.diff(1) > 0  
        dgs2_exhaustion_bear = dgs2_diff.diff(1) < 0  
        
        # 组合最终触发条件
        bull_condition = bull_trigger & dgs2_exhaustion_bull
        bear_condition = bear_trigger & dgs2_exhaustion_bear
        
        # 6. 生成脉冲信号 (Sniper Pulse)
        signal.loc[bull_condition] = 1.0
        signal.loc[bear_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"