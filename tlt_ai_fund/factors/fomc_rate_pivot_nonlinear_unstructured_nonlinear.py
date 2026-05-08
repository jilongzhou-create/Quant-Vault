import numpy as np
import pandas as pd

class FomcRatePivotNonlinearFactor:
    """FOMC预期与短端利率共振因子 (unstructured/nonlinear)

    逻辑: 捕捉美联储鸽派/鹰派预期突变与短期利率动量的非线性共振。FOMC情绪骤变且对政策最敏感的两年期美债收益率同步发生剧烈边际变化时, 表明宏观预期发生历史性跳跃(Pivot)。必须等待动量见顶并跌破3日均线后产生脉冲, 从而避免主跌浪接飞刀。
    数据: fomc_sentiment, dgs2
    触发: FOMC情绪5日边际Z-Score 与 短端利率反转Z-Score 的几何共振得分 > 2.5 + 开始回落(低于3日均值衰竭)。
    输出: 狙击手级别脉冲信号, 只在极端事件的拐点日输出 +1.0 (鸽派降息) 或 -1.0 (鹰派加息), 否则为 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_rate_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠。常态下信号必须严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['fomc_sentiment', 'dgs2']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        # 铁律3: 边际变化。绝对禁止使用阶梯数据的绝对值, 捕捉前瞻预期的瞬间突变
        fomc_diff = data['fomc_sentiment'].diff(5)
        dgs2_diff = data['dgs2'].diff(5)
        
        # 计算 252 交易日 (约一年) 滚动 Z-Score，反映极端偏离
        fomc_mean = fomc_diff.rolling(252, min_periods=60).mean()
        fomc_std = fomc_diff.rolling(252, min_periods=60).std() + 1e-8
        fomc_z = ((fomc_diff - fomc_mean) / fomc_std).fillna(0.0)
        
        dgs2_mean = dgs2_diff.rolling(252, min_periods=60).mean()
        dgs2_std = dgs2_diff.rolling(252, min_periods=60).std() + 1e-8
        
        # dgs2 收益率下降(负向变动) = 降息预期飙升 = 看多美债
        # 将 Z-Score 取负向，统一使得正值代表"看多美债动量"
        dgs2_inv_z = (-(dgs2_diff - dgs2_mean) / dgs2_std).fillna(0.0)
        
        # 方法C: 非线性特征交叉。同向共振时激活，过滤单边杂音
        same_sign = np.sign(fomc_z) == np.sign(dgs2_inv_z)
        
        # 使用几何均值保留同向的非线性共振强度，剔除极小值带来的误判
        cross_score_vals = np.where(
            same_sign, 
            np.sign(fomc_z) * np.sqrt(np.abs(fomc_z * dgs2_inv_z)), 
            0.0
        )
        cross_score = pd.Series(cross_score_vals, index=data.index)
        
        # 铁律2: 二阶导数 (衰竭防飞刀)。用3日平滑均线确认动量已经开始反转
        cross_score_ma3 = cross_score.rolling(3).mean()
        
        # 多头脉冲: 鸽派突变且短端暴跌, 综合得分极高 (> 2.5), 且动量开始向下衰竭
        long_trigger = (
            (cross_score > 2.5) & 
            (cross_score < cross_score_ma3) & 
            (fomc_z > 1.0) & 
            (dgs2_inv_z > 1.0)
        )
        
        # 空头脉冲: 鹰派突变且短端飙升, 综合得分极低 (< -2.5), 且动量开始向上衰竭
        short_trigger = (
            (cross_score < -2.5) & 
            (cross_score > cross_score_ma3) & 
            (fomc_z < -1.0) & 
            (dgs2_inv_z < -1.0)
        )
        
        # 仅在触发时进行极端赋值
        signal.loc[long_trigger] = 1.0
        signal.loc[short_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"