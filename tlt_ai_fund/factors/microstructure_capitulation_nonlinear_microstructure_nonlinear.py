import numpy as np
import pandas as pd

class MicrostructureLiquidityPulseFactor:
    """流动性挤兑与恐慌衰竭因子 (microstructure/nonlinear)

    逻辑: 结合资金面微观结构(DFF-DTB3利差)与情绪面(VIX), 捕捉流动性危机与恐慌衰竭脉冲。当市场出现极端流动性挤兑(短期美债遭抢筹导致DTB3相对联邦基金利率暴跌)且VIX同步飙升时, 形成高维复合压力。当此压力处于高位极值并开始回落时(二阶导数衰竭), 标志着央行干预或恐慌见顶, 是做多美债(TLT)的极佳脉冲点。反之亦然。
    数据: dff (联邦基金利率), dtb3 (3个月美债利率), vixcls (波动率指数)
    触发: 复合压力指数 63日 Z-Score > 1.2 且出现衰竭拐点(diff < 0) -> +1.0; Z-Score < -1.2 且拐头上行 -> -1.0
    输出: [-1.0, 1.0] 的狙击手级脉冲信号, 目标 Trigger Rate 5%-15%
    """

    def __init__(self):
        self.name = 'microstructure_liquidity_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化非触发日信号严格为 0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 1. 字段检查
        req_cols = ['dff', 'dtb3', 'vixcls']
        if not all(col in data.columns for col in req_cols):
            return signal
            
        # 2. 数据前向填充, 避免缺失值导致的计算中断
        dff = data['dff'].ffill()
        dtb3 = data['dtb3'].ffill()
        vix = data['vixcls'].ffill()
        
        # 3. 计算资金面微观利差: DFF - DTB3
        # 危机期避险需求导致 T-Bill 遭抢购, 收益率 DTB3 下行, 使得 spread 飙升
        funding_spread = dff - dtb3
        
        # 4. 定义 Z-Score 计算函数 (采用63个交易日即一个季度的滚动窗口适应宏观状态切换)
        def calc_zscore(s: pd.Series, window: int = 63) -> pd.Series:
            roll_mean = s.rolling(window=window, min_periods=21).mean()
            roll_std = s.rolling(window=window, min_periods=21).std()
            return (s - roll_mean) / (roll_std + 1e-8)
            
        fs_z = calc_zscore(funding_spread)
        vix_z = calc_zscore(vix)
        
        # 5. 非线性特征交叉: 构建高维复合流动性压力指数
        # 同时反映资金面与情绪面的恐慌共振
        composite_stress = fs_z + vix_z
        
        # 对复合指数再次求 Z-Score 以标准化触发阈值
        stress_z = calc_zscore(composite_stress, window=63)
        
        # 6. 边际变化与衰竭条件 (严格遵守二阶导数铁律, 防止接飞刀)
        stress_diff = stress_z.diff()
        stress_roll_mean = stress_z.rolling(window=3, min_periods=1).mean()
        
        # 7. 脉冲信号触发
        # 多头信号: 压力指数处于前10%左右极端高位 (Z > 1.2), 且当天开始回落(衰竭) -> 恐慌见顶, 流动性将重回长债, 买入
        long_cond = (stress_z > 1.2) & (stress_diff < 0) & (stress_z < stress_roll_mean)
        
        # 空头信号: 压力指数处于极度自满低位 (Z < -1.2), 且当天开始反弹上行 -> 紧缩预期抬头或避险需求消退, 卖出
        short_cond = (stress_z < -1.2) & (stress_diff > 0) & (stress_z > stress_roll_mean)
        
        # 赋值信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"