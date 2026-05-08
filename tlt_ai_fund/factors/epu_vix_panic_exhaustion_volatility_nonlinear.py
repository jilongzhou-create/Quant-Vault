import numpy as np
import pandas as pd

class MacroCycleHeatExhaustionFactor:
    """宏观周期冷热极值与衰竭反转因子 (volatility/nonlinear)

    逻辑: 结合通胀预期(t5yie)与信用利差(BBB OAS)构建宏观周期冷热度(Macro Heat)。当通胀极高且信用极度自满(高热度)并开始衰竭时，标志着加息周期见顶，长端美债迎来绝佳配置时机(脉冲看多)；当通胀极低且信用极度恐慌(极寒度)并开始修复时，避险情绪消退，长端美债将被抛售(脉冲看空)。
    数据: t5yie (5年期盈亏平衡通胀), bamlc0a4cbbb (BBB企业债OAS), t10y2y (期限利差)
    触发: Macro Heat Z-Score极值 + 边际回落(二阶导数) + 跨资产确认(期限利差动量)
    输出: [-1.0, 1.0] 的狙击手级脉冲信号
    """

    def __init__(self):
        self.name = 'macro_cycle_heat_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查数据完备性
        req_cols = ['t5yie', 'bamlc0a4cbbb', 't10y2y']
        if not all(col in data.columns for col in req_cols):
            return signal
            
        # 前向填充缺失值(节假日等)
        t5yie = data['t5yie'].ffill()
        cred = data['bamlc0a4cbbb'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 铁律3: 计算 252 日滚动 Z-Score，反映相对历史极值 (非绝对水位)
        roll_mean_t5 = t5yie.rolling(252).mean()
        roll_std_t5 = t5yie.rolling(252).std()
        z_t5 = (t5yie - roll_mean_t5) / roll_std_t5
        
        roll_mean_cred = cred.rolling(252).mean()
        roll_std_cred = cred.rolling(252).std()
        z_cred = (cred - roll_mean_cred) / roll_std_cred
        
        # 构建宏观冷热度 (Macro Heat)
        # 经济学含义: 宏观过热 = 高通胀预期 + 极低信用利差(自满)
        # 宏观极寒 = 低通胀预期 + 极高信用利差(恐慌)
        macro_heat = z_t5 - z_cred
        
        # 寻找宏观冷热度自身的历史极端点
        roll_mean_heat = macro_heat.rolling(252).mean()
        roll_std_heat = macro_heat.rolling(252).std()
        heat_z = (macro_heat - roll_mean_heat) / roll_std_heat
        
        # 铁律2 & 3: 衰竭条件计算 (采用均线与差分捕捉边际变化)
        heat_ma3 = macro_heat.rolling(3).mean()
        heat_diff3 = macro_heat.diff(3)
        
        # 跨资产确认: 期限利差动量 (避免在主跌浪/主升浪接飞刀)
        curve_diff5 = t10y2y.diff(5)
        
        # ---------------- 触发逻辑 ----------------
        
        # 多头脉冲 (+1.0): 
        # 1. 过热极值 (heat_z > 1.8)
        # 2. 边际衰竭 (跌破均线且动量为负)
        # 3. 驱动确认 (通胀预期单边向下 + 收益率曲线开始变陡，联储加息预期退潮)
        buy_cond = (
            (heat_z > 1.8) & 
            (macro_heat < heat_ma3) & 
            (heat_diff3 < 0) & 
            (z_t5.diff(1) < 0) & 
            (curve_diff5 > 0)
        )
        
        # 空头脉冲 (-1.0): 
        # 1. 恐慌极值 (heat_z < -1.8)
        # 2. 边际修复 (突破均线且动量为正)
        # 3. 驱动确认 (信用恐慌单边消退 + 收益率曲线开始平坦化，经济韧性确认)
        sell_cond = (
            (heat_z < -1.8) & 
            (macro_heat > heat_ma3) & 
            (heat_diff3 > 0) & 
            (z_cred.diff(1) < 0) & 
            (curve_diff5 < 0)
        )
        
        # 信号赋值
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        # 铁律1: 常态休眠，同时安全清理前252天的滚动计算带来的NaN
        signal.iloc[:252] = 0.0
        signal = signal.fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"