import numpy as np
import pandas as pd

class FiccCrossVolCurveExhaustionFactor:
    """FICC跨资产波动率与收益率曲线交叉衰竭因子 (volatility/nonlinear)

    逻辑: 纯VIX因子容易在主跌浪中因“恐慌消退而无宽松实质”而接飞刀(从而导致与其他多头因子的内部摩擦和冗余)。
          本因子构建了VIX与黄金波动率(GVZ)的跨资产合成恐慌指数，并与收益率曲线的边际变化进行非线性交叉。
          脉冲看多(+1.0): 合成恐慌处于极值并开始回落，且伴随收益率曲线(t10y2y)骤然变陡(货币政策实质性转向预期)，精确过滤掉熊市反弹，捕捉宽松驱动的美债主升浪起点。
          脉冲看空(-1.0): 市场处于极度自满(波动率极低)且开始飙升，伴随曲线骤然平坦化(短端鹰派冲击)，捕捉美债暴跌起点。
    数据: vixcls, gvzcls, t10y2y
    触发: agg_z > 1.2 且 agg_z < 3日均值 且 t10y2y 3日diff > 0 -> +1.0
          agg_z < -1.0 且 agg_z > 3日均值 且 t10y2y 3日diff < 0 -> -1.0
    输出: [-1.0, 1.0] 脉冲信号
    """

    def __init__(self):
        self.name = 'ficc_crossvol_curve_pulse'
        self.window = 252  # 252个交易日，代表1个自然年的波动率基准Regime
        self.long_z_thresh = 1.2  # 分位数阈值，对应约前11.5%的高波极值区间
        self.short_z_thresh = -1.0 # 分位数阈值，对应约后16%的极度自满区间

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index, name=self.name)
        
        # 依赖数据列检查
        required_cols = ['vixcls', 'gvzcls', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 缺失值前向填充处理
        df = data[required_cols].ffill()
        
        # 1. 计算跨资产波动率的 252日 Z-Score (宏观Regime基准)
        vix_mean = df['vixcls'].rolling(self.window).mean()
        vix_std = df['vixcls'].rolling(self.window).std()
        vix_z = (df['vixcls'] - vix_mean) / vix_std
        
        gvz_mean = df['gvzcls'].rolling(self.window).mean()
        gvz_std = df['gvzcls'].rolling(self.window).std()
        gvz_z = (df['gvzcls'] - gvz_mean) / gvz_std
        
        # 2. 合成 FICC 恐慌指数 (等权合成 VIX 与 GVZ 的 Z-Score)
        agg_z = (vix_z + gvz_z) / 2.0
        
        # 3. 铁律2: 二阶导数反转条件 —— 波动率的衰竭与飙升确认 (对比3日均线)
        agg_z_rolling_3 = agg_z.rolling(3).mean()
        vol_exhausting = agg_z < agg_z_rolling_3
        vol_surging = agg_z > agg_z_rolling_3
        
        # 4. 铁律3: 边际变化确认 —— 收益率曲线的动量交叉 (3日变动捕捉突发宏观冲击)
        # 绝对值t10y2y可能存在长期倒挂（魔法数字失效），因此严格使用 .diff()
        t10y2y_diff_3 = df['t10y2y'].diff(3)
        
        # 5. 组合信号触发条件 (非线性交叉，大幅降低误判和投资组合冗余度)
        long_cond = (
            (agg_z > self.long_z_thresh) &    # 条件1: 跨资产恐慌极度高企
            vol_exhausting &                  # 条件2: 恐慌开始衰竭（防接飞刀）
            (t10y2y_diff_3 > 0.0)             # 条件3: 收益率曲线骤然变陡 (证实为货币政策转向，非虚假衰竭)
        )
        
        short_cond = (
            (agg_z < self.short_z_thresh) &   # 条件1: 市场极度自满
            vol_surging &                     # 条件2: 波动率被唤醒
            (t10y2y_diff_3 < 0.0)             # 条件3: 收益率曲线骤然平坦化 (证实为短端利率鹰派飙升)
        )
        
        # 6. 铁律1: 零值休眠，脉冲输出
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"