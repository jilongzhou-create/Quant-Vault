import numpy as np
import pandas as pd

class UnstructuredPolicyCrossFactor:
    """Unstructured Policy Cross Factor (unstructured/nonlinear)

    逻辑: 政策预期(FOMC)与经济政策不确定性(EPU)的非线性交叉。美债在"高不确定性+美联储转鸽"时具有最强避险属性；而在"低不确定性(极度自满)+美联储超预期转鹰"时遭遇最强抛售。此因子捕捉这两类极值状态的边际突变与衰竭，输出脉冲信号以规避连续接飞刀。
    数据: usepuindxd (经济政策不确定性指数), fomc_sentiment (FOMC鹰鸽情绪得分, NLP提取)
    触发: 
      看多(+1.0): 
        条件A: EPU Z-Score > 2.0 且 EPU < 3日均值(恐慌衰竭) 且 FOMC月度边际变鸽(diff > 0)
        条件B: FOMC鸽派突变 Z-Score > 1.5 且 动量 < 3日均值(鸽派极值衰竭) 且 EPU Z > 0.5(宏观不确定性偏高)
      看空(-1.0): 
        条件A: EPU Z-Score < -2.0 且 EPU > 3日均值(自满破裂) 且 FOMC月度边际变鹰(diff < 0)
        条件B: FOMC鹰派突变 Z-Score < -1.5 且 动量 > 3日均值(鹰派极值衰竭) 且 EPU Z < -0.5(宏观偏向过热)
    输出: 脉冲型信号 [-1.0, 0.0, 1.0]，常态下严格休眠(0.0)。
    """

    def __init__(self):
        self.name = 'unstructured_policy_cross_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 基础异常处理与零值休眠初始化
        signal = pd.Series(0.0, index=data.index)
        
        if 'usepuindxd' not in data.columns or 'fomc_sentiment' not in data.columns:
            return signal

        # 2. 数据提取与前向填充 (避免低频数据带来的 NaN 影响)
        epu = data['usepuindxd'].ffill()
        fomc = data['fomc_sentiment'].ffill()

        # 3. EPU 不确定性指标计算 (252天=1年滚动Z-Score)
        epu_mean_1y = epu.rolling(window=252, min_periods=63).mean()
        epu_std_1y = epu.rolling(window=252, min_periods=63).std()
        epu_z = (epu - epu_mean_1y) / (epu_std_1y + 1e-8)
        
        # 二阶导数铁律: EPU 衰竭特征 (短均线反转)
        epu_ma3 = epu.rolling(window=3).mean()
        epu_exhaustion_high = epu < epu_ma3  # 极高位开始回落
        epu_exhaustion_low = epu > epu_ma3   # 极低位开始反弹

        # 4. FOMC 情绪边际变化指标 (21天=1个月, 捕捉低频阶梯数据的预期突变)
        fomc_mom = fomc.diff(21)
        fomc_mom_mean_1y = fomc_mom.rolling(window=252, min_periods=63).mean()
        fomc_mom_std_1y = fomc_mom.rolling(window=252, min_periods=63).std()
        fomc_mom_z = (fomc_mom - fomc_mom_mean_1y) / (fomc_mom_std_1y + 1e-8)
        
        # 二阶导数铁律: FOMC 动量衰竭特征
        fomc_mom_ma3 = fomc_mom.rolling(window=3).mean()
        fomc_exhaustion_dovish = fomc_mom < fomc_mom_ma3  # 鸽派突变极值见顶回落
        fomc_exhaustion_hawkish = fomc_mom > fomc_mom_ma3 # 鹰派突变极值见底反弹

        # 5. 非线性交叉逻辑与信号生成
        
        # 多头触发: (恐慌极值见顶 + 美联储确认转鸽) 或 (美联储极端转鸽 + 宏观背景配合)
        cond1_long = (epu_z > 2.0) & epu_exhaustion_high & (fomc_mom > 0)
        cond2_long = (fomc_mom_z > 1.5) & fomc_exhaustion_dovish & (epu_z > 0.5)
        
        # 空头触发: (自满极值破裂 + 美联储确认转鹰) 或 (美联储极端转鹰 + 宏观过热配合)
        cond1_short = (epu_z < -2.0) & epu_exhaustion_low & (fomc_mom < 0)
        cond2_short = (fomc_mom_z < -1.5) & fomc_exhaustion_hawkish & (epu_z < -0.5)

        # 严格脉冲赋值 (常态0.0在初始化时已保证)
        signal[cond1_long | cond2_long] = 1.0
        signal[cond1_short | cond2_short] = -1.0

        # 清理由于均线窗口初期产生的无效信号
        signal.iloc[:63] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"UnstructuredPolicyCrossFactor(name='{self.name}')"