import numpy as np
import pandas as pd

class UnstructuredPolicyPivotExhaustionFactor:
    """政策预期突变与不确定性衰竭交叉因子 (unstructured/nonlinear)

    逻辑: 捕捉宏观经济政策不确定性(EPU)极度恐慌后"靴子落地"的平息瞬间。
          当政策不确定性较高且开始衰竭时，若前瞻利率(dgs2)动量剧烈下行，说明降息预期实质落地，形成做多脉冲；
          若dgs2动量剧烈上行，说明超预期鹰派加息情绪确认，形成做空脉冲。
          因要求捕捉边际定价且防接飞刀，采用短端动量极大值 + 不确定性衰退作为二阶联合条件。
    数据: usepuindxd (经济政策不确定性指数), dgs2 (2年期国债收益率)
    触发: usepuindxd 的Z-score近期>1.2且当期开始回落(低于3日均值) + dgs2 3日差分动量Z-score绝对值>1.5
    输出: +1.0 表示多头脉冲(降息突变)，-1.0 表示空头脉冲(加息突变)，常态 0.0
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 - 初始全 0.0 Series
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必须依赖的数据列是否缺失
        if 'usepuindxd' not in data.columns or 'dgs2' not in data.columns:
            signal.name = self.name
            return signal
            
        # 基础数据提取与前向填充(防止中间休市缺值)
        epu = data['usepuindxd'].ffill()
        dgs2 = data['dgs2'].ffill()
        
        # -------------------------------------------------------------
        # 条件1：宏观不确定性 (EPU) 近期极值与当期衰竭反转
        # -------------------------------------------------------------
        # 计算 EPU 的季度基准 Z-score (60日)
        epu_mean = epu.rolling(window=60, min_periods=20).mean()
        epu_std = epu.rolling(window=60, min_periods=20).std()
        epu_z = (epu - epu_mean) / (epu_std + 1e-8)
        
        # 恐慌脉冲记忆：过去两周(10日)内，政策不确定性曾出现脉冲高位
        epu_high_recent = epu_z.rolling(window=10, min_periods=1).max() > 1.2
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # EPU 衰竭反转：不确定性靴子落地，开始低于近3日均线，确保不是在风险恶化途中建仓
        epu_exhaustion = epu < epu.rolling(window=3, min_periods=1).mean()
        
        # -------------------------------------------------------------
        # 条件2：政策预期核心锚定 (dgs2短端利率) 剧烈重定价
        # -------------------------------------------------------------
        # 铁律3: 边际变化 - 绝对禁止使用收益率水位！必须用短期差分(动量)捕捉跳跃
        dgs2_mom = dgs2.diff(3) 
        dgs2_mom_mean = dgs2_mom.rolling(window=60, min_periods=20).mean()
        dgs2_mom_std = dgs2_mom.rolling(window=60, min_periods=20).std()
        
        # dgs2动量的季度 Z-score，衡量本次预期边际跳跃的剧烈程度
        dgs2_z = (dgs2_mom - dgs2_mom_mean) / (dgs2_mom_std + 1e-8)
        
        # 多头场景：短端利率极其剧烈下行，降息恐慌抢跑
        rate_cut_shock = dgs2_z < -1.5
        
        # 空头场景：短端利率极其剧烈上行，鹰派加息抢跑
        rate_hike_shock = dgs2_z > 1.5
        
        # -------------------------------------------------------------
        # 信号合成 (非线性高维交叉)
        # -------------------------------------------------------------
        long_condition = epu_high_recent & epu_exhaustion & rate_cut_shock
        short_condition = epu_high_recent & epu_exhaustion & rate_hike_shock
        
        # 只在触发点输出脉冲极值，完美符合狙击手级(Trigger Rate预计 5%~15%)
        signal[long_condition] = 1.0
        signal[short_condition] = -1.0
        
        # 清理异常值
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"