import numpy as np
import pandas as pd

class PolicyPivotGoldilocksBreakevenFactor:
    """政策转向与金发姑娘脉冲因子 (policy_pivot/nonlinear)

    逻辑: 真正的长牛买点出现在"无衰退降息预期"被定价的瞬间(金发姑娘经济)。当短端利率(DGS2)快速下行定价降息, 
          同时10年期盈亏平衡通胀(T10YIE)不降反升(确认需求端未崩溃、非硬着陆)时, 形成强烈的看多脉冲。
          相反, 当短端利率飙升且通胀预期下行时, 属于典型的"鹰派紧缩+衰退担忧"双杀, 输出看空脉冲。
    数据: dgs2 (2年期国债收益率), t10yie (10年期盈亏平衡通胀)
    输出: +1.0 看多(金发姑娘经济), -1.0 看空(紧缩+衰退恐慌)
    触发条件: DGS2 3日变化量 < -10bps 且 T10YIE 3日变化量 > +1bp 触发多头脉冲; 反之触发空头脉冲。
    """

    def __init__(self):
        self.name = 'policy_pivot_goldilocks_breakeven_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖列
        required_cols = ['dgs2', 't10yie']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        dgs2 = data['dgs2'].ffill()
        t10yie = data['t10yie'].ffill()
        
        # 使用3日差分捕捉短期脉冲变化
        # 3天的窗口既能过滤单日噪音，又能保证信号在事件发生后迅速休眠归零
        dgs2_diff = dgs2.diff(3)
        t10yie_diff = t10yie.diff(3)
        
        # 极值脉冲交叉逻辑 (非线性组合)
        # 阈值经济学含义: 0.10(10个基点)是美债短端的显著定价变化; 0.01(1个基点)是盈亏平衡通胀的边际确认
        
        # 看多脉冲: 市场抢跑降息(DGS2大跌) + 需求端健康(盈亏平衡通胀边际上升) = 金发姑娘经济
        bull_pulse = (dgs2_diff < -0.10) & (t10yie_diff > 0.01)
        
        # 看空脉冲: 鹰派超预期(DGS2飙升) + 衰退担忧(盈亏平衡通胀边际下降) = 戴维斯双杀
        bear_pulse = (dgs2_diff > 0.10) & (t10yie_diff < -0.01)
        
        signal[bull_pulse] = 1.0
        signal[bear_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"