import numpy as np
import pandas as pd

class UnstructuredGoldVolPivotFactor:
    """非结构化金价波动与政策转向因子 (unstructured/nonlinear)

    逻辑: 捕捉美联储政策预期突变时的FICC跨资产恐慌与衰竭。当经济政策不确定性(EPU)、黄金波动率(GVZ)与短期利率(DGS2)出现极端同向共振飙升，且收益率曲线异常异动时，表明宏观预期发生极值跳跃。当这些恐慌/自满情绪指标开始同时边际回落(衰竭)时，精准狙击美债的反转脉冲机会，避免在主跌浪中接飞刀。
    数据: usepuindxd, gvzcls, dgs2, t10y2y
    触发: 复合转向指数 14日 Z-Score > 2.5 (极值) 且核心利率与黄金波动率同时呈现二阶衰竭 (diff < 0)。目标Trigger Rate在5%-15%之间。
    输出: +1.0表示鹰派/通胀恐慌极值衰竭(抄底买入美债)，-1.0表示鸽派/自满情绪极值衰竭(逢高做空美债)。
    """

    def __init__(self):
        self.name = 'unstructured_gold_vol_pivot'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需的高价值(*)数据列是否存在
        required_cols = ['usepuindxd', 'gvzcls', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 1. 缺失值前向填充 (避免跨界未来数据)
        epu = data['usepuindxd'].ffill()
        gvz = data['gvzcls'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 2. 边际变化计算 (绝对铁律: 禁止使用绝对水位，必须使用 .diff() 捕捉瞬间预期改变)
        epu_smooth = epu.rolling(3).mean()
        epu_diff = epu_smooth.diff(3)
        gvz_diff = gvz.diff(3)
        dgs2_diff = dgs2.diff(3)
        t10y2y_diff = t10y2y.diff(3)
        
        # 3. 基础动量 Z-Score (63交易日为中期宏观基准线)
        epu_z = (epu_diff - epu_diff.rolling(63).mean()) / (epu_diff.rolling(63).std() + 1e-8)
        gvz_z = (gvz_diff - gvz_diff.rolling(63).mean()) / (gvz_diff.rolling(63).std() + 1e-8)
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(63).mean()) / (dgs2_diff.rolling(63).std() + 1e-8)
        t10y2y_z = (t10y2y_diff - t10y2y_diff.rolling(63).mean()) / (t10y2y_diff.rolling(63).std() + 1e-8)
        
        # 4. 非线性特征交叉: 构造 FICC 宏观预期突变复合指数
        # 鹰派恐慌 = 利率飙升(+) + 曲线平坦化(-) + 黄金波动飙升(+) + 不确定性飙升(+)
        hawkish_shock = dgs2_z - t10y2y_z + gvz_z + epu_z
        # 鸽派自满 = 利率暴跌(-) + 曲线陡峭化(+) + 黄金波动暴跌(-) + 不确定性暴跌(-)
        dovish_shock = -dgs2_z + t10y2y_z - gvz_z - epu_z
        
        # 5. 极端脉冲检测 (14日短窗口计算 Z-Score > 2.5，既符合极端法则，又确保 Trigger Rate 达标)
        hawkish_z = (hawkish_shock - hawkish_shock.rolling(14).mean()) / (hawkish_shock.rolling(14).std() + 1e-8)
        dovish_z = (dovish_shock - dovish_shock.rolling(14).mean()) / (dovish_shock.rolling(14).std() + 1e-8)
        
        # 记录近3日内是否爆发过极端政策冲击
        extreme_hawkish = hawkish_z.rolling(3).max() > 2.5
        extreme_dovish = dovish_z.rolling(3).max() > 2.5
        
        # 6. 二阶导数衰竭条件 (核心铁律: 波动率与利率的绝对动量必须明确回落，否则直接死于主浪)
        # 鹰派衰竭: 短端利率停止飙升，且黄金波动率(恐慌)开始回落
        hawkish_exhaustion = (dgs2.diff(2) < 0) & (gvz.diff(2) < 0)
        # 鸽派衰竭: 短端利率停止暴跌(利好出尽)，且避险波动触底反弹
        dovish_exhaustion = (dgs2.diff(2) > 0) & (gvz.diff(2) > 0)
        
        # 7. 生成狙击手级脉冲信号 (非触发日严格为 0.0)
        cond_long = extreme_hawkish & hawkish_exhaustion
        cond_short = extreme_dovish & dovish_exhaustion
        
        signal[cond_long] = 1.0
        signal[cond_short] = -1.0
        
        signal.name = self.name
        return signal