import numpy as np
import pandas as pd

class CrossAssetOptionsStressExhaustionFactor:
    """跨资产期权隐波恐慌衰竭因子 (unstructured/options)

    逻辑: 结合标普期权(VIX)与黄金期权(GVZ)隐含波动率构建跨资产期权压力指数。极端的跨资产隐波跳升反映了宏观层面的流动性挤兑或无差别恐慌抛售；而当这种极端恐慌脉冲见顶衰竭时，通常标志着央行干预生效或市场情绪极值已过，此时长线资金会凶猛涌入避险美债(TLT)。设计为脉冲信号以避开主跌浪时的接飞刀风险。
    数据: vixcls (标普500隐含波动率), gvzcls (黄金隐含波动率)
    触发: 5日跨资产隐波变化量的 252日 Z-Score > 2.5 (极端期权恐慌激增)，且当前隐波水平回落至3日均值以下 (二阶反转/恐慌衰竭)。
    输出: +1.0 看多美债(宏观恐慌消退抢筹避险)，-1.0 看空美债(极度亢奋衰竭抛售避险)，其余0.0。
    """

    def __init__(self):
        self.name = 'cross_asset_options_stress_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 - 初始全 0.0，只在触发日输出脉冲
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖数据是否存在，若缺失则静默返回全0
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            signal.name = self.name
            return signal
            
        # 前向填充缺失值以防止计算中断
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 构建跨资产期权压力指数
        options_stress = vix + gvz
        
        # 铁律3: 边际变化 - 绝对禁止使用 VIX/GVZ 水平值，使用 5 日激增量捕捉宏观事件冲击突变
        stress_pulse = options_stress.diff(5)
        
        # 计算 252 日滚动 Z-Score (带有 min_periods 保证冷启动鲁棒性)
        roll_mean = stress_pulse.rolling(window=252, min_periods=60).mean()
        roll_std = stress_pulse.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        stress_zscore = (stress_pulse - roll_mean) / roll_std
        
        # 铁律2: 二阶导数 - 绝对禁止单看指标极值接飞刀，必须附带彻底的衰竭条件才扣动扳机
        # 多头衰竭条件: 隐波开始绝对回落 (日度下行) 且 跌破3日均值 (确立短期退潮趋势)
        exhaustion_down = (options_stress < options_stress.rolling(window=3).mean()) & (options_stress.diff(1) < 0)
        
        # 空头衰竭条件: 隐波极度压缩后触底反弹 (日度上行) 且 升破3日均值 (亢奋瓦解，抛售美债)
        exhaustion_up = (options_stress > options_stress.rolling(window=3).mean()) & (options_stress.diff(1) > 0)
        
        # 狙击手级脉冲条件组合
        long_cond = (stress_zscore > 2.5) & exhaustion_down
        short_cond = (stress_zscore < -2.5) & exhaustion_up
        
        # 赋值强脉冲看多 / 看空信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"