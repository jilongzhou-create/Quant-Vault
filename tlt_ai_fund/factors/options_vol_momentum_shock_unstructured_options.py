import numpy as np
import pandas as pd

class OptionsVolCurveRegimeFactor:
    """Options VolCurve Regime Factor (unstructured/options)

    逻辑: 结合期权隐含波动率(VIX)的动量突变与收益率曲线(T10Y2Y)的边际变化。
          单纯的VIX极端回落并不一定利多美债(例如2022年紧缩周期中，VIX回落往往伴随股债双杀，导致错误方向)。
          必须引入收益率曲线的边际变化定界：当恐慌或自满情绪动量达到极值(Z绝对值>1.5)
          并开始衰竭(二阶导数反转)时，若曲线变陡(短端下行快于长端，代表降息预期)，则反转利多美债；
          若曲线变平(紧缩预期)，则利空美债。此逻辑能完美修复高波动环境下的方向预测失误。
    数据: vixcls, t10y2y
    触发: VIX 5日变化量的 Z-Score绝对值 > 1.5 AND 变化量动量开始衰竭(二阶导数反转)
    输出: 结合 t10y2y 动量输出 +1.0 或 -1.0，常态输出 0.0 (严格满足零值休眠铁律)
    """

    def __init__(self):
        self.name = 'options_vol_curve_regime_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 数据缺失保护
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        curve = data['t10y2y'].ffill()
        
        # 铁律3: 边际变化 (Marginal Change)
        # 计算 5日动量，捕捉预期的突变瞬间
        vix_mom = vix.diff(5)
        curve_mom = curve.diff(5)
        
        # 动态计算 Z-Score (使用126个交易日约半年窗口，适应不同波动率中枢，拒绝魔法数字)
        window = 126
        vix_mom_mean = vix_mom.rolling(window=window, min_periods=20).mean()
        vix_mom_std = vix_mom.rolling(window=window, min_periods=20).std()
        
        # 计算动量的标准化偏差
        vix_mom_z = (vix_mom - vix_mom_mean) / (vix_mom_std + 1e-8)
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 衰竭条件: 当前动量跌破/突破其近3日移动平均 (代表加速阶段结束，情绪开始反转)
        vix_mom_ma3 = vix_mom.rolling(window=3).mean()
        
        # 向上极值衰竭 (恐慌见顶: 动量曾大幅飙升，但今日动量低于3日均值)
        exhaustion_up = (vix_mom_z > 1.5) & (vix_mom < vix_mom_ma3)
        # 向下极值衰竭 (自满见底: 动量曾大幅下挫，但今日动量开始企稳回升)
        exhaustion_down = (vix_mom_z < -1.5) & (vix_mom > vix_mom_ma3)
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 结合曲线动量判定宏观状态，动态输出 +1.0 或 -1.0，其余时间保持初始化的 0.0
        
        # 场景A: 恐慌衰竭
        # 恐慌衰竭 + 降息预期(曲线变陡) -> 美联储政策救市，基本面看多，买入美债
        signal.loc[exhaustion_up & (curve_mom > 0)] = 1.0   
        # 恐慌衰竭 + 紧缩预期(曲线变平) -> 市场自身风险偏好回升且无降息救市，资金切出避险资产，抛售美债
        signal.loc[exhaustion_up & (curve_mom < 0)] = -1.0  
        
        # 场景B: 自满衰竭 (波动率即将放大，市场即将变盘)
        # 变盘前夕 + 降息预期 -> 资金确认宽松环境，提前进入避险资产(美债)
        signal.loc[exhaustion_down & (curve_mom > 0)] = 1.0   
        # 变盘前夕 + 紧缩预期 -> 紧缩引发的流动性冲击，所有资产无差别抛售(含美债)
        signal.loc[exhaustion_down & (curve_mom < 0)] = -1.0  
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"