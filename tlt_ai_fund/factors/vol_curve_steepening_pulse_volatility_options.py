import numpy as np
import pandas as pd

class VolCurveSteepeningPulseFactor:
    """波动率期限微观结构脉冲因子 (volatility/options)

    逻辑: 捕捉期权隐含波动率(VIX)与利率期限预期微观动量的极值共振。当股市恐慌(VIX狂飙)叠加短端利率急剧宽松预期(曲线10日变陡动能剧增)时，标志着宏观避险与对冲盘的极端拥挤。此时若两者的激增动能同时开始衰竭，则意味着恐慌解体与流动性冲击消退，此时为长端美债(TLT)提供确定性极高的做多反转点。反之，当市场狂热自满且紧缩动能极致恶化并衰竭时，输出看空脉冲。这是不接飞刀、只打拐点的狙击手级脉冲信号。
    数据: vixcls, t10y2y
    触发: VIX与曲线变陡动量的复合 Z-Score > 3.0 (极度恐慌) 或 < -2.5 (极度狂热)，且动能跌破3日短期均值发生实质性衰竭。
    输出: +1.0 表示避险拥挤解体做多美债(TLT)，-1.0 表示自满狂热解体做空美债。非触发日严格输出休眠值 0.0。
    """

    def __init__(self):
        self.name = 'vol_curve_steepening_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律: 处理缺失数据
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)
            
        vix = data['vixcls'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        if vix.isna().all() or t10y2y.isna().all():
            return pd.Series(0.0, index=data.index, name=self.name)

        # 1. 计算 VIX 隐含波动率的 252日 Z-Score
        vix_mean = vix.rolling(window=252, min_periods=126).mean()
        vix_std = vix.rolling(window=252, min_periods=126).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        # 铁律: 边际变化 - 绝对禁止使用曲线绝对水位，用 10日 diff 捕捉市场预期的剧烈突变
        curve_mom = t10y2y.diff(10)
        curve_mom_mean = curve_mom.rolling(window=252, min_periods=126).mean()
        curve_mom_std = curve_mom.rolling(window=252, min_periods=126).std()
        curve_mom_z = (curve_mom - curve_mom_mean) / curve_mom_std.replace(0, np.nan)
        
        # 构建跨域避险应力共振指数 (股市恐慌挤压 + 债市宽货币抢跑)
        combined_stress = vix_z + curve_mom_z
        
        # 铁律: 二阶导数 - 使用 3日短窗口均值作为单日平滑参照，验证动能衰竭
        vix_3ma = vix.rolling(window=3, min_periods=1).mean()
        curve_mom_3ma = curve_mom.rolling(window=3, min_periods=1).mean()
        
        # 多头脉冲 (+1.0): 极端恐慌避险反转
        # 极值条件: 复合宏观恐慌应力 > 3.0 (高置信度极值)
        # 衰竭条件: 恐慌VIX和变陡势头都低于过去三日均值 (证明二阶导意见顶)
        long_extreme = combined_stress > 3.0
        long_exhaustion = (vix < vix_3ma) & (curve_mom < curve_mom_3ma)
        
        # 空头脉冲 (-1.0): 流动性泛滥狂热反转
        # 极值条件: 复合自满应力 < -2.5 (股市VIX极低贪婪且曲线倒挂极速压迫)
        # 衰竭条件: 贪婪VIX和倒挂势头抬头，向上突破三日均值 (证明动能减退)
        short_extreme = combined_stress < -2.5
        short_exhaustion = (vix > vix_3ma) & (curve_mom > curve_mom_3ma)
        
        # 铁律: 零值休眠 - 非常态下严格保持 0.0，触发率强制锁在区间内
        signal = pd.Series(0.0, index=data.index)
        
        # 只在严苛条件复合交集时触发
        signal[long_extreme & long_exhaustion] = 1.0
        signal[short_extreme & short_exhaustion] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"