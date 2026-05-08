import numpy as np
import pandas as pd

class YieldCurveVixExhaustionFactor:
    """非线性曲线波动率共振因子 (unstructured/nonlinear)

    逻辑: 捕捉货币政策预期突变与市场风险情绪衰竭的高维非线性共振。美债(TLT)常死于流动性危机的主跌浪，因此看多必须是：短端利率剧烈下行(降息预期骤升) + 曲线牛陡(确认为货币驱动) + VIX恐慌达到极值但开始衰竭回落(规避接飞刀)；看空则相反，为紧缩预期飙升且极度平静被打破。
    数据: dgs2, t10y2y, vixcls
    触发: DGS2 5日动量 Z-Score 极值(>1.5) + t10y2y 形态验证 + VIX Z-Score 极值(>1.0) + VIX 动量衰竭 (当前 < 3日均值)
    输出: +1.0 看多美债(TLT)，-1.0 看空美债，常态 0.0 零值休眠脉冲
    """

    def __init__(self):
        self.name = 'yield_curve_vix_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号全为0，严格遵守零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 依赖列检查
        required_cols = ['dgs2', 't10y2y', 'vixcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 提取数据并向前填充处理缺失值
        df = data[required_cols].ffill()
        
        # 1. 政策预期边际突变: dgs2 的 5日变化量 (动量) 及 Z-Score (63日滚动，约一季度)
        # 遵守边际变化铁律：只看预期的瞬时改变
        dgs2_diff = df['dgs2'].diff(5)
        dgs2_diff_mean = dgs2_diff.rolling(window=63).mean()
        dgs2_diff_std = dgs2_diff.rolling(window=63).std()
        dgs2_z = (dgs2_diff - dgs2_diff_mean) / (dgs2_diff_std + 1e-6)
        
        # 2. 期限结构验证: t10y2y 曲线形态的 5日边际变化
        # t10y2y_diff > 0 结合 dgs2_diff < 0 即为典型的 Bull Steepening (牛陡)
        t10y2y_diff = df['t10y2y'].diff(5)
        
        # 3. 恐慌情绪极值与二阶导数衰竭验证 (Anti-Catch-Falling-Knife)
        # 严格遵守二阶导数铁律：极值 + 衰竭
        vix_mean = df['vixcls'].rolling(window=63).mean()
        vix_std = df['vixcls'].rolling(window=63).std()
        vix_z = (df['vixcls'] - vix_mean) / (vix_std + 1e-6)
        
        vix_3d_mean = df['vixcls'].rolling(window=3).mean()
        # 衰竭条件：VIX虽在高位，但短线已经见顶回落
        vix_exhausting = df['vixcls'] < vix_3d_mean
        # 反转条件：VIX在极低位，但短线已经开始抬头
        vix_rebounding = df['vixcls'] > vix_3d_mean
        
        # 看多信号逻辑：
        # 1. dgs2_z < -1.5：短端急剧下行(美联储被迫降息预期骤升)
        # 2. t10y2y_diff > 0.0：长端下行更慢，曲线牛陡，确认降息性质
        # 3. vix_z > 1.0 & vix_exhausting：处于恐慌期，但恐慌已过极点(不接杀流动性的飞刀)
        long_cond = (dgs2_z < -1.5) & (t10y2y_diff > 0.0) & (vix_z > 1.0) & vix_exhausting
        
        # 看空信号逻辑：
        # 1. dgs2_z > 1.5：短端急剧上行(超预期鹰派冲击)
        # 2. t10y2y_diff < 0.0：长端跟不上，曲线熊平，确认紧缩性质
        # 3. vix_z < -1.0 & vix_rebounding：市场之前极度贪婪/平静，现在波澜刚起(杀估值启动)
        short_cond = (dgs2_z > 1.5) & (t10y2y_diff < 0.0) & (vix_z < -1.0) & vix_rebounding
        
        # 赋值脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"