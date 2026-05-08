import numpy as np
import pandas as pd

class PanicExhaustionMicrostructureFactor:
    """恐慌极值与衰竭反转脉冲 (Microstructure/Unstructured)

    逻辑: 针对流动性危机的"接飞刀"问题，本因子在恐慌极值处捕捉衰竭信号。
          当VIX极度飙升引发无差别抛售(债市放量 capitulation)后，一旦恐慌指标见顶回落(二阶导数向下)，
          标志着流动性冲击结束，避险资金重新回流美债，引发脉冲式反弹。
          相反，在极度自满且金融极度宽松的环境下，一旦恐慌抬头，往往伴随加息或通胀预期重燃，美债遭到抛售。
    数据: vixcls (波动率指数), nfci (国家金融状况指数), volume (TLT成交量微观结构)
    触发: VIX Z-Score > 1.25 + 衰竭(低于3日均值) + NFCI紧缩 + 放量 → 看多脉冲 +1.0
          VIX Z-Score < -1.25 + 抬头(高于3日均值) + NFCI宽松 + 放量 → 看空脉冲 -1.0
    输出: 严格脉冲型信号，触发及随后2天内输出信号(共3天)，常态下为0，目标 Trigger Rate 5%-15%
    """

    def __init__(self):
        self.name = 'panic_exhaustion_microstructure'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须严格为 0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 必须数据列检查
        if 'vixcls' not in data.columns or 'nfci' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        nfci = data['nfci'].ffill()
        
        # 1. 恐慌极值计算 (使用 252 日滚动 Z-Score 捕捉相对偏离)
        vix_mean = vix.rolling(window=252, min_periods=63).mean()
        vix_std = vix.rolling(window=252, min_periods=63).std()
        vix_std = vix_std.replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        # 2. 衰竭与反转条件 (二阶导数铁律: 绝不在极值单边趋势中逆势接飞刀)
        vix_ma3 = vix.rolling(window=3, min_periods=1).mean()
        vix_exhausting = vix < vix_ma3  # 高位恐慌开始回落
        vix_bouncing = vix > vix_ma3    # 低位自满开始抬头
        
        # 3. 金融压力交叉验证 (避免假突破)
        # NFCI > 0 代表金融环境紧缩(确认危机), < 0 代表宽松(确认自满环境)
        stress_tight = nfci > 0
        stress_loose = nfci < 0
        
        # 4. 微观结构验证: 成交量 Capitulation (恐慌抛售或FOMO抢筹)
        # 确认债市确实在经历放量博弈
        if 'volume' in data.columns:
            vol = data['volume'].ffill()
            vol_ma = vol.rolling(window=21, min_periods=5).mean()
            vol_ratio = vol / vol_ma.replace(0, np.nan)
            # 宽容放量条件: 近3天内成交量达到均值 1.2 倍即可
            vol_capitulation = vol_ratio.rolling(window=3, min_periods=1).max() > 1.2
        else:
            # 防御性回退
            vol_capitulation = pd.Series(True, index=data.index)
            
        # 5. 触发逻辑综合
        # 采用 1.25 的 Z-Score 阈值，保证双边合并后的 Trigger Rate 能落在 5% - 15% 的黄金区间
        long_cond = (vix_z > 1.25) & vix_exhausting & stress_tight & vol_capitulation
        short_cond = (vix_z < -1.25) & vix_bouncing & stress_loose & vol_capitulation
        
        # 边缘触发检测 (只在第一天满足条件时记录为 trigger)
        long_trigger = long_cond & (~long_cond.shift(1).fillna(False))
        short_trigger = short_cond & (~short_cond.shift(1).fillna(False))
        
        # 展期 3 天形成有效脉冲 (T, T+1, T+2), 以匹配美债修复周期，优化 Hit Rate
        long_pulse = long_trigger | long_trigger.shift(1).fillna(False) | long_trigger.shift(2).fillna(False)
        short_pulse = short_trigger | short_trigger.shift(1).fillna(False) | short_trigger.shift(2).fillna(False)
        
        # 赋值
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0
        
        # 优先级冲突处理: 若双侧同时满足(极其罕见的异常抖动)，保持休眠
        conflict = long_pulse & short_pulse
        signal[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"