import numpy as np
import pandas as pd

class FundingStressExhaustionFactor:
    """资金面与情绪共振衰竭因子 (microstructure/nonlinear)

    逻辑: 结合微观资金面(DFF-DTB3抵押品溢价)与宏观情绪(VIX)的非线性交叉。在流动性危机期间，无风险抵押品(3个月美债)遭遇抢购导致其收益率大幅低于联邦基金利率，当此抵押品极度短缺与VIX恐慌情绪同步出现极值时，意味着危机达到顶点。二者同步开始回落（衰竭）标志着美联储流动性干预生效或市场恐慌见顶，此时长端美债(TLT)迎来高胜率的脉冲抄底机会。
    数据: dff (联邦基金利率), dtb3 (3个月美债收益率), vixcls (VIX恐慌指数)
    触发: VIX 126日Z-Score > 2.5 且 资金面压力Z-Score > 2.0，并且两者同时低于3日均值发生边际反转衰竭。
    输出: +1.0 (脉冲型，多重危机极值同步见顶回落后做多美债)
    """

    def __init__(self):
        self.name = 'funding_stress_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0.0 的 Series，确保零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖列是否存在，避免跨域报错
        required_cols = ['dff', 'dtb3', 'vixcls']
        for col in required_cols:
            if col not in data.columns:
                signal.name = self.name
                return signal

        # 提取数据并处理缺失值
        df = data[required_cols].ffill()

        # 微观资金面压力: 联邦基金利率(无担保) - 3个月国债(强抵押品)
        funding_stress = df['dff'] - df['dtb3']

        # 动态Z-Score计算 (126个交易日，约半年，适应性捕捉中期脉冲极值)
        fs_mean = funding_stress.rolling(window=126, min_periods=63).mean()
        fs_std = funding_stress.rolling(window=126, min_periods=63).std()
        fs_zscore = (funding_stress - fs_mean) / fs_std.replace(0, 1e-5)

        vix_mean = df['vixcls'].rolling(window=126, min_periods=63).mean()
        vix_std = df['vixcls'].rolling(window=126, min_periods=63).std()
        vix_zscore = (df['vixcls'] - vix_mean) / vix_std.replace(0, 1e-5)

        # 边际变化与衰竭条件 (二阶导数铁律: 绝对禁止极值直接抄底，必须等开始回落)
        # 当日值低于过去3日均值，证明单边恶化动量已被打破
        fs_exhaustion = funding_stress < funding_stress.rolling(window=3).mean()
        vix_exhaustion = df['vixcls'] < df['vixcls'].rolling(window=3).mean()

        # 极值条件定义
        fs_extreme = fs_zscore > 2.0
        vix_extreme = vix_zscore > 2.5

        # 非线性交叉共振触发: 极端恐慌 + 极端资金面短缺 + 同步出现边际改善
        trigger = fs_extreme & vix_extreme & fs_exhaustion & vix_exhaustion

        # 脉冲信号赋值 (仅触发点及其确定的位置为+1.0)
        signal[trigger] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"