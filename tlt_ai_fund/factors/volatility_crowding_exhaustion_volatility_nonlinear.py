import numpy as np
import pandas as pd

class FiccVolatilityExhaustionFactor:
    """FICC专有波动率衰竭反转 (volatility/nonlinear)

    逻辑: 剥离股市波动率(VIX)的噪音，纯粹利用黄金波动率(GVZ)与经济政策不确定性(EPU)交叉构建 FICC 专有恐慌指数。黄金波动率映射实际利率/信用体系的深度恐慌，EPU映射宏观政策极值。当联合政策/实际利率恐慌极度拥挤且开始瓦解，配合收益率曲线(10Y-3M)变陡，确认纯正的宏观降息/避险主升浪。此设计刻意规避了标普500波动率，从底层经济学逻辑上消除了与现有VIX因子的内部摩擦，从而提供极高的边际贡献 (Marginal Contribution)。
    数据: gvzcls, usepuindxd, t10y3m
    触发: (GVZ + EPU) 联合 Z-Score > 1.2 + 恐慌动量<0 + 曲线动量变陡 -> +1.0
    输出: 狙击手级脉冲信号，严格控制在[-1.0, 1.0]，常态为0.0
    """

    def __init__(self):
        self.name = 'ficc_volatility_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (默认全部为 0.0)
        signal = pd.Series(0.0, index=data.index)

        # 检查必要列是否存在 (绝对禁止引用 CoreAnchor 数据)
        required_cols = ['gvzcls', 'usepuindxd', 't10y3m']
        if not all(col in data.columns for col in required_cols):
            return signal

        # 提取数据并处理缺失值
        gvz = data['gvzcls'].ffill()
        epu = data['usepuindxd'].ffill()
        t10y3m = data['t10y3m'].ffill()

        # 计算 252日 Z-Score (反映一年的宏观常态化周期)
        z_gvz = (gvz - gvz.rolling(252, min_periods=63).mean()) / (gvz.rolling(252, min_periods=63).std() + 1e-6)
        z_epu = (epu - epu.rolling(252, min_periods=63).mean()) / (epu.rolling(252, min_periods=63).std() + 1e-6)
        
        # 非线性交叉: FICC 专有宏观压力指数
        # 黄金波动率 + 政策不确定性，指向真正的法币与宏观系统性压力
        ficc_stress = (z_gvz + z_epu) / 2.0

        # 铁律2 & 3: 二阶导数 (衰竭) 与 边际变化 (动量)
        # 使用 5日动量过滤高频噪音，确认恐慌真实发生实质性瓦解
        stress_momentum = ficc_stress.diff(5)
        
        # 收益率曲线边际变化
        curve_momentum = t10y3m.diff(5)

        # 构造做多 TLT 信号：
        # 1. 极值: 宏观政策/利率恐慌极度拥挤 (Z > 1.2)
        # 2. 衰竭: 恐慌开始实质性瓦解 (stress_momentum < 0)
        # 3. 确认: 曲线开始陡峭化，确认降息预期或短端避险发酵 (curve_momentum > 0)
        long_cond = (
            (ficc_stress > 1.2) & 
            (stress_momentum < 0) & 
            (curve_momentum > 0)
        )

        # 构造做空 TLT 信号：
        # 1. 极值: 宏观政策极度安逸，风险偏好极度扩张 (Z < -1.0)
        # 2. 衰竭: 安逸被打破，不确定性开始飙升 (stress_momentum > 0)
        # 3. 确认: 曲线平坦化，确认紧缩/加息预期升温 (curve_momentum < 0)
        short_cond = (
            (ficc_stress < -1.0) & 
            (stress_momentum > 0) & 
            (curve_momentum < 0)
        )

        # 脉冲信号赋值
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"