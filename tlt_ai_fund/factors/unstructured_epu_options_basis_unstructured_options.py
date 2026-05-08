import numpy as np
import pandas as pd

class UnstructuredEpuOptionsBasisFactor:
    """Unstructured EPU vs Options VIX Basis Pulse (unstructured/options)

    逻辑: 衡量基于非结构化新闻的经济政策不确定性(EPU)与基于期权市场的隐含波动率(VIX)之间的边际变化剪刀差。当宏观政策恐慌远超股市恐慌并达到极值后开始衰竭时，往往预示着美联储已被迫向宏观风险妥协（转鸽），触发美债脉冲做多信号；反之，若纯市场流动性冲击极值消退，则触发做空信号。
    数据: usepuindxd (非结构化经济政策不确定性指数), vixcls (VIX期权波动率)
    触发: basis_z > 2.5 且 basis < 3日均值(衰竭) -> +1.0；basis_z < -2.5 且 basis > 3日均值(衰竭) -> -1.0
    输出: 狙击手级脉冲信号，[-1.0, 1.0]，非极端反转日严格输出 0.0
    """

    def __init__(self):
        self.name = 'unstructured_epu_options_basis'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (默认常态全为0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查数据完整性
        if 'usepuindxd' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        epu = data['usepuindxd']
        vix = data['vixcls']
        
        # 铁律3: 边际变化 - 提取周度(5日)动量变化，严禁使用绝对水位
        epu_diff = epu.diff(5)
        vix_diff = vix.diff(5)
        
        # 计算各自的 252日(年度)滚动 Z-Score，将两者的量纲和波动率标准化
        epu_std = epu_diff.rolling(252).std().replace(0, np.nan)
        vix_std = vix_diff.rolling(252).std().replace(0, np.nan)
        
        epu_z = (epu_diff - epu_diff.rolling(252).mean()) / epu_std
        vix_z = (vix_diff - vix_diff.rolling(252).mean()) / vix_std
        
        # 核心逻辑: 构建 非结构化宏观恐慌 vs 期权微观恐慌 的边际剪刀差 (Basis)
        basis = epu_z - vix_z
        
        # 衡量剪刀差自身的极值程度
        basis_std = basis.rolling(252).std().replace(0, np.nan)
        basis_z = (basis - basis.rolling(252).mean()) / basis_std
        
        # 铁律2: 二阶导数 - 必须等极值开始回落(衰竭)时才触发信号，防接飞刀
        basis_ma3 = basis.rolling(3).mean()
        
        # 产生多头脉冲: 政策恐慌远超期权恐慌(纯宏观冲击)，且开始衰竭 -> 政策即将转向让步，看多美债
        long_cond = (basis_z > 2.5) & (basis < basis_ma3)
        
        # 产生空头脉冲: 期权恐慌远超政策恐慌(纯市场流动性冲击)，且开始衰竭 -> 风险偏好见底回升，避险消退，看空美债
        short_cond = (basis_z < -2.5) & (basis > basis_ma3)

        # 触发脉冲赋值
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"