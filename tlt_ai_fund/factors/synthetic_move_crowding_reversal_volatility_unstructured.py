import numpy as np
import pandas as pd

class SyntheticMoveCrowdingReversalFactor:
    """波动率极值与拥挤反转 (Synthetic MOVE Crowding Reversal)

    逻辑: 由于经典的债市恐慌指数(MOVE)数据缺失，本因子采用股市 VIX 与高收益债 OAS 的真实波动率，合成近似的“跨资产债市恐慌指数(Synthetic MOVE)”。美债作为避险终极资产，常态下保持沉默 (0.0)。只有当合成恐慌指数狂飙至罕见极值 (Z-Score > 2.5)，且随后股市(VIX)和黄金(GVZCLS)波动率同步出现二阶向下衰竭时，标志流动性危机错杀结束与避险盘的重新流入，此时输出 +1.0 的极高胜率做多脉冲。反之在极度自满期被边际打破时做空。
    数据: vixcls, bamlhe00ehyioas, gvzcls
    触发: Synthetic MOVE 半年期 Z-Score > 2.5 且 VIX.diff(3) < 0 且 GVZCLS.diff(3) < 0 -> +1.0 脉冲
    输出: 纯脉冲信号，极端事件触发日及其后衰竭日输出 +1.0/-1.0，其余时间为 0.0
    """

    def __init__(self):
        self.name = 'synthetic_move_crowding_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态下必须保持 0.0, 狙击手级脉冲
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'bamlhe00ehyioas', 'gvzcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 向前填充非交易日造成的缺失值
        vix = data['vixcls'].ffill()
        hy_oas = data['bamlhe00ehyioas'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 提取边际波动：信用利差近 1 个月(21个交易日)的真实波动率
        credit_vol = hy_oas.diff().rolling(21).std()
        
        # 使用 126 日 (约半年) 滚动窗口作为宏观基准期，计算波动水位的 Z-Score
        roll_window = 126
        vix_z = (vix - vix.rolling(roll_window).mean()) / vix.rolling(roll_window).std()
        credit_vol_z = (credit_vol - credit_vol.rolling(roll_window).mean()) / credit_vol.rolling(roll_window).std()
        
        # 等权合成宏观债市恐慌度 (Synthetic MOVE Z-Score)
        synthetic_move_z = (vix_z + credit_vol_z) / 2.0
        
        # --- 多头逻辑: 极度恐慌 + 二阶导数衰竭 ---
        
        # 条件1: 跨资产恐慌狂飙至罕见高位 (符合铁律 > 2.5)
        extreme_panic = synthetic_move_z > 2.5
        
        # 条件2: 恐慌开始瓦解的二阶导数衰竭确认 (避免接主跌浪飞刀)
        # 要求 3 日动量向下，确认 VIX 与 黄金波动率 同时回落
        vix_exhaustion = vix.diff(3) < 0
        gvz_exhaustion = gvz.diff(3) < 0
        
        bull_condition = extreme_panic & vix_exhaustion & gvz_exhaustion
        
        # --- 空头逻辑: 极度自满 + 边际恶化 ---
        
        # 条件1: 市场极度自满 (合成波动率极度压缩，拥挤做多风险资产)
        extreme_complacency = synthetic_move_z < -2.0
        
        # 条件2: 自满环境被边际打破 (波动率急剧跳升，流动性趋紧预期重燃)
        vix_shock = vix.diff(3) > 1.5
        credit_shock = hy_oas.diff(3) > 0.05
        
        bear_condition = extreme_complacency & vix_shock & credit_shock
        
        # 填充可能产生的前期 NaN，确保布尔索引安全
        bull_condition = bull_condition.fillna(False)
        bear_condition = bear_condition.fillna(False)
        
        # 最终信号分配，仅在脉冲条件同时满足的短暂窗口中触发
        signal.loc[bull_condition] = 1.0
        signal.loc[bear_condition] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"