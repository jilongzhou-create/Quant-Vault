import numpy as np
import pandas as pd

class CrossAssetImpliedPanicReversalFactor:
    """跨资产隐含恐慌反转因子 (volatility/options)

    逻辑: 极度恐慌时，美股期权(VIX)、黄金期权(GVZCLS)隐含波动率及债市利差波动率会同步狂飙。当三者合成的恐慌指数达到极端高位并衰竭回落时，标志着全资产无差别抛售结束，资金重新流回安全资产，长债因避险属性迎来确定性反弹脉冲；反之，波动率极度低迷(自满)且开始抬头时，常预示利率上行风险暴露，产生做空脉冲。
    数据: vixcls, gvzcls, t10y2y
    触发: 合成跨资产恐慌Z-Score > 2.0 且跌破3日均值 -> +1.0；Z-Score < -1.5 且升破3日均值 -> -1.0。
    输出: 脉冲信号，+1.0为做多美债(恐慌衰竭)，-1.0为做空美债(自满打破)，其余常态为0.0。
    """

    def __init__(self):
        self.name = 'cross_asset_implied_panic_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，常态为0.0
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'gvzcls', 't10y2y']
        missing_cols = [c for c in required_cols if c not in data.columns]
        if missing_cols:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 异常值清理与前向填充
        vix = vix.replace([np.inf, -np.inf], np.nan).ffill()
        gvz = gvz.replace([np.inf, -np.inf], np.nan).ffill()
        t10y2y = t10y2y.replace([np.inf, -np.inf], np.nan).ffill()

        # 铁律3: 边际变化
        # 债市微观波动率代理：10年-2年利差的极短期21日绝对边际变化均值，反映债市抛售定价摩擦动量
        bond_vol = t10y2y.diff().abs().rolling(21, min_periods=5).mean()
        
        # 归一化三大资产波动率以消除绝对水位的量级差异 (转化为相对于过去252日均值的比例)
        vix_norm = vix / (vix.rolling(252, min_periods=21).mean() + 1e-6)
        gvz_norm = gvz / (gvz.rolling(252, min_periods=21).mean() + 1e-6)
        bond_norm = bond_vol / (bond_vol.rolling(252, min_periods=21).mean() + 1e-6)
        
        # 合成跨资产避险情绪期权恐慌指数 (等权加和，确保不被单端绑架)
        cross_panic_idx = vix_norm + gvz_norm + bond_norm
        
        # 计算合成恐慌指数的长期 252日 Z-Score (锁定肥尾极端偏离度)
        cross_panic_mean = cross_panic_idx.rolling(252, min_periods=63).mean()
        cross_panic_std = cross_panic_idx.rolling(252, min_periods=63).std()
        cross_panic_z = (cross_panic_idx - cross_panic_mean) / (cross_panic_std + 1e-6)
        
        # 铁律2: 二阶导数衰竭条件，绝对禁止飞刀
        # 指标不仅要极值，还必须出现折返(边际动量反转)
        idx_3d_mean = cross_panic_idx.rolling(3, min_periods=1).mean()
        is_exhausted_down = cross_panic_idx < idx_3d_mean
        is_exhausted_up = cross_panic_idx > idx_3d_mean
        
        # 脉冲触发核心逻辑：极值 + 衰竭
        # 多头：恐慌因呈右偏肥尾特性，故极值门槛设为严格的 Z > 2.0
        # 空头：自满往往是长期的极度低波(具有天然下界)，故下探门槛设为 Z < -1.5
        long_cond = (cross_panic_z > 2.0) & is_exhausted_down
        short_cond = (cross_panic_z < -1.5) & is_exhausted_up
        
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"