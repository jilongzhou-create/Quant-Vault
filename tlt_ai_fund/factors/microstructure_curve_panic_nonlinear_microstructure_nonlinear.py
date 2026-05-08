import numpy as np
import pandas as pd

class MicrostructureCurvePanicNonlinearFactor:
    """微观曲线动量与恐慌交叉因子 (microstructure/nonlinear)

    逻辑: 结合收益率曲线的短期动量与股市恐慌情绪。在宏观危机爆发且曲线急剧牛陡时(短端骤降爆发Pivot预期)，单纯追高容易死于抛售波动，必须等待 VIX 恐慌边际衰竭瞬间抄底美债；相反，在极端自满且曲线急剧熊平倒挂时(紧缩冲击潜伏)，等待波动率抬头瞬间做空美债。脉冲型触发避免了常态持仓带来的钝化。
    数据: vixcls, t10y2y
    触发: VIX与T10Y2Y边际动量的Z-Score同向非线性乘积 > 3.0，并且伴随VIX的二阶导数衰竭/反转(当前值穿越3日均值)
    输出: 极短期脉冲信号，买入+1.0，卖出-1.0，其余常态非触发日严格为 0.0
    """

    def __init__(self):
        self.name = 'microstructure_curve_panic_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (常态严格为 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 缺失列直接返回0.0
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 铁律3: 边际变化 (只关注收益率曲线的短期骤变动量，禁止看绝对水位)
        # 5日差分捕捉机构周级别的紧急调仓带来的微观陡峭化/平坦化冲击
        t10y2y_mom = t10y2y.diff(5)
        
        # 计算 126 日(半年) 滚动 Z-Score 衡量微观极端偏离度
        vix_mean = vix.rolling(126).mean()
        vix_std = vix.rolling(126).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        mom_mean = t10y2y_mom.rolling(126).mean()
        mom_std = t10y2y_mom.rolling(126).std().replace(0, np.nan)
        mom_z = (t10y2y_mom - mom_mean) / mom_std
        
        # 非线性交叉1: 恐慌共振牛陡 (VIX极高且急剧恶化 + 曲线剧烈变陡)
        bull_steepening_panic = pd.Series(
            np.where((vix_z > 0) & (mom_z > 0), vix_z * mom_z, 0.0), 
            index=data.index
        )
        
        # 非线性交叉2: 自满共振熊平 (VIX低迷且过度安全 + 曲线剧烈倒挂/平坦化)
        bear_flattening_complacency = pd.Series(
            np.where((vix_z < 0) & (mom_z < 0), np.abs(vix_z * mom_z), 0.0), 
            index=data.index
        )
        
        # 极值条件: 乘积 > 3.0 (相当于两个维度同时超过 1.73 倍标准差的极值区域)
        buy_extreme = bull_steepening_panic > 3.0
        sell_extreme = bear_flattening_complacency > 3.0
        
        # 铁律2: 二阶导数 (极端状态必须伴随边际衰竭/反转才触发，绝对禁止接飞刀)
        vix_roll3 = vix.rolling(3).mean()
        vix_exhausted = vix < vix_roll3  # 恐慌开始高位回落
        vix_spiking = vix > vix_roll3    # 波动率开始低位抬头
        
        # 组合脉冲触发条件: 极值 + 衰竭反转
        buy_trigger = buy_extreme & vix_exhausted
        sell_trigger = sell_extreme & vix_spiking
        
        # 赋值脉冲信号
        signal[buy_trigger] = 1.0
        signal[sell_trigger] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"