import numpy as np
import pandas as pd

class CrossAssetVolatilityReversalFactor:
    """跨资产波动率极值与拥挤反转因子 (volatility/options)

    逻辑: 监控期权隐含波动率极值，捕捉对冲盘极度拥挤及瓦解时刻，因子为严格零值休眠脉冲。
          恐慌极值瓦解(VIX极高并回落): 流动性危机解除，避险资金重新买入被错杀的美债，脉冲看多TLT；
          贪婪极值瓦解(VIX极低并抬升): 风险平价基金(Risk Parity)被动降杠杆，引发股债双杀无差别抛售，脉冲看空TLT。
    数据: vixcls (标普500 VIX), gvzcls (黄金 ETF 隐含波动率)
    触发: 多头 -> VIX 63日(单季度) Z-Score > 2.5 且 VIX开始回落(小于3日均值且diff<0) 且 GVZ同步回落确认跨资产恐慌衰竭；
          空头 -> VIX 63日(单季度) Z-Score < -2.0 且 VIX开始抬升(大于3日均值且diff>0) 且 GVZ同步抬升确认风险重燃。
    输出: 仅在衰竭反转节点输出脉冲信号 (+1.0 / -1.0)，其余常态时间严格输出休眠值 (0.0)。
    """

    def __init__(self):
        self.name = 'cross_asset_volatility_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始信号必须为 pd.Series(0.0), 遵守零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 处理缺少必需字段的情况
        if 'vixcls' not in data.columns or 'gvzcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        
        # 计算 63 个交易日 (单季度) 的滚动均值和标准差，获取统计学极限值
        vix_mean = vix.rolling(window=63).mean()
        vix_std = vix.rolling(window=63).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        # 计算二阶导数与边际变化: 捕捉衰竭与爆发瞬间
        vix_diff = vix.diff()
        vix_ma3 = vix.rolling(window=3).mean()
        gvz_diff = gvz.diff()
        
        # 遵守二阶导数铁律: 绝对禁止 'VIX > 某值 -> 直接买入'，必须叠加衰竭条件！
        
        # 多头触发 (恐慌瓦解): 波动率极高 + 见顶回落
        buy_cond = (vix_z > 2.5) & (vix_diff < 0) & (vix < vix_ma3) & (gvz_diff <= 0)
        
        # 空头触发 (贪婪瓦解): 波动率极低 + 抬头飙升
        sell_cond = (vix_z < -2.0) & (vix_diff > 0) & (vix > vix_ma3) & (gvz_diff >= 0)
        
        # 赋值脉冲信号
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"