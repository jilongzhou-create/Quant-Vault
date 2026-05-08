import numpy as np
import pandas as pd

class SyntheticMovePulseFactor:
    """合成MOVE恐慌反转因子 (volatility/options)

    逻辑: 由于FRED缺少债市VIX(MOVE)指数，本因子使用股市 VIX (期权隐含波动) 与 BBB级信用利差(OAS)波动率 合成跨资产恐慌极值。当极度飙升且开始衰竭时，标志流动性无差别抛售结束，美债迎来避险买盘(脉冲看多)；当长期极端低波环境瞬间被打破时看空。
    数据: vixcls (VIX), bamlc0a4cbbb (BBB OAS)
    触发: 多头 -> 合成恐慌 Z-Score > 2.5 且双指标回落低于3日均值；空头 -> 极度低波 Z-Score < -1.5 且开始共振突破均值。
    输出: 严格遵循二阶导数衰竭的极值买卖脉冲 (+1.0 / -1.0)，非触发期休眠 (0.0)。
    """

    def __init__(self):
        self.name = 'synthetic_move_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['vixcls', 'bamlc0a4cbbb']
        if not all(col in data.columns for col in req_cols):
            return signal
            
        # 前向填充处理非交易日缺失值
        vix = data['vixcls'].ffill()
        bbb_oas = data['bamlc0a4cbbb'].ffill()
        
        # 铁律3: 边际变化 - 获取信用利差的边际波动率 (20日diff标准差，捕捉信贷恐慌)
        credit_vol = bbb_oas.diff().rolling(20).std()
        
        # 计算一年期 (252个交易日) 的极值 Z-Score
        vix_z = (vix - vix.rolling(252).mean()) / vix.rolling(252).std()
        credit_vol_z = (credit_vol - credit_vol.rolling(252).mean()) / credit_vol.rolling(252).std()
        
        # 合成跨资产流动性恐慌指数 (等权相加)
        syn_move_z = (vix_z + credit_vol_z) / 2.0
        
        # 铁律2: 二阶导数衰竭条件 - 绝不接飞刀，必须看到狂热回落
        vix_falling = vix < vix.rolling(3).mean()
        credit_vol_falling = credit_vol < credit_vol.rolling(3).mean()
        
        vix_rising = (vix > vix.rolling(3).mean()) & (vix.diff() > 0)
        credit_vol_rising = (credit_vol > credit_vol.rolling(3).mean()) & (credit_vol.diff() > 0)
        
        # 铁律1: 零值休眠触发逻辑
        # 多头: 流动性冲击极度拥挤且开始瓦解 (抛售衰竭 -> 资金重回安全资产美债 TLT)
        long_cond = (syn_move_z > 2.5) & vix_falling & credit_vol_falling
        
        # 空头: 长期"金发女孩"死水期被打破 (低波突变高波 -> 流动性收紧/加息重燃, 抛售美债)
        short_cond = (syn_move_z < -1.5) & vix_rising & credit_vol_rising
        
        # 赋值触发区脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"