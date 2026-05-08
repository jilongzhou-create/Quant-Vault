import numpy as np
import pandas as pd

class FundingMicrostructureExhaustionFactor:
    """资金面与微观流动性共振衰竭因子 (microstructure/nonlinear)

    逻辑: 结合 FICC 资金面(DFF与DTB3的避险抵押品溢价)与 TLT 微观成交量(Volume)构建非线性冲击指数。当市场发生严重流动性危机时，资金疯抢短债导致 T-Bill 溢价飙升，且 ETF 爆量踩踏。当这种抵押品恐慌和微观放量同步达到极值并衰竭时，标志着流动性冲击结束，资金将重新回流，提供绝佳的反转脉冲。
    数据: dff (联邦基金利率), dtb3 (3个月美债收益率), volume (TLT成交量), close (价格)
    触发: (T-Bill溢价 Z-Score + Volume Z-Score) > 2.5 且两者均 > 1.0，并伴随交叉指数回落至3日均值以下(二阶衰竭)。根据前期价格动量决定做多或做空。
    输出: 狙击型脉冲信号 [-1.0, 1.0]，常态严格为 0.0。
    """

    def __init__(self):
        self.name = 'funding_microstructure_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查 FICC 资金面数据与微观价格/成交量是否存在
        required_cols = ['dff', 'dtb3', 'volume', 'close']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 1. 数据清洗与前向填充 (防止对齐产生 NaN)
        dff = data['dff'].ffill()
        dtb3 = data['dtb3'].ffill()
        vol = data['volume'].ffill()
        close = data['close'].ffill()
        
        # 2. 资金面特征：避险抵押品溢价 (T-Bill Premium)
        # 正常情况下 DFF 与 DTB3 接近。当市场恐慌时，极度缺乏安全抵押品，资金疯狂买入T-Bill，导致 dtb3 暴跌，溢价飙升
        t_bill_premium = dff - dtb3
        tb_mean = t_bill_premium.rolling(126, min_periods=21).mean()
        tb_std = t_bill_premium.rolling(126, min_periods=21).std()
        tb_zscore = (t_bill_premium - tb_mean) / tb_std.replace(0, np.nan)
        
        # 3. 微观特征：ETF 极端放量
        vol_mean = vol.rolling(126, min_periods=21).mean()
        vol_std = vol.rolling(126, min_periods=21).std()
        vol_zscore = (vol - vol_mean) / vol_std.replace(0, np.nan)
        
        # 填充NaN以防单边数据缺失导致整个序列计算中断
        tb_zscore = tb_zscore.fillna(0)
        vol_zscore = vol_zscore.fillna(0)
        
        # 4. 非线性交叉特征：资金面与微观放量共振冲击指数
        # 两个标准化指标相加，构建高维交叉度量
        cross_shock = tb_zscore + vol_zscore
        
        # 5. 铁律1: 零值休眠 & 极值触发
        # 要求总冲击指数 > 2.5，且两个维度必须都处于偏高状态 (> 1.0)，确保是双维度的真正共振
        is_extreme = (cross_shock > 2.5) & (tb_zscore > 1.0) & (vol_zscore > 1.0)
        
        # 6. 铁律2: 二阶导数衰竭 (Anti-Catch-Falling-Knife)
        # 冲击拐点：交叉冲击指数开始回落，且边际动量为负
        shock_ma3 = cross_shock.rolling(3).mean()
        is_exhausting = (cross_shock < shock_ma3) & (cross_shock.diff() < 0)
        
        # 7. 价格动量与反转方向判定
        # 区分该次冲击是抛售导致的流动性干涸，还是避险导致的疯狂抢筹
        price_ma10 = close.rolling(10).mean()
        is_oversold = close < price_ma10   # 冲击期间被抛售 (收益率飙升)，衰竭后做多美债
        is_overbought = close > price_ma10 # 冲击期间被抢购 (收益率暴跌)，衰竭后做空美债
        
        # 8. 生成脉冲信号 (结合所有逻辑)
        trigger_long = (is_extreme & is_exhausting & is_oversold).fillna(False)
        trigger_short = (is_extreme & is_exhausting & is_overbought).fillna(False)
        
        signal.loc[trigger_long] = 1.0
        signal.loc[trigger_short] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"