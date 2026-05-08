import numpy as np
import pandas as pd

class MicrostructureOrderFlowPulseFactor:
    """微观结构订单流极值衰竭因子 (microstructure/unstructured)

    逻辑: 恐慌抛售(或逼空)会导致微观订单流(Order Flow)产生极端的单向脉冲，流动性枯竭引发避险资产的暂时性错杀（如2020年3月流动性危机）。因子首先捕捉半年内罕见(Z-score>2.0)的订单流失衡冲击，当这股力量开始衰竭(成交量收敛至3日均值以下，且抛压绝对动能相比昨日减轻)的瞬间，表示流动性真空被修复、拐点已现，此时触发 Sniper 级别的买入/卖出反转脉冲。因子严格遵循零值休眠与二阶导数防接飞刀。
    数据: close, volume (通过价量合成无结构订单流)。
    触发: 近3日订单流发生 Z-Score 极值 (>=2.0 或 <=-2.0) AND 今日成交量 < 3日均值 AND 订单流较昨日边际反向收敛。
    输出: +1.0 看多脉冲(抛压耗尽抄底), -1.0 看空脉冲(买盘耗尽逃顶)。无极值衰竭共振状态严格返回 0.0。
    """

    def __init__(self, zscore_window=126, zscore_threshold=2.0, exhaust_window=3):
        self.name = 'microstructure_order_flow_pulse'
        # 126日对应半年的微观流动性基准
        self.zscore_window = zscore_window
        # 2.0对应统计学中大约单尾 2.27% 概率的流动性异常冲击
        self.threshold = zscore_threshold
        # 短期发酵平息周期，代表情绪退潮的时间窗口
        self.exhaust_window = exhaust_window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 - 初始化全为 0.0 的狙击手脉冲信号
        signal = pd.Series(0.0, index=data.index)
        
        # 处理数据缺失情况
        if 'close' not in data.columns or 'volume' not in data.columns:
            return signal
            
        close = data['close'].ffill()
        volume = data['volume'].fillna(0.0)
        
        if volume.sum() == 0:
            return signal

        # 铁律3: 边际变化 - 提取非结构化订单流失衡(Order Flow Imbalance)作为微观特征
        price_diff = close.diff().fillna(0.0)
        direction = np.sign(price_diff)
        ofi = direction * volume
        
        # 计算订单流不平衡的滚动 Z-Score 
        ofi_mean = ofi.rolling(window=self.zscore_window, min_periods=10).mean()
        ofi_std = ofi.rolling(window=self.zscore_window, min_periods=10).std()
        
        # 避免分母为0
        ofi_zscore = (ofi - ofi_mean) / ofi_std.replace(0, np.nan)
        ofi_zscore = ofi_zscore.fillna(0.0)
        
        # 铁律2: 二阶导数 (极值条件) - 过去N天内发生过极端单向流动性冲击
        extreme_sell_shock = ofi_zscore.rolling(self.exhaust_window).min() <= -self.threshold
        extreme_buy_shock = ofi_zscore.rolling(self.exhaust_window).max() >= self.threshold
        
        # 铁律2: 二阶导数 (衰竭条件) - 交投清淡且边际动能反转，代表主跌/主升浪已经结束
        vol_mean = volume.rolling(self.exhaust_window).mean()
        vol_exhausted = volume < vol_mean
        
        # 边际动能反转: 当天的净资金流向相比昨天有实质性逆向改善
        sell_pressure_easing = ofi > ofi.shift(1).fillna(0.0)
        buy_pressure_easing = ofi < ofi.shift(1).fillna(0.0)
        
        # 组合判定：同时满足 极端冲击 + 动能逆转改善 + 量能萎缩
        long_pulse = extreme_sell_shock & sell_pressure_easing & vol_exhausted
        short_pulse = extreme_buy_shock & buy_pressure_easing & vol_exhausted
        
        # 赋值并防单日重叠互斥
        signal[long_pulse] = 1.0
        signal[short_pulse] = -1.0
        signal[long_pulse & short_pulse] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"MicrostructureOrderFlowPulseFactor(zscore_window={self.zscore_window}, threshold={self.threshold}, exhaust_window={self.exhaust_window})"