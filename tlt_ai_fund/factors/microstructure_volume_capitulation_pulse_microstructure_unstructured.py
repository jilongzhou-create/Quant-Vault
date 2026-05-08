import numpy as np
import pandas as pd

class MicrostructureVolumeCapitulationPulseFactor:
    """TLT成交量恐慌衰竭脉冲因子 (microstructure/unstructured)

    逻辑: 捕捉微观流动性冲击(Volume Spike)与动能衰竭(Volume Fade)。当资产经历剧烈单边行情后，极端的成交量脉冲代表主力的集中止损或FOMO(Capitulation)。严格遵守二阶导数铁律：绝不在极值爆量当天接飞刀，而是等待成交量开始急速萎缩确认流动性冲击耗尽时，才输出精准的反转脉冲。
    数据: 仅依赖传入自身行情数据(market_data_tlt)的 volume 与 close，纯粹微观结构，零跨域过滤。
    触发: 
      - 条件1 (极值): 昨日成交量的 120日 Z-Score > 2.5 (恐慌爆量)
      - 条件2 (衰竭): 今日成交量 < 过去3日均量 (动能消退，抛压/买盘衰竭)
      - 条件3 (边际): 过去10日收益率绝对值 > 2% (确认处于短期趋势极端)
    输出: 狙击手级脉冲。抛售高潮衰竭输出 +1.0 (看多抄底)；买入高潮衰竭输出 -1.0 (看空滞涨)。非触发日严格输出 0.0。
    """

    def __init__(self):
        self.name = 'microstructure_volume_capitulation_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始信号全为0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需字段 (容错处理)
        if not {'volume', 'close'}.issubset(data.columns):
            return signal
            
        vol = data['volume'].copy()
        close = data['close'].copy()
        
        # 清洗异常值，防止除以零
        vol = vol.replace(0, np.nan).fillna(method='ffill')
        
        # 1. 提取微观流动性极值 (边际变化铁律)
        # 使用 120日(约半年) 滚动窗口计算 Z-Score，捕捉罕见的机构止损/抢筹脉冲
        vol_ma = vol.rolling(window=120, min_periods=40).mean()
        vol_std = vol.rolling(window=120, min_periods=40).std()
        vol_zscore = (vol - vol_ma) / (vol_std + 1e-8)
        
        # 2. 短期趋势动能提取 (辅助确认当前处于下跌抛售还是上涨FOMO)
        # 关注10个交易日的边际变化率
        price_momentum = close.pct_change(10)
        
        # 3. 构造触发条件 (反接飞刀铁律：必须同时满足 极值 + 衰竭)
        # 条件1: 昨日爆出天量 (流动性冲击极值)
        extreme_volume_yesterday = vol_zscore.shift(1) > 2.5
        
        # 条件2: 今日迅速缩量 (动能衰竭，低于短线3日均值)
        vol_exhaustion_today = vol < vol.rolling(window=3).mean()
        
        # 4. 生成多空狙击脉冲
        # 看多脉冲 (+1.0): 处于短线急跌中，爆出天量后抛压耗尽 (Selling Climax)
        selling_climax = price_momentum.shift(1) < -0.02
        bull_pulse = extreme_volume_yesterday & vol_exhaustion_today & selling_climax
        
        # 看空脉冲 (-1.0): 处于短线急涨中，爆出天量后买盘耗尽 (Buying Climax)
        buying_climax = price_momentum.shift(1) > 0.02
        bear_pulse = extreme_volume_yesterday & vol_exhaustion_today & buying_climax
        
        # 赋值信号
        signal[bull_pulse] = 1.0
        signal[bear_pulse] = -1.0
        
        # 清理可能因 rolling 产生的 NaN
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"