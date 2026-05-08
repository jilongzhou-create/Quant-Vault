import numpy as np
import pandas as pd

class MicrostructureCollateralExhaustionNonlinearFactor:
    """微观流动性与抵押品恐慌衰竭 (microstructure/nonlinear)

    逻辑: 纯微观结构域因子，严格遵守跨域隔离铁律。使用 DFF (联邦基金利率) 与 DTB3 (3个月美债) 的利差来衡量底层抵押品短缺与 Dash-for-Cash 恐慌。在危机爆发时，短端优质抵押品(T-Bill)被疯抢导致收益率暴跌，DFF-DTB3利差极端飙升。当该利差(Z-Score > 2.5)且微观抛售爆出天量(Volume Z-Score > 2.5)，随后两者同步回落时，标志着微观流动性危机(如2020年3月)见顶，抛压枯竭，美债迎来强力脉冲反弹。
    数据: dff, dtb3, volume
    触发: (DFF - DTB3) Z-Score > 2.5 且开始回落 + Volume Z-Score > 2.5 且开始回落
    输出: +1.0 表示微观结构流动性挤兑衰竭，狙击式看多美债(TLT)
    """

    def __init__(self, spread_z_thresh=2.5, vol_z_thresh=2.5):
        self.name = 'microstructure_collateral_exhaustion_nonlinear'
        self.spread_z_thresh = spread_z_thresh
        self.vol_z_thresh = vol_z_thresh

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (常态下输出 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需字段 (纯微观结构数据)
        req_cols = ['dff', 'dtb3', 'volume']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 填充缺失值，避免因节假日导致计算中断
        dff = data['dff'].ffill()
        dtb3 = data['dtb3'].ffill()
        vol = data['volume'].ffill()
        
        # 1. 构建微观结构抵押品压力利差 (Collateral Stress Spread)
        spread = dff - dtb3
        
        # 计算 252 日滚动 Z-Score 捕捉极端脉冲
        spread_mean = spread.rolling(window=252, min_periods=63).mean()
        spread_std = spread.rolling(window=252, min_periods=63).std()
        spread_z = (spread - spread_mean) / spread_std.replace(0, np.nan)
        
        # 计算 63 日 Volume 滚动 Z-Score 捕捉微观爆量
        vol_mean = vol.rolling(window=63, min_periods=21).mean()
        vol_std = vol.rolling(window=63, min_periods=21).std()
        vol_z = (vol - vol_mean) / vol_std.replace(0, np.nan)
        
        # 2. 极端水位条件 (满足 Z-Score > 2.5)
        spread_extreme = spread_z > self.spread_z_thresh
        # 允许微观抛量在过去3天内发生即可，防止天量与情绪高点差一两天
        vol_extreme = vol_z.rolling(window=3, min_periods=1).max() > self.vol_z_thresh
        
        # 3. 铁律2 & 3: 二阶导数与边际变化 (Anti-Catch-Falling-Knife)
        # 抵押品利差不再恶化且低于3日均值 (流动性挤兑开始缓解)
        spread_exhaustion = (spread.diff() < 0) & (spread < spread.rolling(window=3, min_periods=1).mean())
        # 成交量必须从天量高位开始萎缩 (微观抛压真正枯竭)
        vol_exhaustion = (vol.diff() < 0) & (vol < vol.rolling(window=3, min_periods=1).mean())
        
        # 4. 非线性特征交叉触发
        # 两个微观维度同时满足：流动性抵押品挤兑衰竭 + 微观交易抛压衰竭
        trigger = spread_extreme & spread_exhaustion & vol_extreme & vol_exhaustion
        
        # 输出脉冲信号
        signal[trigger] = 1.0
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(spread_z={self.spread_z_thresh}, vol_z={self.vol_z_thresh})"