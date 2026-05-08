import numpy as np
import pandas as pd

class RateEquityVolDivergenceFactor:
    """Rate vs Equity Volatility Divergence (volatility/options)

    逻辑: 比较短期利率市场(dgs2)与期权市场(VIX)的相对波动率。当利率恐慌极度高于股市恐慌(Z-Score > 1.5，如加息周期尾声)并开始衰竭时，债市见底，脉冲做多TLT；当股市期权恐慌极度高于利率恐慌(Z-Score < -1.5，如黑天鹅崩盘极值)并开始衰竭时，避险情绪消退，脉冲做空TLT。脉冲设计能精准捕捉不同恐慌主导环境下的情绪极值反转瞬间。
    数据: dgs2 (2年期国债收益率, 捕捉债市波动), vixcls (VIX隐含波动率, 捕捉股市波动)
    触发: 相对波动率对数 Z-Score > 1.5 且跌破3日均线 -> +1.0; Z-Score < -1.5 且突破3日均线 -> -1.0
    输出: 脉冲型信号 [-1.0, 1.0] (控制 Trigger Rate 在 5% - 15% 目标区间)
    """

    def __init__(self):
        self.name = 'rate_equity_vol_divergence'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 常态下信号必须严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需卫星数据
        if 'vixcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        # VIX常态下在9-80之间波动，设底防止异常数据导致的除0
        vix = vix.replace(0, np.nan).ffill().clip(lower=5.0) 
        
        # 获取利率代理变量 (优先使用高频且非CoreAnchor限制的 dgs2，备用 t10y2y)
        if 'dgs2' in data.columns:
            rate_proxy = data['dgs2'].ffill()
        elif 't10y2y' in data.columns:
            rate_proxy = data['t10y2y'].ffill()
        else:
            return signal
            
        # 计算利率的短期微观波动率 (20日一阶差分的标准差)
        rate_vol = rate_proxy.diff().rolling(window=20).std()
        rate_vol = rate_vol.replace(0, np.nan).ffill().clip(lower=1e-6)
        
        # 构建核心跨域指标：利率与股市相对恐慌比率
        vol_ratio = rate_vol / vix
        
        # 对数化处理，压缩极值，使得分布更接近正态，提高Z-score的统计学有效性
        log_vol_ratio = np.log(vol_ratio)
        
        # 计算252日滚动Z-Score (衡量宏观波动率偏离度)
        roll_mean = log_vol_ratio.rolling(window=252).mean()
        roll_std = log_vol_ratio.rolling(window=252).std()
        
        # 避免除0问题
        roll_std = roll_std.replace(0, np.nan).ffill().clip(lower=1e-6)
        zscore = (log_vol_ratio - roll_mean) / roll_std
        
        # 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 计算3日均值作为衰竭判定基准，避免单日噪音
        ma3 = log_vol_ratio.rolling(window=3).mean()
        
        # 触发条件 A: 债市主导的恐慌极值 + 恐慌开始衰竭 -> 债券见底，脉冲做多
        # 阈值使用 1.5 以保证单侧大概 6.6% 的触发率，结合双尾保证总 Trigger Rate 落在 5%-15%
        cond_long = (zscore > 1.5) & (log_vol_ratio < ma3)
        
        # 触发条件 B: 股市主导的恐慌极值 + 恐慌开始衰竭 -> 避险盘解体，资金回流风险资产，脉冲做空
        cond_short = (zscore < -1.5) & (log_vol_ratio > ma3)
        
        # 赋值脉冲信号
        signal[cond_long] = 1.0
        signal[cond_short] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"