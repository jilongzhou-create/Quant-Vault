import numpy as np
import pandas as pd

class DovishLiquidityPulseFactor:
    """政策转向与流动性冲量 (policy_pivot/nonlinear)

    逻辑: 捕捉美联储预期剧变的"金发姑娘"时刻或流动性紧缩时刻。当短端利率(DGS2)急剧下行(市场抢跑降息)，且高收益债利差未处于危机状态并伴随收窄时，说明是优质的宽松预期，触发做多脉冲；当短端利率急速上升且信用利差边际恶化时，引发"股债双杀"担忧，触发做空脉冲。
    数据: dgs2 (2年期美债收益率), bamlh0a0hym2 (高收益债信用利差)
    输出: 1.0 (鸽派转向驱动看多), -1.0 (鹰派紧缩驱动看空), 0.0 (休眠)
    触发条件: DGS2的5日变化与信用利差动量及Z-Score的非线性交叉。脉冲持续3天，预期 Trigger Rate 8% - 12%。
    """

    def __init__(self):
        self.name = 'dovish_liquidity_pulse'
        # 参数设定: 经济学含义明确
        self.dgs2_lookback = 5        # 观察短端利率预期的短窗口 (1周)
        self.hy_lookback = 3          # 观察信用利差变化的极短窗口
        self.zscore_window = 252      # 过去一年的宏观水位评估
        
        self.rate_drop_bps = -0.15    # 降息抢跑阈值: 短端急降 15个基点
        self.rate_hike_bps = 0.15     # 加息抢跑阈值: 短端急升 15个基点
        self.hy_deteriorate_bps = 0.05 # 信用恶化: 利差短期上升 5个基点

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 缺失必要数据时返回0值休眠信号
        if 'dgs2' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 数据清洗: 避免NaN影响运算
        dgs2 = data['dgs2'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()

        # -------------------------------------------------------------
        # 计算特征变量 (基于边际变化与二阶导数法则)
        # -------------------------------------------------------------
        # 1. 政策利率预期动量
        dgs2_diff = dgs2.diff(self.dgs2_lookback)
        
        # 2. 信用环境边际变化
        hy_diff = hy_spread.diff(self.hy_lookback)
        
        # 3. 信用环境历史水位评估 (绝对极值规避飞刀)
        hy_roll_mean = hy_spread.rolling(self.zscore_window).mean()
        hy_roll_std = hy_spread.rolling(self.zscore_window).std()
        hy_zscore = (hy_spread - hy_roll_mean) / (hy_roll_std + 1e-6)

        # -------------------------------------------------------------
        # 触发逻辑
        # -------------------------------------------------------------
        # 看多脉冲: 短端抢跑降息预期 + 信用市场未处于深度恐慌极值(<1.0) + 且信用利差收窄(未发生危机恶化)
        long_trigger = (
            (dgs2_diff <= self.rate_drop_bps) & 
            (hy_diff < 0.0) & 
            (hy_zscore < 1.0)
        )

        # 看空脉冲: 短端鹰派超预期紧缩 + 信用市场确立恶化边际变化(>5bps) + 未处于极度宽松泡沫垫中(>-1.0)
        short_trigger = (
            (dgs2_diff >= self.rate_hike_bps) & 
            (hy_diff > self.hy_deteriorate_bps) & 
            (hy_zscore > -1.0)
        )

        # -------------------------------------------------------------
        # 信号合成与脉冲延伸
        # -------------------------------------------------------------
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal.loc[long_trigger] = 1.0
        raw_signal.loc[short_trigger] = -1.0

        # 为了满足5%~15%的触发率，将瞬时脉冲向后延伸2个交易日(共存活3天)
        # 代表市场需要极短的几天时间消化政策跳跃冲量
        signal = raw_signal.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(dgs2_window={self.dgs2_lookback}, hy_window={self.hy_lookback})"