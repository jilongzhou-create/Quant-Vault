import numpy as np
import pandas as pd

class DovishLiquidityShockFactor:
    """因子名称 (policy_pivot/nonlinear)

    逻辑: 捕捉美联储政策预期突变(流动性冲击)与信用市场确认的非线性交叉。当短端利率(2年期)在极短时间内暴跌(市场抢跑降息), 且高收益债信用利差同时收窄时, 说明是"软着陆/防御性降息"的纯流动性利好, 而非衰退恐慌(防接飞刀), 触发强烈看多脉冲。反之, 利率飙升且信用恶化触发看空脉冲。
    数据: [dgs2, bamlh0a0hym2]
    输出: +1.0 (鸽派流动性喷发, 看多), -1.0 (鹰派冲击且信用恶化, 看空), 0.0 (常态休眠)
    触发条件: 2年期国债收益率5日动量的Z-Score超过极端阈值(±1.2), 且被高收益债利差的边际变化方向交叉确认。预期 Trigger Rate 约 8% - 12%。
    """

    def __init__(self):
        self.name = 'dovish_liquidity_shock'
        self.momentum_window = 5
        self.zscore_window = 252
        self.z_threshold = 1.2
        self.credit_deterioration_threshold = 0.05 # 信用恶化的微小确认阈值 (5 bps)

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 信号
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必需的数据字段是否存在
        required_cols = ['dgs2', 'bamlh0a0hym2']
        if not all(col in data.columns for col in required_cols):
            return signal

        # 提取并前向填充宏观数据以处理节假日缺失值
        df = data[required_cols].ffill()

        # 1. 计算短端利率 (2年期) 的脉冲动量
        # 经济学含义: 5天内的快速下行代表市场极度抢跑鸽派预期(变陡)，快速上行代表鹰派惊吓
        dgs2_chg = df['dgs2'].diff(self.momentum_window)
        
        # 计算利率动量的滚动 Z-Score (防未来函数)
        dgs2_chg_mean = dgs2_chg.rolling(window=self.zscore_window, min_periods=60).mean()
        dgs2_chg_std = dgs2_chg.rolling(window=self.zscore_window, min_periods=60).std()
        dgs2_zscore = (dgs2_chg - dgs2_chg_mean) / dgs2_chg_std

        # 2. 计算信用利差的边际变化 (二阶导数防飞刀铁律)
        # 经济学含义: 如果利率暴跌是因为史诗级股灾/衰退(利差暴涨)，绝对不能买。只有利差稳定/收窄，才是真正的流动性牛市。
        credit_chg = df['bamlh0a0hym2'].diff(self.momentum_window)

        # 3. 脉冲触发逻辑 (非线性特征交叉)
        # 看多脉冲 (+1.0): 市场疯狂定价降息 (dgs2急跌) AND 信用环境向好/无恐慌 (利差未走阔)
        long_condition = (dgs2_zscore < -self.z_threshold) & (credit_chg <= 0.0)
        
        # 看空脉冲 (-1.0): 突发鹰派紧缩 (dgs2急升) AND 信用环境实质性恶化 (趋势破坏)
        short_condition = (dgs2_zscore > self.z_threshold) & (credit_chg > self.credit_deterioration_threshold)

        # 赋值信号
        signal.loc[long_condition] = 1.0
        signal.loc[short_condition] = -1.0

        # 清理由于均值方差窗口期产生的初始NaN值
        signal = signal.fillna(0.0)

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(z_threshold={self.z_threshold}, window={self.momentum_window})"