import numpy as np
import pandas as pd

class CreditVolCapitulationCrossFactor:
    """恐慌极值与均值回归 (panic_mean_reversion/nonlinear)

    逻辑: 跨维交叉验证恐慌极值。当美股隐含波动率(VIX)与高收益债信用利差(HY OAS)双双达到历史高位(机构与散户共同恐慌), 且双边动量同步由正转负时, 确认恐慌衰竭(Capitulation), 触发抄底买入。反之, 若两者同步急剧飙升且未达极值, 说明新一轮抛售刚刚开始, 触发看空。
    数据: vixcls (VIX指数), bamlh0a0hym2 (美银美林美国高收益债期权调整利差)
    输出: +1.0 表示恐慌衰竭/见底回升 (强烈看多); -1.0 表示恐慌爆发初期 (看空); 0.0 为常态
    触发条件: 多头脉冲=双Z-Score>1.0且二阶导数<0; 空头脉冲=短期急剧飙升且Z-Score<1.0。预期 Trigger Rate 控制在 8% - 12% 之间。
    """

    def __init__(self, window=252, z_threshold=1.0, short_vix_spike=0.15):
        self.name = 'credit_vol_capitulation_cross'
        self.window = window
        self.z_threshold = z_threshold
        self.short_vix_spike = short_vix_spike

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 检查必要数据字段是否存在
        required_cols = ['vixcls', 'bamlh0a0hym2']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0.0, index=data.index, name=self.name)

        # 2. 提取数据并处理缺失值 (前向填充)
        vix = data['vixcls'].ffill()
        spread = data['bamlh0a0hym2'].ffill()

        # 3. 计算 252 日 (约一年) Z-Score 识别极值状态
        vix_mean = vix.rolling(window=self.window, min_periods=self.window//2).mean()
        vix_std = vix.rolling(window=self.window, min_periods=self.window//2).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)

        spread_mean = spread.rolling(window=self.window, min_periods=self.window//2).mean()
        spread_std = spread.rolling(window=self.window, min_periods=self.window//2).std()
        spread_z = (spread - spread_mean) / (spread_std + 1e-8)

        # 4. 计算边际变化 (动量与二阶导数)
        # VIX 反应极快, 看单日回落; 信用利差偏宏观, 看3日趋势转折
        vix_diff_1 = vix.diff(1)
        spread_diff_3 = spread.diff(3)
        vix_pct_change_3 = vix.diff(3) / (vix.shift(3) + 1e-8)

        # 5. 脉冲信号逻辑构建
        
        # 多头信号 (+1.0): 极度恐慌 + 衰竭确认 (防接飞刀)
        # 条件: 波动率与信用利差均处于高位 (Z > 1.0), 且今日 VIX 回落, 且利差近3日停止走阔并开始收窄
        long_cond = (vix_z > self.z_threshold) & \
                    (spread_z > self.z_threshold) & \
                    (vix_diff_1 < 0.0) & \
                    (spread_diff_3 < 0.0)

        # 空头信号 (-1.0): 钝刀割肉 / 恐慌爆发初期
        # 条件: VIX 在3天内飙升超过 15%, 且信用利差同步走阔, 但整体 VIX Z-Score 尚未达到极端水平 (< 1.0, 意味着还有下跌空间)
        short_cond = (vix_pct_change_3 > self.short_vix_spike) & \
                     (spread_diff_3 > 0.0) & \
                     (vix_z < self.z_threshold) & \
                     (spread_z > -1.0) # 排除在极度宽松环境下的微小波动噪声

        # 6. 生成脉冲信号
        signal = pd.Series(0.0, index=data.index)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        # 清理由于 rolling 造成的初期 NaN
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold})"