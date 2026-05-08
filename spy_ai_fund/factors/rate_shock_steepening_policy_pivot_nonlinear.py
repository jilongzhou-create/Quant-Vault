import numpy as np
import pandas as pd

class RateShockSteepeningFactor:
    """Rate Shock & Bull Steepening Factor (policy_pivot/nonlinear)

    逻辑: 捕捉美联储预期剧变的脉冲时刻。看多：2年期美债收益率极速暴跌(市场抢跑降息预期)导致收益率曲线瞬间急剧变陡(Bull Steepening)；看空：2年期极速飙升导致曲线急剧倒挂(Bear Flattening的鹰派冲击)。
    数据: dgs2 (2年期国债收益率), t10y2y (10年-2年期限利差)
    输出: +1.0 (鸽派突变看多), -1.0 (鹰派恐慌看空), 0.0 (常态)
    触发条件: dgs2和t10y2y的5日动量同时突破126日滚动Z-Score的1.5倍极值, 且绝对变化超过10个基点。预期 Trigger Rate 5%-10%
    """

    def __init__(self, mom_window=5, z_window=126, z_threshold=1.5, min_rate_change=0.10, min_spread_change=0.05):
        self.name = 'rate_shock_steepening_factor'
        self.mom_window = mom_window        # 测量政策预期剧变的极短窗口
        self.z_window = z_window            # 约半年交易日，反映中期宏观波动率基准
        self.z_threshold = z_threshold      # 1.5 约对应 6.6% 的单侧尾部极值概率
        self.min_rate_change = min_rate_change       # 绝对基点阈值：避免低波动期微小杂音触发信号(10个基点)
        self.min_spread_change = min_spread_change   # 绝对基点阈值：曲线陡峭/平坦的实质性变化(5个基点)

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 检查所需基础数据字段
        if 'dgs2' not in data.columns or 't10y2y' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 2. 数据清洗与前向填充
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 3. 计算极短期边际变化 (边际变化铁律: 不看绝对水位, 只看5天内的动量冲击)
        dgs2_mom = dgs2.diff(self.mom_window)
        t10y2y_mom = t10y2y.diff(self.mom_window)

        # 4. 计算滚动 Z-Score 以识别"极值"状态 (防接飞刀/适应不同波动率环境)
        dgs2_mom_mean = dgs2_mom.rolling(window=self.z_window, min_periods=self.z_window//2).mean()
        dgs2_mom_std = dgs2_mom.rolling(window=self.z_window, min_periods=self.z_window//2).std()
        dgs2_z = (dgs2_mom - dgs2_mom_mean) / (dgs2_mom_std + 1e-8)

        t10y2y_mom_mean = t10y2y_mom.rolling(window=self.z_window, min_periods=self.z_window//2).mean()
        t10y2y_mom_std = t10y2y_mom.rolling(window=self.z_window, min_periods=self.z_window//2).std()
        t10y2y_z = (t10y2y_mom - t10y2y_mom_mean) / (t10y2y_mom_std + 1e-8)

        # 5. 非线性特征交叉逻辑
        
        # 【多头脉冲】Bull Steepening 冲击: 2Y极速下行 (降息抢跑) + 曲线急剧变陡 (衰退/宽货币定价)
        bull_steepening = (
            (dgs2_z < -self.z_threshold) &                 # 极值: 短端利率极速下行
            (t10y2y_z > self.z_threshold) &                # 极值: 曲线急剧变陡
            (dgs2_mom <= -self.min_rate_change) &          # 经济学意义: 至少跌10个基点
            (t10y2y_mom >= self.min_spread_change)         # 经济学意义: 利差至少走扩5个基点
        )

        # 【空头脉冲】Bear Flattening 冲击: 2Y极速飙升 (通胀/紧缩恐慌) + 曲线急剧平坦化/倒挂加深
        bear_flattening = (
            (dgs2_z > self.z_threshold) &                  # 极值: 短端利率极速飙升
            (t10y2y_z < -self.z_threshold) &               # 极值: 曲线急剧平坦/倒挂
            (dgs2_mom >= self.min_rate_change) &           # 经济学意义: 至少涨10个基点
            (t10y2y_mom <= -self.min_spread_change)        # 经济学意义: 利差至少收窄5个基点
        )

        # 6. 生成脉冲信号 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        signal[bull_steepening] = 1.0
        signal[bear_flattening] = -1.0

        # 处理可能存在的 NaN，确保返回纯净数字
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"RateShockSteepeningFactor(mom_window={self.mom_window}, z_threshold={self.z_threshold})"