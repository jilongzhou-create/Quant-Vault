import numpy as np
import pandas as pd

class RateRepricingPulseFactor:
    """政策转向与流动性冲量 (policy_pivot/nonlinear)

    逻辑: 捕捉美联储政策预期的剧烈反转。当短端利率(2年期国债)急剧下降且收益率曲线陡峭化(Bull Steepening)时,
          意味着市场在快速定价“鸽派降息/流动性释放”, 此时输出看多脉冲。反之, 利率飙升且曲线平坦化输出看空脉冲。
          同时结合FOMC情感边际跃升(鸽派突变)直接触发看多脉冲。绝对不看利率绝对水位, 只看边际动量变化。
    数据: dgs2(2年期美债), t10y2y(长短端利差), fomc_sentiment(FOMC会议情感得分)
    输出: +1.0 强烈看多(流动性宽松预期), -1.0 看空(紧缩恐慌), 0.0 常态休眠
    触发条件: 利率变动动量Z-Score极值伴随日内动能确认, 或FOMC情绪单日跳跃>0.25; 预期 Trigger Rate 8% - 12%
    """

    def __init__(self):
        self.name = 'rate_repricing_pulse_factor'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        has_dgs2 = 'dgs2' in data.columns
        has_t10y2y = 't10y2y' in data.columns
        
        rate_buy = pd.Series(False, index=data.index)
        rate_sell = pd.Series(False, index=data.index)
        
        if has_dgs2 and has_t10y2y:
            # 填补可能存在的节假日缺失值
            dgs2_ts = data['dgs2'].ffill()
            t10y2y_ts = data['t10y2y'].ffill()
            
            # 计算5日变化动量, 捕捉短期的剧烈Repricing
            dgs2_diff5 = dgs2_ts.diff(5)
            t10y2y_diff5 = t10y2y_ts.diff(5)
            
            # 使用252个交易日滚动窗口计算动量的Z-Score, 衡量利率变化的极端程度
            dgs2_mean = dgs2_diff5.rolling(window=252, min_periods=63).mean()
            dgs2_std = dgs2_diff5.rolling(window=252, min_periods=63).std()
            dgs2_z = (dgs2_diff5 - dgs2_mean) / (dgs2_std + 1e-6)
            
            t10y2y_mean = t10y2y_diff5.rolling(window=252, min_periods=63).mean()
            t10y2y_std = t10y2y_diff5.rolling(window=252, min_periods=63).std()
            t10y2y_z = (t10y2y_diff5 - t10y2y_mean) / (t10y2y_std + 1e-6)
            
            # 获取日内确认动量 (防止在下跌趋势的中途反弹日错误开仓)
            dgs2_daily_change = dgs2_ts.diff(1)
            
            # Bull steepening 逻辑: 短端极速下行 (Z < -1.25) 且 曲线剧烈变陡 (Z > 0.75), 且今日短端仍在下行
            rate_buy = (dgs2_z < -1.25) & (t10y2y_z > 0.75) & (dgs2_daily_change < 0)
            
            # Bear flattening 逻辑: 短端极速飙升 (Z > 1.25) 且 曲线平坦化/倒挂加深 (Z < -0.75), 且今日短端仍在飙升
            rate_sell = (dgs2_z > 1.25) & (t10y2y_z < -0.75) & (dgs2_daily_change > 0)
            
        fomc_buy = pd.Series(False, index=data.index)
        fomc_sell = pd.Series(False, index=data.index)
        
        if 'fomc_sentiment' in data.columns:
            fomc_ts = data['fomc_sentiment'].ffill()
            # 严格遵守边际变化铁律: 捕捉低频阶梯状数据的Jump
            fomc_diff = fomc_ts.diff(1).fillna(0.0)
            
            fomc_jump_up = fomc_diff > 0.25
            fomc_jump_down = fomc_diff < -0.25
            
            # 跳跃发生当天及随后的2个交易日内维持脉冲信号, 使流动性反转有发酵时间
            fomc_buy = fomc_jump_up.rolling(window=3, min_periods=1).max() > 0
            fomc_sell = fomc_jump_down.rolling(window=3, min_periods=1).max() > 0

        # 综合交叉逻辑
        buy_cond = rate_buy | fomc_buy
        sell_cond = rate_sell | fomc_sell
        
        # 信号赋值
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        # 处理极其罕见的冲突状态
        signal[buy_cond & sell_cond] = 0.0
        
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"