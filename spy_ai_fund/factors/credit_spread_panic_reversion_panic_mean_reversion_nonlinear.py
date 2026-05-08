import numpy as np
import pandas as pd

class CreditSpreadPanicReversionFactor:
    """Credit Spread Panic Reversion (panic_mean_reversion/nonlinear)

    逻辑: 高收益债(HY)与BBB级企业债利差反映市场对企业违约的真实恐慌。
          在美股长牛属性下，当信用利差飙升至一年内极端高位(Z-Score > 1.5)时代表极端恐慌；
          必须等待最新3日动量由正转负(利差收窄)，确认恐慌开始衰竭，此时输出强烈看多(+1.0)实现抄底。
          反之，当利差处于常态环境(Z-Score < 0.5)且短期内(5日)快速走阔时，代表无预期的轻微信用恶化，
          对股市钝刀割肉，输出看空(-1.0)。
    数据: bamlh0a0hym2 (HY利差), bamlc0a4cbbb (BBB利差)
    输出: +1.0 (恐慌极值见顶衰竭，强多), -1.0 (常态下信用条件突然恶化，看空), 0.0 (休眠)
    触发条件: 极值+衰竭触发脉冲多头，常态急剧恶化触发脉冲空头。预期 Trigger Rate 在 5% - 15% 之间。
    """

    def __init__(self):
        self.name = 'credit_spread_panic_reversion_nonlinear'
        self.window = 252
        self.extreme_z_threshold = 1.5   # 极端恐慌阈值 (Z-Score > 1.5，约前 6% 分位)
        self.normal_z_threshold = 0.5    # 常态环境阈值 (不高于过去一年均值0.5个标准差)
        self.hy_widen_bps = 0.20         # 5天走阔 20个基点 (0.2%)
        self.bbb_widen_bps = 0.08        # 5天走阔 8个基点 (0.08%)

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['bamlh0a0hym2', 'bamlc0a4cbbb']
        for col in req_cols:
            if col not in data.columns:
                signal.name = self.name
                return signal
                
        # 填充缺失值并保持时序逻辑
        hy_spread = data['bamlh0a0hym2'].ffill()
        bbb_spread = data['bamlc0a4cbbb'].ffill()
        
        # 计算 252 日滚动的 Z-Score
        hy_mean = hy_spread.rolling(self.window, min_periods=60).mean()
        hy_std = hy_spread.rolling(self.window, min_periods=60).std()
        hy_zscore = (hy_spread - hy_mean) / hy_std.replace(0, np.nan)
        
        bbb_mean = bbb_spread.rolling(self.window, min_periods=60).mean()
        bbb_std = bbb_spread.rolling(self.window, min_periods=60).std()
        bbb_zscore = (bbb_spread - bbb_mean) / bbb_std.replace(0, np.nan)
        
        # 计算衰竭边际变化 (极值后是否回落)
        hy_diff_3 = hy_spread.diff(3)
        bbb_diff_3 = bbb_spread.diff(3)
        
        # 计算突发恶化的边际变化 (突然的走阔)
        hy_diff_5 = hy_spread.diff(5)
        bbb_diff_5 = bbb_spread.diff(5)
        
        # 多头信号: 极端恐慌 (双利差高位) + 二阶导衰竭 (3日变动小于0，恐慌见顶收窄)
        long_cond = (
            (hy_zscore > self.extreme_z_threshold) &
            (bbb_zscore > self.extreme_z_threshold) &
            (hy_diff_3 < 0.0) &
            (bbb_diff_3 < 0.0)
        )
        
        # 空头信号: 常态环境 (未提前计入风险) 下，短时间内急剧走阔 (轻微恶化)
        short_cond = (
            (hy_zscore < self.normal_z_threshold) &
            (bbb_zscore < self.normal_z_threshold) &
            (hy_diff_5 > self.hy_widen_bps) &
            (bbb_diff_5 > self.bbb_widen_bps) &
            (~long_cond)
        )
        
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"