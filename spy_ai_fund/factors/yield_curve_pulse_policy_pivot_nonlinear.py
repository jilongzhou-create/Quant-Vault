import numpy as np
import pandas as pd

class YieldCurvePulseFactor:
    """政策转向与预期反转因子 (policy_pivot/nonlinear)

    逻辑: 捕捉美联储预期突然反转导致的流动性冲量。关注短端利率(2年期)的急剧变化与收益率曲线形态(变陡/变平)的非线性交叉。当2年期国债收益率急剧下行(市场抢跑降息)导致曲线急剧牛陡，且高收益信用利差未见恶化(排除经济崩盘带来的被动衰退宽货币)时，代表良性鸽派宽松，输出强看多信号；反之当2年期急剧上行导致曲线熊平且信用利差走阔时，代表鹰派紧缩恐慌，输出看空信号。
    数据: [dgs2, t10y2y, bamlh0a0hym2]
    输出: [-1.0, 0.0, 1.0] 极端鸽派突变看多(+1.0)，极端鹰派突变看空(-1.0)
    触发条件: 5日内短端预期变动幅度达到近一次议息决议量级(20bps)，收益率曲线发生实质变性(10bps)。脉冲触发，只有从常态突变进入极值状态的第一个交易日输出信号，预期 Trigger Rate 5%-15%。
    """

    def __init__(self):
        self.name = 'yield_curve_pulse'
        # 经济学阈值设定
        self.lookback_days = 5
        self.dgs2_threshold = 0.20  # 20个基点，约等于一次美联储标准加降息幅度(25bps)的市场前置定价
        self.curve_threshold = 0.10 # 10个基点，确认长短端非平移的显著形变
        self.cred_threshold = 0.05  # 5个基点，用于剥离"经济衰退式崩盘"和"纯粹流动性转向"

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格遵守"零值休眠铁律"，默认全为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        required_cols = ['dgs2', 't10y2y', 'bamlh0a0hym2']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        # 前向填充数据缺失值，避免交易日历不对齐造成的NaN缺口
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        cred = data['bamlh0a0hym2'].ffill()
        
        # 遵循"边际变化铁律"与"二阶导数铁律"，禁止使用绝对水位判断
        dgs2_diff = dgs2.diff(self.lookback_days)
        t10y2y_diff = t10y2y.diff(self.lookback_days)
        cred_diff = cred.diff(self.lookback_days)
        
        # 状态定义:
        # 1. 鸽派突变(良性牛陡): 市场迅速定价降息 + 曲线变陡 + 信用风险未显著飙升(不是危机)
        is_bull_steep = (dgs2_diff <= -self.dgs2_threshold) & \
                        (t10y2y_diff >= self.curve_threshold) & \
                        (cred_diff <= self.cred_threshold)
                        
        # 2. 鹰派突变(恐慌熊平): 市场迅速定标紧缩 + 倒挂加深/曲线走平 + 信用利差扩大(紧缩传导至企业)
        is_bear_flat = (dgs2_diff >= self.dgs2_threshold) & \
                       (t10y2y_diff <= -self.curve_threshold) & \
                       (cred_diff >= self.cred_threshold)
                       
        # 脉冲触发控制: 仅在状态改变(首次突破极值阈值)的第一天发射子弹，防止接飞刀和连续输出
        trigger_long = is_bull_steep & (~is_bull_steep.shift(1).fillna(False))
        trigger_short = is_bear_flat & (~is_bear_flat.shift(1).fillna(False))
        
        # 填充信号
        signal.loc[trigger_long] = 1.0
        signal.loc[trigger_short] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback_days}, dgs2_th={self.dgs2_threshold})"