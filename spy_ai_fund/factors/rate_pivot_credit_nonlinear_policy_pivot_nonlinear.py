import numpy as np
import pandas as pd

class YieldCreditPivotPulseFactor:
    """Yield Credit Pivot Pulse (policy_pivot/nonlinear)

    逻辑: 捕捉由于短期利率预期剧变并伴随信用利差同向确认所带来的流动性冲击。
          短端美债收益率(DGS2)急跌代表市场抢跑降息，但若此时高收益债利差(HY Spread)走阔，则为衰退恐慌(死于主跌浪，过滤不买)；
          只有当DGS2急跌且利差同步收窄(信用环境宽松)时，才是纯粹的鸽派流动性释放(Goldilocks)，触发强看多。
          反之，利率急升且利差走阔为鹰派紧缩恐慌，触发看空。
    数据: dgs2 (2年期美债), bamlh0a0hym2 (高收益债利差)
    输出: +1.0 (鸽派流动性脉冲), -1.0 (鹰派紧缩脉冲)
    触发条件: DGS2的5天变动Z-Score突破+/-1.20极值，且信用利差5天变化予以非线性同向验证。预期Trigger Rate 8%-14%。
    """

    def __init__(self):
        self.name = 'yield_credit_pivot_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖列是否存在
        if 'dgs2' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return signal
            
        # 前向填充缺失值，避免交易日历错位
        dgs2 = data['dgs2'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()
        
        # 计算5天(一周)维度的边际动量变化
        dgs2_diff = dgs2.diff(5)
        hy_diff = hy_spread.diff(5)
        
        # 计算DGS2变化的126天(半年)滚动Z-Score，动态自适应不同波动率周期
        roll_mean = dgs2_diff.rolling(window=126, min_periods=21).mean()
        roll_std = dgs2_diff.rolling(window=126, min_periods=21).std()
        
        # 避免除以0
        dgs2_diff_z = (dgs2_diff - roll_mean) / (roll_std + 1e-8)
        
        # 极值+转折 的非线性交叉条件
        
        # 鸽派狂欢 (看多): 短期利率极度下行 (降息抢跑) AND 信用利差收窄 (软着陆/无衰退恐慌)
        long_cond = (dgs2_diff_z < -1.20) & (hy_diff < 0.0)
        
        # 鹰派惊吓 (看空): 短期利率极度上行 (加息/通胀恐慌) AND 信用利差走阔 (紧缩导致风险偏好恶化)
        short_cond = (dgs2_diff_z > 1.20) & (hy_diff > 0.0)
        
        # 赋予脉冲信号
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        # 数据初始化阶段强制置0
        signal.iloc[:21] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"