import numpy as np
import pandas as pd

class PolicyPivotCreditTransmissionFactor:
    """政策转向与信用传导因子 (policy_pivot/nonlinear)

    逻辑: 捕捉美联储政策预期突变(短端国债急跌或FOMC情绪转鸽)与企业融资环境的共振。只有在政策转向成功驱动信用利差从高位回落(流动性传导成功,恐慌衰竭)时,才是安全的美股抄底买点; 反之,在市场原本乐观时遭遇鹰派冲击且信用开始紧缩,则看空。
    数据: dgs2 (2年期国债收益率), bamlh0a0hym2 (高收益债利差), fomc_sentiment (FOMC情绪)
    输出: +1.0 看多 (鸽派转向且信用恐慌衰竭), -1.0 看空 (鹰派冲击且信用趋势恶化), 常态 0.0
    触发条件: 政策预期剧变 且 信用状态发生二阶转向瞬间触发, 预期 Trigger Rate 8%-12%
    """

    def __init__(self):
        self.name = 'policy_pivot_credit_transmission'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律: 默认返回0.0的脉冲信号
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必要数据字段
        if 'dgs2' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return signal
            
        dgs2 = data['dgs2'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()
        
        # ----------------------------------------------------
        # 1. 政策预期的边际突变动量 (动量法则)
        # 短期(5天)收益率变动10个基点视为对降息/加息预期的剧烈抢跑
        dgs2_change_5d = dgs2.diff(5)
        
        # 边际变化铁律: 加入 FOMC 文本情绪的边际 jump, 绝对禁止使用绝对值
        dovish_text_jump = pd.Series(False, index=data.index)
        hawkish_text_jump = pd.Series(False, index=data.index)
        if 'fomc_sentiment' in data.columns:
            fomc = data['fomc_sentiment'].ffill()
            dovish_text_jump = fomc.diff(3) > 0.15   # 情绪突发转鸽
            hawkish_text_jump = fomc.diff(3) < -0.15 # 情绪突发转鹰
            
        # 宏观政策脉冲条件
        dovish_shock = (dgs2_change_5d < -0.10) | dovish_text_jump
        hawkish_shock = (dgs2_change_5d > 0.10) | hawkish_text_jump
        
        # ----------------------------------------------------
        # 2. 信用环境水位与二阶导数 (防接飞刀铁律)
        # 计算高收益债利差的120天Z-Score, 评估信用环境压力水位 (相对值逻辑)
        hy_mean_120d = hy_spread.rolling(window=120, min_periods=60).mean()
        hy_std_120d = hy_spread.rolling(window=120, min_periods=60).std()
        hy_zscore = (hy_spread - hy_mean_120d) / (hy_std_120d + 1e-6)
        
        # 信用利差的短期衰竭/爆发动量 (评估恐慌是否结束)
        hy_change_3d = hy_spread.diff(3)
        
        # ----------------------------------------------------
        # 3. 非线性交叉触发逻辑 (SPY核心物理属性)
        
        # 多头脉冲: 政策强力转鸽 + 信用利差原本偏高(高压) + 但利差已经开始收窄(恐慌二阶导回落)
        # 代表流动性注入成功, 阻断了危机蔓延 -> 坚决看多
        buy_pulse = dovish_shock & (hy_zscore > 0.5) & (hy_change_3d < 0.0)
        
        # 空头脉冲: 政策意外转鹰 + 信用利差原本极低(市场极度乐观) + 但利差突然走阔(轻微恐慌与趋势恶化)
        # 代表流动性收缩开始刺破资产泡沫 -> 趋势恶化看空
        sell_pulse = hawkish_shock & (hy_zscore < 0.0) & (hy_change_3d > 0.05)
        
        # 赋值狙击型脉冲信号
        signal.loc[buy_pulse] = 1.0
        signal.loc[sell_pulse] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"