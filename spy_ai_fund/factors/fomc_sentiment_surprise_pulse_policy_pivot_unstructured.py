import numpy as np
import pandas as pd

class UnstructuredPolicyUncertaintyPulseFactor:
    """政策不确定性衰竭与联储转向共振因子 (policy_pivot/unstructured)

    逻辑: 
    单纯的联储边际转鸽往往发生在经济剧烈衰退的崩盘期(买预期卖事实)，此时做多容易接飞刀导致胜率极低。
    本因子通过融合NLP抽取的每日"美国经济政策不确定性指数(EPU)", 构建严密的"政策底+情绪底"共振逻辑：
    1. 极值衰竭看多：当联储边际转鸽，且市场政策不确定性(EPU)从高位开始显著衰竭回落时，标志着靴子落地，形成高胜率抄底脉冲。
    2. 黑天鹅飙升看空：当且仅当市场原本处于常态平静期，突然遭遇联储转鹰引发EPU剧烈飙升，代表趋势彻底恶化，产生避险做空信号。
    
    数据: fomc_sentiment (FOMC情绪得分), usepuindxd (美国经济政策不确定性指数EPU)
    输出: +1.0 看多(恐慌衰竭落地), -1.0 看空(突发不确定性飙升), 常态0.0
    触发条件: 联储边际突变 + EPU动态Z-score出现剧烈拐点, 预期Trigger Rate控制在8%~12%
    """

    def __init__(self):
        self.name = 'unstructured_policy_uncertainty_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据是否存在 (同属 unstructured NLP宏观数据领域)
        if 'fomc_sentiment' not in data.columns or 'usepuindxd' not in data.columns:
            return signal
            
        fomc = data['fomc_sentiment'].ffill()
        epu = data['usepuindxd'].ffill()
        
        if epu.dropna().empty or fomc.dropna().empty:
            return signal

        # 1. 提取 FOMC 情绪跳跃边际信号 (阶梯状低频数据必须使用diff捕捉预期改变的瞬间)
        fomc_diff = fomc.diff(1).fillna(0.0)
        
        # 建立近3周(15个交易日)内的政策转向窗口记忆
        # 大于0.05代表向鸽派突变，小于-0.05代表向鹰派突变
        cond_dovish = fomc_diff.rolling(window=15, min_periods=1).max() > 0.05
        cond_hawkish = fomc_diff.rolling(window=15, min_periods=1).min() < -0.05
        
        # 2. 计算 EPU 的半年(126天)滚动动态 Z-Score，反映不确定性相较于最近宏观常态的偏离度
        epu_mean = epu.rolling(window=126, min_periods=21).mean()
        epu_std = epu.rolling(window=126, min_periods=21).std()
        
        # 处理异常波动率情况
        epu_std = epu_std.replace(0, np.nan).ffill().fillna(1.0)
        epu_z = ((epu - epu_mean) / epu_std).fillna(0.0)
        
        # 3. 构造狙击手级别的高胜率看多脉冲 (+1.0)
        # 坚决防飞刀：EPU必须处于高位(恐慌)，且近几日开始明确回落(动量 < 0)
        epu_high_recent = epu_z.rolling(window=5, min_periods=1).max() > 1.0
        epu_falling = (epu.diff(3).fillna(0.0) < 0) & (epu.diff(1).fillna(0.0) < 0)
        
        # 买点A: FOMC边际转鸽 + 不确定性由高位落地衰竭
        buy_dovish = cond_dovish & epu_high_recent & epu_falling
        
        # 买点B: 无视FOMC，极端黑天鹅政策恐慌本身迎来衰竭反转 (Z-Score > 2.5极值 + 开始回落)
        buy_extreme = (epu_z.rolling(window=5, min_periods=1).max() > 2.5) & epu_falling
        
        buy_conditions = buy_dovish | buy_extreme
        
        # 4. 构造避险看空脉冲 (-1.0)
        # 不在主跌浪追空：只在EPU此前处于相对平静期，随后突然单日剧烈飙升时，确认为重大转折
        epu_calm_before = epu_z.shift(3).fillna(0.0) < 0.5
        epu_surging = epu_z.diff(3).fillna(0.0) > 1.5
        
        # 卖点A: FOMC意外转鹰 引发的市场强烈政策不确定性恐慌
        sell_hawkish = cond_hawkish & epu_surging
        
        # 卖点B: 纯粹的突发特大政策恐慌 (常态下暴增超过2个标准差)
        sell_black_swan = epu_calm_before & (epu_z.diff(3).fillna(0.0) > 2.0) & (epu.diff(1).fillna(0.0) > 0)
        
        sell_conditions = sell_hawkish | sell_black_swan
        
        # 5. 合成极值脉冲信号，买入优先级高于做空(长牛物理属性)
        signal.loc[buy_conditions] = 1.0
        signal.loc[sell_conditions & ~buy_conditions] = -1.0
        
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"