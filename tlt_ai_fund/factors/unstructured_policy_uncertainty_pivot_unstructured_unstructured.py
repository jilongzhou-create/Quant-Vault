import numpy as np
import pandas as pd

class UnstructuredPolicyEpuExhaustionFactor:
    """政策不确定性衰竭与FOMC突变共振因子 (Unstructured / NLP)

    逻辑: 捕捉两类非结构化政策极端事件带来的美债(TLT)脉冲机会：
          1. FOMC情绪突变: 联储声明发生极端鸽派/鹰派突变(Z-Score>2.5), 并等待市场短端利率(dgs2)动能顺应美联储(衰竭确认)时顺势生成脉冲。
          2. 经济政策不确定性(EPU)衰竭: 当EPU发生极端恐慌飙升(Z-Score>2.5)后开始回落，且避险买盘枯竭(短端利率开始回升)时，说明极度恐慌结束，做空美债；
             反之，极度自满被打破(EPU从极低位抬头)，资金开始避险(利率下行)时，做多美债。
    数据: fomc_sentiment (NLP情绪), usepuindxd (经济政策不确定性), dgs2 (短端政策利率，用作市场衰竭确认)
    触发: 边际变化极值条件 (Z-Score > 2.5) + 二阶导数/反转确认 (diff(2) 反向)
    输出: 严格脉冲型信号 [-1.0, 1.0], 非触发日为 0.0, 满足零值休眠铁律。
    """

    def __init__(self):
        self.name = 'unstructured_policy_epu_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        required_cols = ['fomc_sentiment', 'usepuindxd', 'dgs2']
        
        # 缺失字段处理：返回全 0 序列
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        # 前向填充避免未来数据泄漏
        df = data[required_cols].ffill()
        fomc = df['fomc_sentiment']
        usepu = df['usepuindxd']
        dgs2 = df['dgs2']

        # ---------------------------------------------------------
        # 铁律3: 边际变化 (绝对禁止使用水平绝对值，必须使用 diff 计算动量)
        # ---------------------------------------------------------
        # FOMC情绪：使用 10日窗口捕捉会议期间的阶梯跃升，并维持活跃期
        fomc_chg10 = fomc.diff(10)
        # EPU不确定性：使用 5日窗口捕捉短期突变
        usepu_chg5 = usepu.diff(5)
        
        # ---------------------------------------------------------
        # 计算 Z-Score (动态窗口，避免前瞻偏差)
        # 添加 1e-4 防止 FOMC 在非会议期的 std 趋零导致除零错误
        # ---------------------------------------------------------
        fomc_std = fomc_chg10.rolling(window=252, min_periods=21).std().clip(lower=1e-4)
        z_fomc = fomc_chg10 / fomc_std

        usepu_std = usepu_chg5.rolling(window=252, min_periods=21).std().clip(lower=1e-4)
        z_usepu = usepu_chg5 / usepu_std

        # ---------------------------------------------------------
        # 铁律2: 二阶导数 / 衰竭确认 (Anti-Catch-Falling-Knife)
        # 必须等待市场动能开始出现实质性拐点才能扣动扳机
        # ---------------------------------------------------------
        dgs2_diff2 = dgs2.diff(2)
        usepu_diff2 = usepu.diff(2)

        # ---------------------------------------------------------
        # 脉冲触发条件判定
        # ---------------------------------------------------------
        
        # 多头触发 (Bullish TLT = +1.0)
        # 场景A: FOMC极端鸽派突变 (Z > 2.5) + 短端利率开始下行(确认降息交易/无飞刀风险)
        bull_fomc = (z_fomc > 2.5) & (dgs2_diff2 < 0)
        
        # 场景B: EPU极度自满后觉醒 (Z < -2.5) + 不确定性重新抬头(diff>0) + 短端利率下行(资金确立避险买入美债)
        bull_epu = (z_usepu < -2.5) & (usepu_diff2 > 0) & (dgs2_diff2 < 0)
        
        bull_trigger = bull_fomc | bull_epu

        # 空头触发 (Bearish TLT = -1.0)
        # 场景A: FOMC极端鹰派突变 (Z < -2.5) + 短端利率开始上行(确认加息交易/抛售美债)
        bear_fomc = (z_fomc < -2.5) & (dgs2_diff2 > 0)
        
        # 场景B: EPU极度恐慌后衰竭 (Z > 2.5) + 不确定性开始回落(diff<0) + 短端利率上行(避险盘撤出美债)
        bear_epu = (z_usepu > 2.5) & (usepu_diff2 < 0) & (dgs2_diff2 > 0)
        
        bear_trigger = bear_fomc | bear_epu

        # ---------------------------------------------------------
        # 铁律1: 零值休眠 (默认常态为 0.0，仅在脉冲发生时赋值为 1.0 / -1.0)
        # ---------------------------------------------------------
        signal = pd.Series(0.0, index=df.index)
        signal[bull_trigger] = 1.0
        signal[bear_trigger] = -1.0

        # 处理极其罕见的同时触发冲突，安全归零
        conflict = bull_trigger & bear_trigger
        signal[conflict] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"