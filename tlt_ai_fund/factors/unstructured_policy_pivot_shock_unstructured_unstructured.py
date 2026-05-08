import numpy as np
import pandas as pd

class UnstructuredPolicyPivotShockFactor:
    """政策预期突变因子 (Unstructured / NLP Sentiment)

    逻辑: 捕捉美联储FOMC鹰鸽情绪的边际极值跳跃。因为NLP情绪是低频阶梯数据，必须只在其边际突变的极短窗口期内，配合短端前瞻利率(dgs2)上行/下行动能衰竭产生共振时，才发出狙击脉冲，从而避免逆势接飞刀。
    数据: fomc_sentiment (非结构化情绪), dgs2 (短端政策敏感利率), t10y2y (曲线形态)
    触发: fomc_sentiment 3日变化量Z-Score > 2.0 (边际突变) + dgs2跌破3日均线 (势头衰竭) + t10y2y边际变陡。
    输出: 脉冲信号 [-1.0, 1.0]，常态严格为0.0。
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 依赖列检查
        req_cols = ['fomc_sentiment', 'dgs2', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        df = data[req_cols].ffill()
        
        # ---------------- 铁律3: 边际变化 ----------------
        # 绝对禁止用 fomc_sentiment > 0 直接买入！使用 3 日变动量捕捉最新会议预期的跳跃
        fomc_delta = df['fomc_sentiment'].diff(3)
        
        # 计算 126 日(半年)滚动 Z-Score 识别突变极值
        roll_mean = fomc_delta.rolling(126, min_periods=21).mean()
        roll_std = fomc_delta.rolling(126, min_periods=21).std()
        fomc_z = (fomc_delta - roll_mean) / (roll_std + 1e-6)
        
        # 识别脉冲日: 统计学极值(Z>2.0) 或 具备经济学意义的实质性态度跃升(绝对变动>0.4)
        is_pigeon_shock = (fomc_z > 2.0) | (fomc_delta > 0.4)
        is_hawk_shock = (fomc_z < -2.0) | (fomc_delta < -0.4)
        
        # FOMC情绪跳跃的余波发酵窗口 (持续向后传播 4 个交易日)
        pigeon_window = is_pigeon_shock.rolling(4, min_periods=1).max() > 0
        hawk_window = is_hawk_shock.rolling(4, min_periods=1).max() > 0
        
        # ---------------- 铁律2: 二阶导数 (Anti-Catch-Falling-Knife) ----------------
        # 即使美联储转鸽，如果短端利率(dgs2)仍在惯性飙升，买入会被主跌浪杀死。
        # 必须等待短端利率动能衰竭，以及曲线结构(t10y2y)给出正确的 Price-in 形态。
        
        # 鸽派脉冲衰竭与确立条件：短端利率回落(下穿均线) + 曲线出现 Bull Steepening (边际变陡)
        dgs2_falling = df['dgs2'] < df['dgs2'].rolling(3).mean()
        curve_steepening = df['t10y2y'].diff(2) > 0
        
        # 鹰派脉冲衰竭与确立条件：短端利率狂飙(上穿均线) + 曲线出现 Bear Flattening (边际变平/倒挂加剧)
        dgs2_rising = df['dgs2'] > df['dgs2'].rolling(3).mean()
        curve_flattening = df['t10y2y'].diff(2) < 0
        
        # ---------------- 铁律1: 零值休眠 (Sniper Pulse) ----------------
        # 只有在 (跳跃窗口期) 且 (资产走势开始配合) 的共振瞬间，才扣动扳机
        long_cond = pigeon_window & dgs2_falling & curve_steepening
        short_cond = hawk_window & dgs2_rising & curve_flattening
        
        # 生成最终脉冲
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"