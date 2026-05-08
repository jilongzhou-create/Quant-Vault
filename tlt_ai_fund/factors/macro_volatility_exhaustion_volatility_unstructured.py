import numpy as np
import pandas as pd

class MacroVolatilityExhaustionFactor:
    """波动率极值与拥挤反转脉冲 (Volatility Crowding Reversal Pulse)

    逻辑: 监控非结构化经济政策不确定性(EPU)与跨资产波动率(VIX/GVZ)的极端狂飙。绝对禁止在恐慌主升浪抄底！必须等待恐慌指标处于极端极值(半年 Z-Score > 2.5)且出现动量衰竭(diff() < 0)时，才视为拥挤盘瓦解的反转确认。确认反转后，利用收益率曲线的边际动量决定交易方向：曲线突然变陡(降息急救定价)则做多美债；曲线平坦化(通胀加息恐慌)则做空美债。
    数据: usepuindxd (经济政策不确定性), vixcls (VIX), gvzcls (黄金波动率), t10y2y (长短端利差)
    触发: (VIX Z-Score>2.5且VIX与GVZ回落) 或 (EPU Z-Score>2.5且EPU回落)。结合 T10Y2Y 3日动量生成 +/- 1.0 信号。
    输出: 狙击手级别的脉冲信号，极端事件触发后维持 5 天 (limit=4) 以满足 5%-15% Trigger Rate，常态休眠为 0.0。
    """

    def __init__(self):
        self.name = 'macro_volatility_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始信号全设为 0.0
        signal = pd.Series(0.0, index=data.index)

        # 检查所需数据是否存在 (禁止使用 CoreAnchor 数据)
        req_cols = ['usepuindxd', 'vixcls', 'gvzcls', 't10y2y']
        missing_cols = [col for col in req_cols if col not in data.columns]
        if missing_cols:
            return signal

        df = data[req_cols].ffill()

        # 计算 126日 (半年) 滚动 Z-Score，用于捕捉中短期宏观局部恐慌极值
        vix_mean = df['vixcls'].rolling(window=126, min_periods=63).mean()
        vix_std = df['vixcls'].rolling(window=126, min_periods=63).std()
        vix_zscore = (df['vixcls'] - vix_mean) / (vix_std + 1e-6)
        
        epu_mean = df['usepuindxd'].rolling(window=126, min_periods=63).mean()
        epu_std = df['usepuindxd'].rolling(window=126, min_periods=63).std()
        epu_zscore = (df['usepuindxd'] - epu_mean) / (epu_std + 1e-6)

        # 铁律2: 二阶导数 (极值条件)
        vix_extreme = vix_zscore > 2.5
        epu_extreme = epu_zscore > 2.5

        # 铁律2: 二阶导数 (衰竭条件 - Anti-Catch-Falling-Knife)
        vix_exhaustion = df['vixcls'].diff() < 0
        gvz_exhaustion = df['gvzcls'].diff() < 0
        epu_exhaustion = df['usepuindxd'].diff() < 0

        # 触发源 A: 跨资产恐慌极值 + 同步衰竭确认
        vol_trigger = vix_extreme & vix_exhaustion & gvz_exhaustion
        
        # 触发源 B: 非结构化宏观政策不确定性极值 + 衰竭确认
        epu_trigger = epu_extreme & epu_exhaustion
        
        # 只要任意一种宏观恐慌进入衰竭瓦解状态，即视为反转触发点
        panic_exhausted = vol_trigger | epu_trigger

        # 铁律3: 边际变化 (Marginal Change Only)
        # 使用收益率曲线的边际动量而非绝对水位
        # diff(3) > 0: 曲线剧烈陡峭化 (Bull Steepening, 往往是降息预期突增, 利好美债)
        # diff(3) <= 0: 曲线平坦化 (熊平, 往往是通胀失控+加息预期确认, 此时极值回落反而是风险偏好修复, 利空美债)
        curve_momentum = df['t10y2y'].diff(3)

        long_cond = panic_exhausted & (curve_momentum > 0)
        short_cond = panic_exhausted & (curve_momentum <= 0)

        # 生成脉冲信号
        raw_signal = pd.Series(0.0, index=df.index)
        raw_signal.loc[long_cond] = 1.0
        raw_signal.loc[short_cond] = -1.0

        # 将脉冲向后保持 4 天 (连同触发日共 5 天生效)
        # 这一步确保在满足极端苛刻条件的前提下，Signal 能够落在 5%-15% 的目标 Trigger Rate 区间内
        final_signal = raw_signal.replace(0.0, np.nan).ffill(limit=4).fillna(0.0)

        final_signal.name = self.name
        return final_signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"