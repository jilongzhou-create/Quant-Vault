import numpy as np
import pandas as pd

class UnstructuredPolicyPivotFactor:
    """Unstructured Policy Pivot (policy_pivot / unstructured)

    逻辑: 捕捉美联储政策预期的剧变(NLP情绪突变)与宏观流动性冲量。鸽派突变若伴随恐慌飙升(紧急降息)则看空；若恐慌稳定(常规放水)则看多。鹰派突变直接看空。同时结合EPU政策不确定性脉冲，以及曲线牛陡(Bull Steepening)+极度恐慌衰竭作为绝对抄底信号。
    数据: fomc_sentiment, vixcls, usepuindxd, t10y2y, dgs2
    输出: 1.0 看多美股 (流动性释放/恐慌衰竭), -1.0 看空美股 (流动性收紧/政策恐慌或紧急降息)
    触发条件: FOMC情绪跳跃后5天内，或EPU极大极小变动，或收益率曲线牛陡且VIX见顶回落。预期 Trigger Rate 8-12%。
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 必须检查所需数据是否存在
        required_cols = ['fomc_sentiment', 'vixcls', 'usepuindxd', 't10y2y', 'dgs2']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 填充缺失值，避免因节假日导致的数据断层
        fomc = data['fomc_sentiment'].ffill()
        vix = data['vixcls'].ffill()
        epu = data['usepuindxd'].ffill()
        t10y2y = data['t10y2y'].ffill()
        dgs2 = data['dgs2'].ffill()

        # ---------------------------------------------------------
        # 1. FOMC 情绪跳跃信号 (NLP Unstructured)
        # ---------------------------------------------------------
        fomc_diff = fomc.diff(1)
        
        # 突变阈值 0.10，代表FOMC文本基调明显改变 (阶梯数据的边际变化)
        dovish_event = fomc_diff >= 0.10
        hawkish_event = fomc_diff <= -0.10
        
        # 脉冲信号保持 5 天（覆盖从会议公布到市场完全消化的时间，确保 Trigger Rate 达标且非连续）
        dovish_window = dovish_event.rolling(window=5, min_periods=1).max() > 0
        hawkish_window = hawkish_event.rolling(window=5, min_periods=1).max() > 0
        
        # 判断市场恐慌状态（避免接飞刀：降息可能因为市场正在崩盘）
        vix_momentum = vix / vix.shift(3) - 1.0
        vix_surge = vix_momentum > 0.10  # VIX 3天内飙升超过 10%
        
        dovish_window = dovish_window.fillna(False)
        hawkish_window = hawkish_window.fillna(False)
        vix_surge = vix_surge.fillna(False)
        
        # 鹰派转向：收紧流动性 -> 直接看空 (-1.0)
        signal.loc[hawkish_window] = -1.0
        
        # 鸽派转向：
        # 若 VIX 飙升 -> 联储因经济崩盘恐慌而紧急降息 -> 趋势恶化，看空 (-1.0)
        # 若 VIX 稳定 -> 常规的流动性宽松/利好 -> 顺势看多 (+1.0)
        signal.loc[dovish_window & vix_surge] = -1.0
        signal.loc[dovish_window & ~vix_surge] = 1.0

        # ---------------------------------------------------------
        # 2. EPU 政策不确定性震荡信号
        # ---------------------------------------------------------
        epu_mean = epu.rolling(window=252, min_periods=60).mean()
        epu_std = epu.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        epu_z = (epu - epu_mean) / epu_std
        epu_momentum = epu / epu.shift(3) - 1.0
        
        # 不确定性暴增 -> 抛售/看空 (-1.0)
        epu_shock = (epu_z > 1.5) & (epu_momentum > 0.15)
        epu_shock = epu_shock.fillna(False)
        signal.loc[epu_shock] = -1.0
        
        # 不确定性极值后回落 -> 恐慌衰竭，均值回归看多 (+1.0)
        epu_exhaustion = (epu_z > 2.0) & (epu_momentum < -0.10)
        epu_exhaustion = epu_exhaustion.fillna(False)
        signal.loc[epu_exhaustion] = 1.0

        # ---------------------------------------------------------
        # 3. 宏观流动性确认：Bull Steepening + 极度恐慌衰竭
        # ---------------------------------------------------------
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std
        
        # 市场抢跑降息：曲线变陡(15bps)且2年期大幅下行(15bps)
        t10_2_steepen = t10y2y.diff(5) >= 0.15
        dgs2_plunge = dgs2.diff(5) <= -0.15
        bull_steepening = t10_2_steepen & dgs2_plunge
        
        # 符合二阶导数铁律：极值(Z>2.0) + 衰竭(diff<0)
        vix_exhausted = (vix_z >= 2.0) & (vix.diff(3) < 0)
        
        bull_steepening = bull_steepening.fillna(False)
        vix_exhausted = vix_exhausted.fillna(False)
        
        # 产生共振 -> 强力抄底 (+1.0)
        signal.loc[bull_steepening & vix_exhausted] = 1.0

        # 确保输出清晰的极值脉冲信号
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"