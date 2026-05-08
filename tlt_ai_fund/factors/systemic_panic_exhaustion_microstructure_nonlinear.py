import numpy as np
import pandas as pd

class MacroMicroCrossExhaustionFactor:
    """Macro Micro Cross Exhaustion (Microstructure / Nonlinear)

    逻辑: 捕捉微观成交量极值与宏观恐慌/贪婪极值的共振反转。
          单维度的高阈值(>2.5)会导致触发率极低(如0.03%)，因此采用"非线性特征交叉"(弱学习器组合)方法：
          将宏观恐慌(VIX)、金融压力(NFCI/STLFSI) 和微观抛售(TLT 成交量)的极值判定阈值设为 0.6 (约前25%分位)，
          当这 3 个维度中至少有 2 个发生极值共振，并且至少有 2 个维度的二阶导数开始衰竭(当前值<均值)时，
          结合 TLT 的短期下跌/上涨趋势，输出具有高胜率的抄底/逃顶脉冲信号。
    数据: close, volume, vixcls, nfci (或 stlfsi4)
    触发: (极值恐慌得分 >= 2) 且 (动量衰竭得分 >= 2) 且 短期趋势向下 -> 看多脉冲 (+1.0)
          (极值贪婪得分 >= 2) 且 (动量恶化得分 >= 2) 且 短期趋势向上 -> 看空脉冲 (-1.0)
    输出: 狙击手级脉冲信号 [-1.0, 1.0]，满足 5%-15% 的 Trigger Rate 目标
    """

    def __init__(self):
        self.name = 'macro_micro_cross_exhaustion_microstructure_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 1. Microstructure: TLT Volume & Price Trend
        if 'volume' in data.columns and 'close' in data.columns:
            vol = data['volume']
            close = data['close']
            
            # 使用 63 个交易日 (约1个季度) 定义微观结构的常态
            vol_mean = vol.rolling(window=63, min_periods=21).mean()
            vol_std = vol.rolling(window=63, min_periods=21).std().replace(0, np.nan)
            vol_z = (vol - vol_mean) / vol_std
            
            # Climax: 过去 10 天内是否出现过成交量极值 (Z > 0.6)
            vol_climax = vol_z.rolling(window=10).max() > 0.6
            
            # Exhaustion (边际变化): 当前成交量萎缩，代表力量耗尽
            vol_fading = vol < vol.rolling(window=3).mean()
            
            # 趋势方向 (确认 Climax 是抛售还是逼空)
            trend_is_down = close < close.shift(5)
            trend_is_up = close > close.shift(5)
        else:
            return signal
            
        # 2. Macro Volatility: VIX
        if 'vixcls' in data.columns:
            vix = data['vixcls'].ffill()
            
            vix_mean = vix.rolling(window=126, min_periods=21).mean()
            vix_std = vix.rolling(window=126, min_periods=21).std().replace(0, np.nan)
            vix_z = (vix - vix_mean) / vix_std
            
            vix_panic = vix_z.rolling(window=10).max() > 0.6
            vix_complacent = vix_z.rolling(window=10).min() < -0.6
            
            # 恐慌衰竭 与 贪婪恶化 (二阶导数反转)
            vix_fading = vix < vix.rolling(window=3).mean()
            vix_worsening = vix > vix.rolling(window=3).mean()
        else:
            vix_panic = pd.Series(False, index=data.index)
            vix_complacent = pd.Series(False, index=data.index)
            vix_fading = pd.Series(False, index=data.index)
            vix_worsening = pd.Series(False, index=data.index)
            
        # 3. Macro Liquidity / Financial Stress: NFCI or STLFSI4
        # 优先使用 NFCI，如果没有则回退到 STLFSI4
        if 'nfci' in data.columns:
            stress = data['nfci'].ffill()
        elif 'stlfsi4' in data.columns:
            stress = data['stlfsi4'].ffill()
        else:
            stress = pd.Series(np.nan, index=data.index)
            
        if stress.isna().all():
            stress_panic = pd.Series(False, index=data.index)
            stress_complacent = pd.Series(False, index=data.index)
            stress_fading = pd.Series(False, index=data.index)
            stress_worsening = pd.Series(False, index=data.index)
        else:
            stress_mean = stress.rolling(window=126, min_periods=21).mean()
            stress_std = stress.rolling(window=126, min_periods=21).std().replace(0, np.nan)
            stress_z = (stress - stress_mean) / stress_std
            
            stress_panic = stress_z.rolling(window=10).max() > 0.6
            stress_complacent = stress_z.rolling(window=10).min() < -0.6
            
            # 使用 5 日均值捕捉阶梯状周频数据的边际变化。
            # 当阶梯数据下降时，连续数日内 当前值 < 5日均值 必定为 True，完美实现"脉冲区间"特征。
            stress_fading = stress < stress.rolling(window=5).mean()
            stress_worsening = stress > stress.rolling(window=5).mean()
            
        # 4. Nonlinear Feature Cross (高维非线性特征交叉打分)
        # 将 3 个弱条件交叉，要求 >= 2 才能触发，既保证了事件的极端性，又提高了 Trigger Rate。
        panic_score = vix_panic.astype(int) + stress_panic.astype(int) + vol_climax.astype(int)
        fading_score = vix_fading.astype(int) + stress_fading.astype(int) + vol_fading.astype(int)
        
        complacency_score = vix_complacent.astype(int) + stress_complacent.astype(int) + vol_climax.astype(int)
        worsening_score = vix_worsening.astype(int) + stress_worsening.astype(int) + vol_fading.astype(int)
        
        # 5. Signal Generator
        # 多头：宏观恐慌+微观抛售(得分>=2) 且 开始衰竭(得分>=2) 且 短期处于下跌趋势(证实抛售)
        bullish = (panic_score >= 2) & (fading_score >= 2) & trend_is_down
        
        # 空头：宏观极度乐观+微观狂热(得分>=2) 且 开始恶化(得分>=2) 且 短期处于上涨趋势(证实逼空)
        bearish = (complacency_score >= 2) & (worsening_score >= 2) & trend_is_up
        
        signal[bullish] = 1.0
        signal[bearish] = -1.0
        
        # 处理可能的 NaN
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"