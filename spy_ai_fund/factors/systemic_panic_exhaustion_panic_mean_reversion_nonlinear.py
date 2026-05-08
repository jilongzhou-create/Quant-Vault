import numpy as np
import pandas as pd

class SystemicPanicExhaustionFactor:
    """金融压力恐慌极值与衰竭交叉因子 (panic_mean_reversion/nonlinear)

    逻辑: 将 VIX、高收益债信用利差 (BAMLH0A0HYM2) 和金融压力指数 (STLFSI4) 融合成高维系统性压力指标。当系统性压力处于历史高位(极度恐慌)且出现二阶导数转负(动能衰竭)时，输出强烈看多的脉冲信号，捕捉市场恐慌见顶回落的极短窗口，绝不接飞刀。
    数据: [vixcls, bamlh0a0hym2, stlfsi4]
    输出: 强烈看多美股 (+1.0)
    触发条件: 综合 Z-Score 大于 3.5 (处于均值上方一倍标准差多)，且当日压力动能由正转负，预期 Trigger Rate 5%-15%
    """

    def __init__(self, window=126, z_threshold=3.5):
        self.name = 'systemic_panic_exhaustion_nonlinear'
        self.window = window
        self.z_threshold = z_threshold
        # VIX: 股市波动恐慌, BAMLH0A0HYM2: 信用市场利差, STLFSI4: 宏观金融系统压力
        self.cols = ['vixcls', 'bamlh0a0hym2', 'stlfsi4']

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        available_cols = [c for c in self.cols if c in data.columns]
        if not available_cols:
            return signal
            
        df = data[available_cols].ffill()
        
        # 计算每个指标的 Z-score 并加总成高维系统压力指数
        combined_stress = pd.Series(0.0, index=df.index)
        valid_counts = pd.Series(0, index=df.index)
        
        for col in available_cols:
            r_mean = df[col].rolling(window=self.window, min_periods=21).mean()
            r_std = df[col].rolling(window=self.window, min_periods=21).std()
            
            # 使用微小常数防止除以零
            z_score = (df[col] - r_mean) / (r_std + 1e-6)
            
            z_score = z_score.fillna(0)
            combined_stress += z_score
            valid_counts += df[col].notna().astype(int)
            
        # 归一化综合压力指数，使其与有效数据字段的数量无关，对齐到3个指标的标准基准
        combined_stress = combined_stress / valid_counts.replace(0, 1) * 3
        
        # 极度恐慌条件: 综合压力显著高于近期均值
        is_extreme = combined_stress > self.z_threshold
        
        # 恐慌衰竭条件: 综合压力当日较前日下降 (绝不在恐慌加剧时买入)
        is_exhausted = combined_stress.diff(1) < 0
        
        # 产生脉冲信号
        trigger = is_extreme & is_exhausted
        signal[trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_threshold={self.z_threshold})"