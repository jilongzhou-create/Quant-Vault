import numpy as np
import pandas as pd

class MicrostructureRepoCurveFactor:
    """Microstructure Repo Curve Factor (microstructure/nonlinear)

    逻辑: 结合了FICC资金面(FTS: 联邦基金-3月国库券利差)和情绪面(VIX)的微观结构耗竭信号。脉冲逻辑在于: 当市场极度定价降息(FTS飙升)或极度恐慌(VIX飙升)并开始衰竭回落时，配合收益率曲线陡峭化，表明流动性危机解除，美债将迎来报复性反弹；反之当市场极度定价加息或极度贪婪且开始反转时，看空美债。必须使用二阶导(3日均值)捕捉衰竭瞬间，平时处于零值休眠状态。
    数据: dff, dtb3, t10y2y, vixcls
    触发: FTS或VIX的126日Z-Score极值(>1.25或<-1.25) + 3日均值衰竭(二阶导) + 曲线动量配合
    输出: +1.0 (恐慌/加息定价耗竭买入), -1.0 (贪婪/降息定价耗竭卖出), 平时 0.0
    """

    def __init__(self):
        self.name = 'microstructure_repo_curve_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # Check required columns
        req_cols = ['dff', 'dtb3', 't10y2y', 'vixcls']
        for col in req_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)

        # Create safe dataframe and forward fill missing data
        df = pd.DataFrame(index=data.index)
        df['dff'] = data['dff'].ffill()
        df['dtb3'] = data['dtb3'].ffill()
        df['t10y2y'] = data['t10y2y'].ffill()
        df['vix'] = data['vixcls'].ffill()

        # FTS (Flight to Safety / Policy Expectation Premium)
        # 资金面微观结构: 当dtb3远低于dff时(极度正值)，为资金避险或强降息预期
        fts = df['dff'] - df['dtb3']
        fts_mean = fts.rolling(126).mean()
        fts_std = fts.rolling(126).std()
        fts_z = (fts - fts_mean) / (fts_std + 1e-6)
        
        fts_mean3 = fts.rolling(3).mean()
        fts_falling = fts < fts_mean3
        fts_rising = fts > fts_mean3

        # VIX Equity Panic Microstructure
        vix_mean = df['vix'].rolling(126).mean()
        vix_std = df['vix'].rolling(126).std()
        vix_z = (df['vix'] - vix_mean) / (vix_std + 1e-6)
        
        vix_mean3 = df['vix'].rolling(3).mean()
        vix_falling = df['vix'] < vix_mean3
        vix_rising = df['vix'] > vix_mean3

        # Curve steepening/flattening structure (FICC macro confirmation)
        t10y2y_mean3 = df['t10y2y'].rolling(3).mean()
        curve_steep = df['t10y2y'] > t10y2y_mean3
        curve_flat = df['t10y2y'] < t10y2y_mean3

        # =========================================================
        # Construct LONG conditions (+1.0)
        # =========================================================
        # 1. VIX Panic Exhaustion: 恐慌极值 + 开始回落衰竭 + 曲线陡峭化(降息预期/避险资金流入)
        long_vix = (vix_z > 1.25) & vix_falling & curve_steep
        
        # 2. FTS Hike-Pricing Exhaustion: 极度加息预期极值(3M远高于隔夜) + 开始反转衰竭 + 曲线陡峭化
        long_fts = (fts_z < -1.25) & fts_rising & curve_steep
        
        long_cond = long_vix | long_fts

        # =========================================================
        # Construct SHORT conditions (-1.0)
        # =========================================================
        # 1. VIX Complacency Reversal: 极度贪婪 + 波动率开始扩张 + 曲线平坦化(短端利率上行)
        short_vix = (vix_z < -1.25) & vix_rising & curve_flat
        
        # 2. FTS Cut-Pricing Exhaustion: 极度降息预期极值(3M远低于隔夜) + 预期落空衰竭(higher for longer) + 曲线平坦化
        short_fts = (fts_z > 1.25) & fts_falling & curve_flat
        
        short_cond = short_vix | short_fts

        # =========================================================
        # Assign Sniper Pulse Signals
        # =========================================================
        signal = pd.Series(0.0, index=df.index)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        # Handle highly unlikely clashes directly
        signal[long_cond & short_cond] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"