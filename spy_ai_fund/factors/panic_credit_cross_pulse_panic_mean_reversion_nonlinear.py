import numpy as np
import pandas as pd

class PanicCreditCrossPulseFactor:
    """恐慌极值与信用利差交叉回归因子 (panic_mean_reversion/nonlinear)

    逻辑: 将VIX与高收益债信用利差(基本面恐慌)进行非线性交叉。美股极度恐慌且不再恶化时往往是抄底良机(极值+衰竭); 信用利差走阔且波动率温和抬升时往往面临主跌浪，应看空。
    数据: vixcls (VIX), bamlh0a0hym2 (高收益债信用利差)
    输出: +1.0 表示极端恐慌衰竭(强烈看多); -1.0 表示恐慌发酵初期且未衰竭(看空); 0.0 为常态
    触发条件: VIX或利差的252日Z-score>1.5且开始回落为多头; VIX日升>10%且利差走阔为空头。预期Trigger Rate 8%-12%
    """

    def __init__(self):
        self.name = 'panic_credit_cross_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['vixcls', 'bamlh0a0hym2']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 填充缺失值，避免非交易日NaN影响计算
        vix = data['vixcls'].ffill()
        hy_spread = data['bamlh0a0hym2'].ffill()
        
        # 1. 计算252日(约1年) Z-Score 判断当前状态是否处于历史尾部极值区
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-6)
        
        hy_mean = hy_spread.rolling(window=252, min_periods=60).mean()
        hy_std = hy_spread.rolling(window=252, min_periods=60).std()
        hy_z = (hy_spread - hy_mean) / (hy_std + 1e-6)
        
        # 2. 计算动量与边际变化 (防接飞刀二阶导数法则)
        vix_diff = vix.diff()
        vix_pct = vix.pct_change()
        
        hy_diff = hy_spread.diff()
        hy_diff_3d = hy_spread.diff(3) # 三日趋势，过滤单日噪音
        
        # 3. 强看多脉冲 (+1.0): 极度恐慌 + 开始衰竭
        # 必须满足: (某一个指标极端高但今日下降) 且 (另一个指标不继续恶化)
        buy_pulse = (
            ((vix_z > 1.5) & (vix_diff < 0) & (hy_diff_3d <= 0)) |
            ((hy_z > 1.5) & (hy_diff < 0) & (vix_diff <= 0))
        )
        
        # 4. 看空脉冲 (-1.0): 恐慌升温初期 (非极端，处于温水煮青蛙或主跌启动阶段)
        # 必须满足: VIX单日暴涨但远未到极值，且信用利差同期走阔; 或者信用利差刚刚向上跨越1.0预警线
        sell_pulse = (
            ((vix_pct > 0.10) & (hy_diff_3d > 0) & (vix_z < 1.5) & (vix_z > 0.5)) |
            ((hy_z > 1.0) & (hy_z.shift(1) <= 1.0) & (vix_diff > 0))
        )
        
        # 5. 组合脉冲信号
        signal.loc[buy_pulse] = 1.0
        signal.loc[sell_pulse] = -1.0
        
        # 确保全空和异常数据被处理为0.0
        signal = signal.fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"