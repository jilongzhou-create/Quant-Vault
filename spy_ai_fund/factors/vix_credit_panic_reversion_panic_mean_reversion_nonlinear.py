import numpy as np
import pandas as pd

class VixCreditPanicReversionFactor:
    """股债双杀恐慌衰竭交叉因子 (panic_mean_reversion/nonlinear)

    逻辑: 结合VIX与高收益债信用利差(OAS)。当股市与信用市场同时陷入极端恐慌(Z-Score显著大于1)，且两者均在今日出现环比回落(恐慌衰竭)，触发抄底买入；若在轻中度恐慌区间连续攀升，且信用利差走阔，则输出趋势恶化的看空信号。
    数据: [vixcls, bamlh0a0hym2]
    输出: +1.0(恐慌极值+衰竭，强烈抄底), -1.0(轻度恐慌恶化，趋势向下)
    触发条件: 买入(VIX Z>1.5, CRED Z>1.2, 两者diff<=0); 卖出(0.5<VIX Z<=1.5, VIX连续2日走高, CRED走阔)。预期 Trigger Rate: 6-10%。
    """

    def __init__(self, window: int = 252):
        self.name = 'vix_credit_panic_reversion'
        self.window = window

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['vixcls', 'bamlh0a0hym2']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        vix = data['vixcls'].ffill()
        cred = data['bamlh0a0hym2'].ffill()
        
        # 计算 252日 Z-Score 以判定极值状态
        vix_mean = vix.rolling(self.window, min_periods=min(self.window, 20)).mean()
        vix_std = vix.rolling(self.window, min_periods=min(self.window, 20)).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)
        
        cred_mean = cred.rolling(self.window, min_periods=min(self.window, 20)).mean()
        cred_std = cred.rolling(self.window, min_periods=min(self.window, 20)).std()
        cred_z = (cred - cred_mean) / cred_std.replace(0, np.nan)
        
        # 计算边际变化，捕捉极值后的转折
        vix_diff = vix.diff(1)
        cred_diff = cred.diff(1)
        vix_diff_shift = vix_diff.shift(1)
        
        # 条件1：极度恐慌 + 恐慌衰竭 (抄底脉冲)
        buy_cond = (
            (vix_z > 1.5) & 
            (cred_z > 1.2) & 
            (vix_diff < 0) & 
            (vix < vix.rolling(3).mean()) & 
            (cred_diff <= 0)
        )
        
        # 条件2：轻中度恐慌 + 势头持续恶化 (卖出脉冲，防钝刀割肉)
        sell_cond = (
            (vix_z > 0.5) & 
            (vix_z <= 1.5) & 
            (vix_diff > 0) & 
            (vix_diff_shift > 0) & 
            (cred_diff > 0)
        )
        
        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0
        
        signal.name = self.name
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window})"