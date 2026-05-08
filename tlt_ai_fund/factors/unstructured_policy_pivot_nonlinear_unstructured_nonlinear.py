import numpy as np
import pandas as pd

class UnstructuredPolicyPivotNonlinearFactor:
    """Unstructured Policy Pivot Shock (unstructured/nonlinear)

    逻辑: 捕捉由于极端政策不确定性(EPU)衰竭叠加美联储超预期转向(FOMC鸽派突变/短端利率暴跌)带来的美债长端脉冲做多机会。只在预期突变的边际瞬间产生脉冲信号，常态下为0。
    数据: fomc_sentiment, dgs2, t10y2y, usepuindxd
    触发: 政策不确定性指数 Z-Score > 1.5 且开始回落 (衰竭) 的窗口期内，叠加 FOMC情感边际变化 Z-Score > 1.5 或 2年期美债收益率连续急降且曲线牛陡。
    输出: +1.0 (鸽派转向/看多TLT), -1.0 (鹰派转向/看空TLT), 其余非触发日为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # Rule 1: Zero value hibernation (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)
        
        # Verify required columns
        required_cols = ['fomc_sentiment', 'dgs2', 't10y2y', 'usepuindxd']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # Forward fill to handle NaNs cleanly across aligned indices
        fomc = data['fomc_sentiment'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        epu = data['usepuindxd'].ffill()
        
        # Rule 3: Marginal Change Only (Strictly forbidding absolute values for step data)
        # Using 5-day difference to capture the step jumps and sharp rate movements safely
        fomc_diff = fomc.diff(5)
        dgs2_diff = dgs2.diff(5)
        t10y2y_diff = t10y2y.diff(5)
        
        # Calculate 252-day Z-scores of the marginal changes (momentum momentum)
        # Added 1e-4 to std to avoid division by zero during flat periods (fomc step changes)
        fomc_z = (fomc_diff - fomc_diff.rolling(252).mean()) / (fomc_diff.rolling(252).std() + 1e-4)
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(252).mean()) / (dgs2_diff.rolling(252).std() + 1e-4)
        t10y2y_z = (t10y2y_diff - t10y2y_diff.rolling(252).mean()) / (t10y2y_diff.rolling(252).std() + 1e-4)
        
        # Rule 2: Anti-Catch-Falling-Knife (Extreme + Exhaustion for Panic/Uncertainty Indicator)
        # Smooth EPU slightly to handle intense daily noise 
        epu_ma = epu.ewm(span=5, adjust=False).mean()
        epu_z = (epu_ma - epu_ma.rolling(252).mean()) / (epu_ma.rolling(252).std() + 1e-4)
        
        # Exhaustion Condition: EPU was extreme recently, but is now starting to cool off
        epu_extreme = epu_z > 1.5
        epu_dropping = epu_ma < epu_ma.rolling(3).mean()
        epu_exhaustion = epu_extreme & epu_dropping
        
        # Create a valid window (15 days) where uncertainty is resolving, welcoming policy pricing
        epu_valid = epu_exhaustion.rolling(15).max() > 0
        
        # Pivot Catalysts
        # 1. Unstructured Dovish Pivot: FOMC sentiment shifts more dovish significantly
        dove_fomc = fomc_z > 1.5
        # 2. Market Pricing Dovish Pivot (Bull Steepening): DGS2 drops sharply AND curve steepens
        bull_steepening = (dgs2_z < -1.5) & (t10y2y_z > 1.0)
        
        # 3. Unstructured Hawkish Pivot: FOMC sentiment shifts hawkish
        hawk_fomc = fomc_z < -1.5
        # 4. Market Pricing Hawkish Pivot (Bear Flattening/Inverting): DGS2 spikes sharply AND curve flattens
        bear_flattening = (dgs2_z > 1.5) & (t10y2y_z < -1.0)
        
        # Nonlinear Feature Cross: Macro Uncertainty Exhaustion crossed with Policy/Yield Shock
        long_cond = (dove_fomc | bull_steepening) & epu_valid
        short_cond = (hawk_fomc | bear_flattening) & epu_valid
        
        # Generate Output Pulses (Only on trigger)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"