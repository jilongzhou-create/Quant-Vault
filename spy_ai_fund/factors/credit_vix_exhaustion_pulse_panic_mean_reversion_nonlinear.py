import numpy as np
import pandas as pd

class CreditVixExhaustionPulseFactor:
    """Credit and Volatility Panic Exhaustion Pulse (panic_mean_reversion/nonlinear)

    逻辑: 极度恐慌产生抄底买点, 但绝对不能接飞刀, 需等待高收益债信用利差与VIX同时见顶回落(极值+二阶导衰竭)。同时, 信用利差与VIX缓慢双升但未达极值时, 为钝刀割肉的看空信号。
    数据: bamlh0a0hym2 (高收益债OAS, 衡量系统性信用风险), vixcls (VIX, 衡量股市恐慌)
    输出: +1.0 (恐慌极值且开始衰竭, 强烈看多抄底), -1.0 (轻度恐慌发酵, 趋势恶化, 看空), 0.0 (常态休眠)
    触发条件: VIX处于过去一年Top 2%极值且日内回落, 叠加HY利差连续3日收窄时触发看多脉冲。预期Trigger Rate 8%-12%。
    """

    def __init__(self):
        self.name = 'credit_vix_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        req_cols = ['bamlh0a0hym2', 'vixcls']
        if not all(col in data.columns for col in req_cols):
            return pd.Series(0.0, index=data.index, name=self.name)

        # 宏观数据前向填充以对齐交易日(最多填5天)
        df = data[req_cols].ffill(limit=5)
        
        hy_spread = df['bamlh0a0hym2']
        vix = df['vixcls']
        
        # 计算 VIX 252日(约一年) Z-Score, 识别绝对的恐慌极值点
        vix_mean_252 = vix.rolling(window=252).mean()
        vix_std_252 = vix.rolling(window=252).std()
        vix_z252 = (vix - vix_mean_252) / vix_std_252
        
        # 1. 狙击买入信号 (+1.0) - 满足二阶导数铁律: 极值 + 衰竭
        # - VIX Z-Score > 2.0 (极度恐慌, 处于历史高位)
        # - vix.diff(1) < 0 (今日VIX下跌, 恐慌开始回落)
        # - hy_spread.diff(3) < 0 (高收益债利差在过去3天内收窄, 信用市场确认恐慌衰退)
        buy_condition = (
            (vix_z252 > 2.0) & 
            (vix.diff(1) < 0) & 
            (hy_spread.diff(3) < 0)
        )
        
        # 2. 趋势看空信号 (-1.0) - 钝刀割肉阶段
        # - hy_spread.diff(5) > 0.25 (信用利差5天内走阔超25个基点, 信用环境实质性恶化)
        # - vix.diff(5) > 2.0 (VIX 5天内抬升超2点, 恐慌逐步积累)
        # - vix_z252 < 1.5 (VIX尚未达到极端洗盘位置, 此时做空不会死于主跌浪末期的深V反弹)
        # - vix.diff(1) > 0 (今日VIX仍在上升, 动量向下)
        sell_condition = (
            (hy_spread.diff(5) > 0.25) & 
            (vix.diff(5) > 2.0) & 
            (vix_z252 < 1.5) & 
            (vix.diff(1) > 0)
        )
        
        # 零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        signal[buy_condition] = 1.0
        signal[sell_condition] = -1.0
        
        # 补充清理: 确保在任何全量数据不足的初始阶段不输出噪音
        signal.iloc[:252] = 0.0
        
        signal.name = self.name
        return signal