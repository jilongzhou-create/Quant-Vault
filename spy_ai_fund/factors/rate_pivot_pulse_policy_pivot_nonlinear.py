import numpy as np
import pandas as pd

class RatePivotPulseFactor:
    """政策转向与利率动量脉冲因子 (policy_pivot/nonlinear)

    逻辑: 捕捉市场对美联储政策预期的剧烈反转。当短端利率(DGS2)急速下跌且曲线急剧变陡(Bull Steepening)时，说明市场强力抢跑降息，点燃多头脉冲；当短端急速飙升且曲线熊平(Bear Flattening)时，说明紧缩预期突增，压制估值。多头触发时严格要求恐慌极值必须衰竭(防接飞刀)。
    数据: dgs2, t10y2y, vixcls
    输出: +1.0 看多 (鸽派流动性释放), -1.0 看空 (鹰派紧缩压制), 0.0 常态休眠
    触发条件: 5日短端利率边际变化处于过去一年的极值状态，并伴随曲线斜率异动，Trigger Rate 目标控制在 5% - 15% 之间。
    """

    def __init__(self):
        self.name = 'rate_pivot_pulse_policy_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 必须依赖的列
        required_cols = ['dgs2', 't10y2y', 'vixcls']
        
        # 处理数据缺失情况，直接返回 0.0 Series
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)
        
        # 向前填充，防止美债市场或VIX假日带来的缺失
        df = data[required_cols].ffill()
        
        # --- 边际变化铁律 ---
        # 计算 5个交易日的动量差值，捕捉阶梯式预期的瞬间改变，而非绝对水位
        dgs2_diff = df['dgs2'].diff(5)
        t10y2y_diff = df['t10y2y'].diff(5)
        
        # --- 极值定位 ---
        # 计算 252个交易日(约1年)的滚动 Z-Score 
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(252).mean()) / (dgs2_diff.rolling(252).std() + 1e-8)
        t10y2y_z = (t10y2y_diff - t10y2y_diff.rolling(252).mean()) / (t10y2y_diff.rolling(252).std() + 1e-8)
        
        # --- 二阶导数防飞刀铁律所需特征 ---
        # 计算 VIX 的 Z-Score 与动量，用来识别恐慌极值与衰竭状态
        vix_z = (df['vixcls'] - df['vixcls'].rolling(252).mean()) / (df['vixcls'].rolling(252).std() + 1e-8)
        vix_diff_2 = df['vixcls'].diff(2)
        
        # ==================================================
        # 多头脉冲触发条件：鸽派转向突变 (Bull Steepening)
        # ==================================================
        # 1. DGS2 发生极为剧烈的跳水 (Z < -1.5, 抢跑降息)
        # 2. 且收益率曲线急剧变陡 (Z > 1.0, 确认流动性宽松主轴)
        is_bull_steepening = (dgs2_z < -1.5) & (t10y2y_z > 1.0)
        
        # 3. 绝对禁止在主跌浪接飞刀：如果当前处于极端恐慌状态(VIX Z-score >= 2.0)，
        #    则必须满足 VIX 边际下行(近两天diff < 0, 即恐慌开始衰竭)才允许做多。
        vix_safe_to_buy = (vix_z < 2.0) | ((vix_z >= 2.0) & (vix_diff_2 < 0))
        
        long_cond = is_bull_steepening & vix_safe_to_buy
        
        # ==================================================
        # 空头脉冲触发条件：鹰派转向突变 (Bear Flattening)
        # ==================================================
        # 1. DGS2 发生极为剧烈的飙升 (Z > 1.5, 加息/紧缩预期大幅抬头)
        # 2. 且收益率曲线急剧变平/倒挂加深 (Z < -1.0, 确认短期压制长端，估值毒药)
        short_cond = (dgs2_z > 1.5) & (t10y2y_z < -1.0)
        
        # --- 信号合成 ---
        signal = pd.Series(0.0, index=df.index)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        # 过滤计算过程中因 rolling 等引起的 NaN
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(方向: policy_pivot, 方法: nonlinear)"