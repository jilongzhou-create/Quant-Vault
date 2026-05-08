import numpy as np
import pandas as pd

class CrossAssetVolCurvePivotFactor:
    """波动率极值衰竭与收益率曲线共振因子 (volatility/nonlinear)

    逻辑: 结合跨资产波动率极值与收益率曲线动量，捕捉宏观流动性拐点。当恐慌极端且边际消退时，若短端利率加速下行导致曲线牛陡，视为美联储宽松预期落地，强力看多美债(此时因子的Conditional IC将大幅提升，避免单一买入VIX极值造成的接飞刀)；反之当市场极度自满且突然被打破时，若曲线熊平，视为紧缩冲击，看空美债。因子使用移动平均交叉严格控制脉冲属性。
    数据: vixcls, gvzcls, usepuindxd, t10y2y, dgs2
    触发: 恐慌指标Z>1.25且向下交叉均线(衰竭) + 曲线动量Z>0.5且短端下行(牛陡) -> +1.0
    输出: 仅在宏观定价突变的脉冲点输出 +1.0/-1.0，其余时间为 0.0
    """

    def __init__(self):
        self.name = 'cross_asset_vol_curve_pivot'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 缺失列检查
        required_cols = ['vixcls', 'gvzcls', 'usepuindxd', 't10y2y', 'dgs2']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0.0, index=data.index, name=self.name)

        if len(data) < 252:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 数据前向填充以防止缺失值干扰
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        epu = data['usepuindxd'].ffill()
        curve = data['t10y2y'].ffill()
        dgs2 = data['dgs2'].ffill()

        # Step 1: 极值判断 (252日 Z-Score)
        # 加入极小值平滑防止除以0
        vix_z = (vix - vix.rolling(252).mean()) / (vix.rolling(252).std() + 1e-8)
        gvz_z = (gvz - gvz.rolling(252).mean()) / (gvz.rolling(252).std() + 1e-8)
        epu_z = (epu - epu.rolling(252).mean()) / (epu.rolling(252).std() + 1e-8)

        # Step 2: 衰竭/边际变化特征 (短周期均线交叉，严格遵循边际变化铁律)
        vix_ma = vix.rolling(5).mean()
        gvz_ma = gvz.rolling(5).mean()
        epu_ma = epu.rolling(5).mean()

        def crossover_down(series, ma):
            # 严格脉冲：当天下穿均线 (代表极端情绪突然衰竭的瞬间)
            return (series < ma) & (series.shift(1) >= ma.shift(1))

        def crossover_up(series, ma):
            # 严格脉冲：当天上穿均线 (代表自满情绪突然恶化的瞬间)
            return (series > ma) & (series.shift(1) <= ma.shift(1))

        # 恐慌极值 + 衰竭脉冲 (看多宏观避险准备)
        vol_fear = (vix_z > 1.25) & crossover_down(vix, vix_ma)
        gvz_fear = (gvz_z > 1.25) & crossover_down(gvz, gvz_ma)
        epu_fear = (epu_z > 1.25) & crossover_down(epu, epu_ma)
        fear_exhaustion = vol_fear | gvz_fear | epu_fear
        
        # 自满极值 + 被打破脉冲 (看空紧缩冲击准备)
        vol_comp = (vix_z < -0.75) & crossover_up(vix, vix_ma)
        gvz_comp = (gvz_z < -0.75) & crossover_up(gvz, gvz_ma)
        epu_comp = (epu_z < -0.75) & crossover_up(epu, epu_ma)
        comp_break = vol_comp | gvz_comp | epu_comp

        # 赋予脉冲信号5天的有效窗口，以等待债券市场的确认
        # 这是为了解决跨资产流动性传导的时滞问题
        fear_exhaustion_window = fear_exhaustion.rolling(5).max() == 1
        comp_break_window = comp_break.rolling(5).max() == 1

        # Step 3: 收益率曲线动量验证 (防接飞刀，二阶导数核心约束)
        # 10天动量，捕捉短端货币政策预期的剧烈再定价
        curve_diff = curve.diff(10)
        curve_diff_z = (curve_diff - curve_diff.rolling(252).mean()) / (curve_diff.rolling(252).std() + 1e-8)
        dgs2_diff = dgs2.diff(10)

        # Bull Steepening (牛陡): 曲线突然陡峭，且绝对短端(2Y)利率实质性下降 -> 美联储降息预期爆拉
        bull_steep = (curve_diff_z > 0.5) & (dgs2_diff < 0)
        
        # Bear Flattening (熊平): 曲线突然平坦/倒挂，且绝对短端(2Y)利率实质性上升 -> 美联储鹰派超预期
        bear_flatten = (curve_diff_z < -0.5) & (dgs2_diff > 0)

        # Step 4: 综合交叉生成最终脉冲信号
        long_cond = fear_exhaustion_window & bull_steep
        short_cond = comp_break_window & bear_flatten

        signal = pd.Series(0.0, index=data.index)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        # 清理异常值及未触发的NaN
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(...)"