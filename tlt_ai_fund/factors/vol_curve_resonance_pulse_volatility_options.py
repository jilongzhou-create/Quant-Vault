import numpy as np
import pandas as pd

class VolCurveResonancePulseFactor:
    """波动率与曲线形变共振脉冲因子 (volatility/options)

    逻辑: 系统性恐慌与收益率曲线边际形变共振。期权隐含波动率极值反映市场对冲极度拥挤或极度自满，绝对禁止直接在极值买入(会接飞刀)。只有当跨资产波动率(股市VIX+黄金GVZ)的极值开始瓦解或苏醒，且伴随短端利率预期突变(美债利差剧烈陡峭/平坦化)时，才确立宏观流动性与加息预期的非线性拐点，触发高胜率的美债趋势反转脉冲。
    数据: vixcls (CBOE股票波动率), gvzcls (CBOE黄金波动率), t10y2y (期限利差)
    触发: 
      - 多头脉冲: VIX 252日 Z-Score > 1.5(拥挤) 且 跨资产波动率回落(二阶导数<0) 且 收益率曲线边际变陡(利差5日diff>0) -> +1.0
      - 空头脉冲: VIX 252日 Z-Score < -1.0(自满) 且 跨资产波动率苏醒(二阶导数>0) 且 收益率曲线边际变平(利差5日diff<0) -> -1.0
    输出: [-1.0, 1.0] 狙击手级极端脉冲信号
    """

    def __init__(self):
        self.name = 'vol_curve_resonance_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        req_cols = ['vixcls', 'gvzcls', 't10y2y']
        missing_cols = [col for col in req_cols if col not in data.columns]
        if missing_cols:
            return signal

        # 填充缺失值，避免因节假日跨资产停盘不一致导致的计算中断
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        t10y2y = data['t10y2y'].ffill()

        # 计算 VIX 的长期极值 (252个交易日约对应自然年，最小预热一季度63天)
        vix_mean = vix.rolling(window=252, min_periods=63).mean()
        vix_std = vix.rolling(window=252, min_periods=63).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std

        # 衰竭/苏醒条件 (微观二阶导数：动量反转确认，防接飞刀铁律)
        # 用 3 日均值代表极短期平滑均线，跌破/突破视为动量反转确认
        vix_ma3 = vix.rolling(window=3).mean()
        gvz_ma3 = gvz.rolling(window=3).mean()
        
        vix_falling = vix < vix_ma3
        gvz_falling = gvz < gvz_ma3
        
        vix_rising = vix > vix_ma3
        gvz_rising = gvz > gvz_ma3

        # 收益率曲线边际变化 (边际变化铁律)
        # diff(5) 代表一周内的形变方向，不在乎绝对水位是否倒挂，只在乎“势”
        curve_diff = t10y2y.diff(5)
        curve_steepening = curve_diff > 0.0
        curve_flattening = curve_diff < 0.0

        # 多头脉冲：VIX处于高位恐慌 + 波动率全面退潮(二阶衰竭) + 曲线变陡(降息预期共振) => 避险流动性回归，买入美债
        long_cond = (vix_z > 1.5) & vix_falling & gvz_falling & curve_steepening
        
        # 空头脉冲：VIX极度低位自满 + 波动率全面苏醒(二阶发散) + 曲线变平(加息/通胀预期突袭) => 风险偏好逆转，卖出美债
        short_cond = (vix_z < -1.0) & vix_rising & gvz_rising & curve_flattening

        # 零值休眠铁律：绝大部分时间保持0，仅在极端复合条件满足时发波
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"