import numpy as np
import pandas as pd

class VolatilityCurveMicrostructureFactor:
    """波动率曲线微观结构反转因子 (volatility/options)

    逻辑: 结合跨资产波动率极值衰竭与收益率曲线动量，捕捉美债的极值反转脉冲。
          在宏观流动性冲击期间(VIX极高)，避险资产美债可能因为"Dash for Cash"被无差别抛售。
          当股市恐慌(VIX Z-Score > 2.5)与黄金波动(GVZ)同步见顶回落，且伴随收益率曲线近期
          开始陡峭化(降息/避险预期发酵)，标志着流动性冲击结束，长端美债迎来确定性的绝佳做多脉冲。
          反之，当收益率曲线遭遇极端暴力熊平(短端利率暴涨，动量 Z-Score < -2.5)，
          且重定价刚过极值点(二阶导企稳)，若同时缺乏VIX恐慌(说明经济较热而非危机)，
          则确立"Higher for Longer"逻辑，产生做空长债脉冲。
          
    数据: vixcls (VIX指数), gvzcls (黄金ETF隐含波动率), t10y2y (期限利差)
    触发:
        看多 (+1.0): VIX 252日 Z-Score > 2.5 AND VIX < 3日均值 AND VIX日变化 < 0 AND GVZ日变化 < 0 AND t10y2y(5日动量) > 0
        看空 (-1.0): t10y2y(5日动量) 252日 Z-Score < -2.5 AND 动量二阶导 > 0 (衰竭) AND VIX Z-Score < 0.5
    输出: 狙击手级脉冲信号 [-1.0, 1.0]，常态严格为 0.0 (满足三大铁律)
    """

    def __init__(self):
        self.name = 'volatility_curve_microstructure_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 数据校验处理 (必须处理缺失字段)
        req_cols = ['vixcls', 'gvzcls', 't10y2y']
        if not all(col in data.columns for col in req_cols):
            return pd.Series(0.0, index=data.index, name=self.name)

        df = data[req_cols].ffill()
        signal = pd.Series(0.0, index=df.index)

        # 2. VIX 波动率极值与微观结构衰竭计算 (遵守铁律2：二阶导数防飞刀)
        vix = df['vixcls']
        vix_252d_mean = vix.rolling(window=252).mean()
        vix_252d_std = vix.rolling(window=252).std().replace(0, 1e-5)
        vix_zscore = (vix - vix_252d_mean) / vix_252d_std

        # 恐慌衰竭条件: 跌破3日均线 且 当日确切回落
        vix_exhausted = (vix < vix.rolling(window=3).mean()) & (vix.diff(1) < 0)

        # 3. 跨资产确认: 黄金波动率同步回落 (排除单一资产的噪音)
        gvz_exhausted = df['gvzcls'].diff(1) < 0

        # 4. 收益率曲线动量微观结构 (遵守铁律3：边际变化，绝对禁用水平值)
        curve = df['t10y2y']
        # 使用5日差分捕捉曲线短期"斜率动能"(Marginal Velocity)
        curve_mom = curve.diff(5)
        curve_mom_mean = curve_mom.rolling(window=252).mean()
        curve_mom_std = curve_mom.rolling(window=252).std().replace(0, 1e-5)
        curve_mom_zscore = (curve_mom - curve_mom_mean) / curve_mom_std

        # 曲线极端熊平冲击衰竭 (Hawkish Shock)
        # 抄底铁律逻辑应用至做空端: 极端的平坦化动量 (Z < -2.5) + 动量开始收敛发散 (diff > 0，防止逆势接飞刀)
        bear_flattening_exhausted = (curve_mom_zscore < -2.5) & (curve_mom.diff(1) > 0)

        # --- 信号逻辑合成 (遵守铁律1：零值休眠，严控Trigger Rate) ---

        # 看多脉冲 (+1.0): 跨资产恐慌(VIX+GVZ)极高后同步瓦解衰竭 + 收益率曲线动能确认为正(陡峭化避险定价)
        long_pulse = (vix_zscore > 2.5) & vix_exhausted & gvz_exhausted & (curve_mom > 0)

        # 看空脉冲 (-1.0): 暴力的紧缩预期重定价(曲线平坦化动能极值)刚刚达峰衰竭 + 股市相对平稳无严重恐慌对冲
        short_pulse = bear_flattening_exhausted & (vix_zscore < 0.5)

        signal.loc[long_pulse] = 1.0
        signal.loc[short_pulse] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"