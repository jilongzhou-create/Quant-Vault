import numpy as np
import pandas as pd

class CrossAssetVolPivotFactor:
    """波动率极值与政策枢纽交叉因子 (unstructured/options)

    逻辑: 区分两种截然不同的宏观冲击机制。
         1. 增长恐慌 (Growth Shock): 当股市恐慌(VIX极值)开始衰竭，且短端利率(DGS2)下行时，确认美联储已被迫转向鸽派救市，此时脉冲看多美债(TLT)。
         2. 通胀/避险冲击 (Inflation Shock): 当避险与通胀恐慌(GVZ极值)开始衰竭，但短端利率依然坚挺上行时，确认通胀粘性打破了降息预期，美联储维持鹰派，此时脉冲看空美债(TLT)。
    数据: vixcls (股市恐慌), gvzcls (黄金避险恐慌), dgs2 (对政策预期最敏感的短端指标)
    触发: VIX Z-Score > 1.0 + 跌破3日均线 + DGS2 5日下降超2bps -> +1.0
          GVZ Z-Score > 1.0 + 跌破3日均线 + DGS2 5日上升超2bps -> -1.0
    输出: 严格的狙击手级脉冲信号，[-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'cross_asset_vol_pivot'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 验证必需字段是否存在
        required_cols = ['vixcls', 'gvzcls', 'dgs2']
        for col in required_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index)

        # 前向填充数据，处理非交易日或缺失值
        vix = data['vixcls'].ffill()
        gvz = data['gvzcls'].ffill()
        dgs2 = data['dgs2'].ffill()

        # 定义宏观状态窗口 (63个交易日约为一个季度，能快速适应最新的波动率水位)
        window_z = 63
        
        # 1. 计算边际极值 (Z-Scores)
        vix_mean = vix.rolling(window=window_z, min_periods=10).mean()
        vix_std = vix.rolling(window=window_z, min_periods=10).std().replace(0, np.nan)
        vix_z = (vix - vix_mean) / vix_std

        gvz_mean = gvz.rolling(window=window_z, min_periods=10).mean()
        gvz_std = gvz.rolling(window=window_z, min_periods=10).std().replace(0, np.nan)
        gvz_z = (gvz - gvz_mean) / gvz_std

        # 2. 二阶导数/衰竭条件 (绝对禁止接飞刀，必须等指标跌破3日均线确认脉冲结束)
        vix_exhaust = vix < vix.rolling(window=3, min_periods=1).mean()
        gvz_exhaust = gvz < gvz.rolling(window=3, min_periods=1).mean()

        # 3. 边际变化条件 (使用5日一阶差分捕捉货币政策预期的真实突变方向，过滤掉每日绝对水位的噪音)
        dgs2_diff = dgs2.diff(5)

        # 初始化零值信号 (严格遵守零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)

        # ---------------- 触发逻辑判定 ----------------
        
        # 看多美债 (+1.0): 股市恐慌超1个标准差 + 恐慌见顶回落 + 短端利率明确下行(<-2bps, 降息预期Price-in)
        long_cond = (vix_z > 1.0) & vix_exhaust & (dgs2_diff < -0.02)
        
        # 看空美债 (-1.0): 通胀/地缘恐慌超1个标准差 + 恐慌见顶回落 + 短端利率反而上行(>2bps, 鹰派预期Price-in)
        short_cond = (gvz_z > 1.0) & gvz_exhaust & (dgs2_diff > 0.02)

        # 赋值脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        # 处理极其罕见的同日冲突 (防呆设计)
        conflict = long_cond & short_cond
        signal.loc[conflict] = 0.0

        signal.name = self.name
        
        # 处理最初计算窗口期的 NaN 为 0.0
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"