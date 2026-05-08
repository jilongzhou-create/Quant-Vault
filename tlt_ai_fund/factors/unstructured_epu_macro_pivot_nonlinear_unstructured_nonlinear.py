import numpy as np
import pandas as pd

class UnstructuredEpuMacroPivotNonlinearFactor:
    """政策预期突变高维交叉因子 (unstructured/nonlinear)

    逻辑: 捕捉极端政策不确定性衰竭与市场降息/加息预期突变的非线性共振。当且仅当 EPU(新闻政策不确定性) 达到极度恐慌且开始回落时，若同期 2年期美债收益率(dgs2) 急剧下行且收益率曲线(t10y2y) 急剧变陡，说明多重维度的政策转向(Policy Pivot)已经确立，美联储即将降息救市，此时买入 TLT 具有极高胜率。此因子将非结构化新闻情绪与纯宏观市场定价进行高维交叉，避免单一指标的假突破。
    数据: usepuindxd (非结构化EPU), dgs2 (短端利率), t10y2y (长短端利差)
    触发: 
      - 条件1 (衰竭铁律): usepuindxd 的 Z-Score > 2.5 (极度恐慌) 且其单日 diff() < 0 (恐慌开始衰竭，拒绝接飞刀)
      - 条件2 (边际铁律): dgs2 的5日变化量 Z-Score < -2.5 (短端利率剧烈暴跌，市场陡然 Price-in 降息)
      - 条件3 (边际铁律): t10y2y 的5日变化量 Z-Score > 2.0 (收益率曲线急剧 Bull Steepening 变陡)
      三者在 5 日短窗口内共振触发 +1.0 脉冲，并维持极短的 3 天持仓以捕捉主升浪。
    输出: 严格的狙击手级脉冲信号 [-1.0, 1.0]。正值看多美债，负值看空美债，常态下强制休眠 (0.0)。
    """

    def __init__(self):
        self.name = 'unstructured_epu_macro_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 基础数据校验与清洗 (缺失列安全返回全0)
        req_cols = ['usepuindxd', 'dgs2', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index, name=self.name)
            
        df = data[req_cols].ffill()

        # 2. 非结构化数据维: EPU 极端脉冲与二阶导数衰竭 (严格遵循铁律2)
        epu = df['usepuindxd']
        epu_mean = epu.rolling(window=252, min_periods=60).mean()
        epu_std = epu.rolling(window=252, min_periods=60).std() + 1e-6
        epu_z = (epu - epu_mean) / epu_std
        
        # 恐慌极值 + 恐慌见顶回落
        epu_panic_exhaustion = (epu_z > 2.5) & (epu.diff(1) < 0)
        # 狂热极值 + 狂热见底回升 (EPU右偏分布下, 负向极值同样需要严格衰竭条件)
        epu_complacency_exhaustion = (epu_z < -2.5) & (epu.diff(1) > 0)

        # 3. 市场定价维: 短端利率动量变化 (严格遵循铁律3)
        dgs2 = df['dgs2']
        dgs2_diff5 = dgs2.diff(5)
        dgs2_diff_mean = dgs2_diff5.rolling(window=252, min_periods=60).mean()
        dgs2_diff_std = dgs2_diff5.rolling(window=252, min_periods=60).std() + 1e-6
        dgs2_z = (dgs2_diff5 - dgs2_diff_mean) / dgs2_diff_std
        
        # dgs2 暴跌 = 降息预期骤升; dgs2 暴涨 = 加息预期骤升
        dgs2_plunge = dgs2_z < -2.5
        dgs2_spike = dgs2_z > 2.5

        # 4. 市场定价维: 收益率曲线形态动量变化 (严格遵循铁律3)
        t10y2y = df['t10y2y']
        t10y2y_diff5 = t10y2y.diff(5)
        t10y2y_diff_mean = t10y2y_diff5.rolling(window=252, min_periods=60).mean()
        t10y2y_diff_std = t10y2y_diff5.rolling(window=252, min_periods=60).std() + 1e-6
        t10y2y_z = (t10y2y_diff5 - t10y2y_diff_mean) / t10y2y_diff_std
        
        # 曲线急剧变陡(Bull Steepening)确认降息; 曲线急剧平坦化/倒挂确认加息
        curve_steepen = t10y2y_z > 2.0
        curve_flatten = t10y2y_z < -2.0

        # 5. 非线性特征交叉逻辑 (允许宏观定价在 EPU 衰竭前的 5 日内发生以捕捉共振)
        dgs2_plunge_recent = dgs2_plunge.rolling(window=5).max() > 0
        curve_steepen_recent = curve_steepen.rolling(window=5).max() > 0
        
        dgs2_spike_recent = dgs2_spike.rolling(window=5).max() > 0
        curve_flatten_recent = curve_flatten.rolling(window=5).max() > 0

        # 跨域共振触发条件
        long_trigger = epu_panic_exhaustion & dgs2_plunge_recent & curve_steepen_recent
        short_trigger = epu_complacency_exhaustion & dgs2_spike_recent & curve_flatten_recent

        # 6. 零值休眠与狙击手脉冲生成 (严格遵循铁律1)
        signal = pd.Series(0.0, index=df.index)
        signal.loc[long_trigger] = 1.0
        signal.loc[short_trigger] = -1.0

        # 脉冲后延 2 天 (总计 3 天窗口)，确保抓取事件发酵期的主升浪，并将 Trigger Rate 控制在 5%-15% 健康区间
        signal = signal.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"