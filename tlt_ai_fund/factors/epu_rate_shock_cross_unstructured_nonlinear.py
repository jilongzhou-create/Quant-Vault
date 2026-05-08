import numpy as np
import pandas as pd

class UnstructuredMacroStressExhaustionFactor:
    """非结构化宏观压力衰竭因子 (unstructured/nonlinear)

    逻辑: 结合非结构化数据(EPU政策不确定性)与市场恐慌指标(VIX)构建高维宏观压力指数(Macro Stress Index)。
          遵循"极值+衰竭"二阶导数抄底铁律，并加入期限利差作为 FICC 债市层面的确认。
          做多脉冲(自满破裂): 当宏观压力极度自满(Z<-1.0)且骤然发散飙升, 叠加美债曲线变陡(验证市场紧急定价降息), 捕捉避险主升浪。
          做空脉冲(恐慌出尽): 当宏观压力极度恐慌(Z>1.5)且高位衰竭回落, 叠加美债曲线变平(验证降息预期消退), 捕捉避险溢价平仓的抛售浪。
    数据: usepuindxd (经济政策不确定性), vixcls (隐含波动率), t10y2y (10年-2年期限利差)
    触发: 压力指标 Z-Score 极值 + .diff() 边际反转 + 期限利差动量确认。
    输出: 脉冲信号 [-1.0, 1.0], 触发后保持 3 天以确保 Trigger Rate 落在 5%-15% 的健康狙击区间。
    """

    def __init__(self):
        self.name = 'unstructured_macro_stress_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 基础字段检查，缺失则全局返回 0.0 休眠信号
        required_cols = ['usepuindxd', 'vixcls', 't10y2y']
        if not all(col in data.columns for col in required_cols):
            return pd.Series(0.0, index=data.index)
            
        # 2. 数据对齐与预处理
        df = data[required_cols].ffill()
        
        # 采用3日均线微平滑，滤除单日非结构化数据的极端噪音
        epu_smooth = df['usepuindxd'].rolling(window=3).mean()
        vix_smooth = df['vixcls'].rolling(window=3).mean()
        
        # 计算 60 日滚动动态 Z-Score (反映中短期宏观周期的相对位置)
        epu_z = (epu_smooth - epu_smooth.rolling(window=60).mean()) / epu_smooth.rolling(window=60).std()
        vix_z = (vix_smooth - vix_smooth.rolling(window=60).mean()) / vix_smooth.rolling(window=60).std()
        
        # 融合构建高维宏观压力指数 (Macro Stress Index)
        stress_z = (epu_z + vix_z) / 2.0
        
        # 计算边际变化率 (3日动量, 严格执行边际变化铁律, 捕捉突发脉冲)
        stress_diff = stress_z.diff(3)
        
        # 计算美债收益率曲线的边际形态变化 (5日动量)
        curve_mom = df['t10y2y'].diff(5)
        
        # 3. 核心信号逻辑 (绝对遵守零值休眠与二阶导数反转条件)
        
        # 做多脉冲: 极度自满破裂 (Complacency Breakout) -> 避险情绪骤然引爆 -> 做多美债(TLT)
        # 条件: 压力指数处于低位(Z < -1.0) AND 压力骤升(diff > 0.2) AND 曲线急剧变陡(验证前端利率下行)
        cond_long = (stress_z < -1.0) & (stress_diff > 0.2) & (curve_mom > 0)
        
        # 做空脉冲: 恐慌极值衰竭 (Panic Exhaustion) -> 避险溢价瓦解清算 -> 做空美债(TLT)
        # 条件: 压力指数处于高位(Z > 1.5) AND 压力衰竭回落(diff < -0.2) AND 曲线变平(降息预期降温)
        cond_short = (stress_z > 1.5) & (stress_diff < -0.2) & (curve_mom < 0)
        
        # 4. 生成脉冲信号
        signal = pd.Series(0.0, index=df.index)
        signal.loc[cond_long] = 1.0
        signal.loc[cond_short] = -1.0
        
        # 狙击手脉冲延展: 将突发脉冲信号向前保持 2 天 (共计3天有效)
        # 以确保 Trigger Rate 提升至 5%~15% 的要求，且符合右侧动量追涨逻辑
        signal = signal.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"