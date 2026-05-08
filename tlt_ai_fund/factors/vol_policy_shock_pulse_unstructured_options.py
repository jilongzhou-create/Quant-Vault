import numpy as np
import pandas as pd

class VolPolicyShockPulseFactor:
    """波动率与政策预期极值脉冲因子 (unstructured/options)

    逻辑: 结合波动率微观结构(VIX)的极端情绪衰竭与宏观政策预期(DGS2/T10Y2Y)的瞬间突变，捕捉美债(TLT)拐点。常态休眠。
          1. 恐慌衰竭脉冲：VIX极度飙升(Z>2.5)且今日开始回落(防止流动性接飞刀)，标志无差别抛售结束，安全生息资产买盘回归。
          2. 政策转向脉冲：2年期收益率下行且曲线骤然变陡(Z>2.5)，确认牛陡(Bull Steepening)，联储货币政策由鹰转鸽突变。
    数据: vixcls, t10y2y, dgs2
    触发: VIX Z-Score > 2.5 且 VIX < 3日均值 (二阶导数衰竭) 或 t10y2y 5日动量 Z-Score > 2.5 (边际变化突发)。
    输出: 脉冲型 [-1.0, 1.0]。+1.0为降息交易/恐慌见顶(看多TLT)，-1.0为紧缩恐慌/极度自满被打破(看空TLT)。
    """

    def __init__(self):
        self.name = 'vol_policy_shock_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (常态信号必须为 0.0)
        signal = pd.Series(0.0, index=data.index)
        
        # 数据依赖检查
        req_cols = ['vixcls', 't10y2y', 'dgs2']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 填补可能存在的数据前瞻空缺
        vix = data['vixcls'].ffill()
        t10y2y = data['t10y2y'].ffill()
        dgs2 = data['dgs2'].ffill()

        # ==========================================
        # 模块A: 波动率微观结构 (极值恐慌与衰竭)
        # ==========================================
        vix_mean = vix.rolling(252, min_periods=60).mean()
        vix_std = vix.rolling(252, min_periods=60).std()
        vix_z = (vix - vix_mean) / vix_std
        
        # 铁律2: 二阶导数 (必须等待极端恐慌开始衰竭，绝对禁止直接接飞刀)
        vix_exhaustion = (vix < vix.rolling(3).mean()) & (vix.diff(1) < 0)
        vol_panic_buy = (vix_z > 2.5) & vix_exhaustion
        
        # 衍生空头: 市场极度自满 (VIX极低) 突然被打破，避险与去杠杆开启
        vix_complacent = (vix_z < -1.5)
        vix_erupting = (vix > vix.rolling(3).mean()) & (vix.diff(1) > 0)
        vol_complacency_sell = vix_complacent & vix_erupting
        
        # ==========================================
        # 模块B: 政策预期突变 (陡峭化/扁平化冲击)
        # ==========================================
        # 铁律3: 边际变化 (绝对禁止用利差绝对值判定，必须看瞬时动量突变)
        steep_mom = t10y2y.diff(5)
        steep_mean = steep_mom.rolling(252, min_periods=60).mean()
        steep_std = steep_mom.rolling(252, min_periods=60).std()
        steep_z = (steep_mom - steep_mean) / steep_std
        
        # 政策前端对预期的反应
        dgs2_mom = dgs2.diff(5)
        
        # 脉冲多头: 牛陡 (Bull Steepening) 突变
        # 逻辑: 曲线边际极度变陡 (Z>2.5) 且是由短端利率(DGS2)快速下行驱动，代表急剧的降息Price-in
        policy_pivot_buy = (steep_z > 2.5) & (dgs2_mom < 0) & (t10y2y.diff(1) > 0)
        
        # 脉冲空头: 熊平/倒挂 (Bear Flattening) 突变
        # 逻辑: 曲线边际极度倒挂/变平 (Z<-2.5) 且由短端飙升驱动，代表超预期鹰派加息冲击
        policy_tighten_sell = (steep_z < -2.5) & (dgs2_mom > 0) & (t10y2y.diff(1) < 0)
        
        # ==========================================
        # 狙击手信号合成 (Sniper Target)
        # ==========================================
        # 多头触发
        signal[vol_panic_buy | policy_pivot_buy] = 1.0
        
        # 空头触发
        signal[vol_complacency_sell | policy_tighten_sell] = -1.0
        
        # 防御性过滤: 多空逻辑意外发生同日碰撞时，撤销操作保持休眠
        conflict = (vol_panic_buy | policy_pivot_buy) & (vol_complacency_sell | policy_tighten_sell)
        signal[conflict] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"