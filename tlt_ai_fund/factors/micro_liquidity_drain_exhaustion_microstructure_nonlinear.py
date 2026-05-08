import numpy as np
import pandas as pd

class MicrostructureStirOvershootFactor:
    """Microstructure STIR Overshoot Factor (microstructure/nonlinear)

    逻辑: 捕捉短期利率市场(STIR)对美联储加息/降息预期的非理性过度定价及其衰竭反转点。
         当2年期美债收益率相对于联邦基金利率发生剧烈偏离(极端定价加息或降息预期)，
         且这种单向动量开始出现衰竭(低于/高于3日均值)时，产生狙击脉冲。
         配合VIX的非线性交叉过滤，确保只在宏观情绪与利率预期的转折共振点触发，避免接飞刀。
    数据: dgs2 (2年期美债), dff (联邦基金有效利率), vixcls (VIX波动率)
    触发: 2Y-FF利差的5日边际动量 Z-Score 绝对值 > 1.2 + 动量反向越过3日均值(衰竭) + VIX状态非线性过滤
    输出: +1.0 (鹰派超卖衰竭，脉冲看多TLT) / -1.0 (鸽派超买衰竭，脉冲看空TLT)
    """

    def __init__(self):
        self.name = 'microstructure_stir_overshoot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 - 初始信号全为0.0，严格狙击脉冲
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖数据是否存在
        required_cols = ['dgs2', 'dff', 'vixcls']
        for col in required_cols:
            if col not in data.columns:
                return signal

        # 处理非交易日或缺失数据
        dgs2 = data['dgs2'].ffill()
        dff = data['dff'].ffill()
        vix = data['vixcls'].ffill()

        # 铁律3: 边际变化 - 绝对禁止使用绝对水位，计算微观定价利差的5日动量
        spread = dgs2 - dff
        mom5 = spread.diff(5)

        # 动量的252日Z-Score极值判断 (约1年窗口)
        mom_mean = mom5.rolling(252).mean()
        mom_std = mom5.rolling(252).std()
        z_mom = (mom5 - mom_mean) / mom_std

        # VIX 非线性交叉环境过滤
        vix_mean = vix.rolling(252).mean()
        vix_std = vix.rolling(252).std()
        vix_z = (vix - vix_mean) / vix_std

        # 铁律2: 二阶导数衰竭 - 绝对禁止单边买入，必须等待动量指标开始反转(衰竭)
        mom_exhaustion_long = mom5 < mom5.rolling(3).mean()
        mom_exhaustion_short = mom5 > mom5.rolling(3).mean()

        # 多头触发 (看多美债/TLT):
        # 1. z_mom > 1.2: 短端利率(dgs2)相对于基准(dff)快速飙升，市场极度恐慌性计价加息，TLT暴跌
        # 2. mom_exhaustion_long: 鹰派飙升动量开始回落，抛压见顶 (二阶衰竭)
        # 3. vix_z > 0.0: 市场整体处于承压状态，存在潜在的避险买盘(Flight-to-safety)共振
        long_cond = (z_mom > 1.2) & mom_exhaustion_long & (vix_z > 0.0)

        # 空头触发 (看空美债/TLT):
        # 1. z_mom < -1.2: 短端利率快速崩塌，市场极度乐观地计价降息，TLT暴涨出现FOMO
        # 2. mom_exhaustion_short: 鸽派崩塌动量开始反转向上，抢跑降息的买盘见顶 (二阶衰竭)
        # 3. vix_z < 0.0: 市场整体自满(VIX低位)，避险情绪消退，无法提供持续买盘支撑
        short_cond = (z_mom < -1.2) & mom_exhaustion_short & (vix_z < 0.0)

        # 赋值脉冲信号 (+1.0 / -1.0)
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"