import numpy as np
import pandas as pd

class MicrostructureCollateralSqueezeFactor:
    """微观结构抵押品挤兑衰竭反转因子 (microstructure/nonlinear)

    逻辑: 危机发生时，极高质量避险抵押品(3个月美债DTB3)遭到疯抢，其收益率瞬间暴跌，甚至远低于联邦基金无担保利率(DFF)，形成深负利差(Collateral Scarcity极值)。该因子结合VIX恐慌峰值，捕捉极度挤兑后"恐慌见顶且利差开始回升"的瞬间脉冲做多美债(TLT)。此时微观流动性黑洞结束，资产抛售停止，开启报复性反弹。
    数据: dtb3 (3月期国库券), dff (联邦基金利率), vixcls (VIX波动率)
    触发: 抵押品利差(DTB3-DFF) 63日 Z-Score < -1.5 且 VIX Z-Score > 1.5，同时叠加衰竭判定（VIX日内回落且小于3日均值，利差日内回升且大于3日均值）。
    输出: 脉冲型信号，触发时输出 +1.0 或 -1.0，其余状态处于0.0休眠。
    """

    def __init__(self):
        self.name = 'microstructure_collateral_squeeze_factor'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格遵守铁律1：初始常态必须为 0.0，且处理数据缺失
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['dtb3', 'dff', 'vixcls']
        for col in required_cols:
            if col not in data.columns:
                signal.name = self.name
                return signal

        # 提取原始数据并消除缺失
        dtb3 = data['dtb3'].ffill()
        dff = data['dff'].ffill()
        vix = data['vixcls'].ffill()

        # 微观结构核心指标：抵押品安全溢价利差
        # 常态下 dtb3 和 dff 十分接近；恐慌时 dtb3 被疯狂抢购导致急剧低于 dff
        collateral_spread = dtb3 - dff

        # 使用 63 日滚动窗口捕捉局部微观环境突变极值
        window = 63
        
        spread_mean = collateral_spread.rolling(window).mean()
        spread_std = collateral_spread.rolling(window).std()
        # 计算 Z-Score，规避 std=0 导致的无意义结果
        spread_z = (collateral_spread - spread_mean) / spread_std.replace(0, np.nan)

        vix_mean = vix.rolling(window).mean()
        vix_std = vix.rolling(window).std()
        vix_z = (vix - vix_mean) / vix_std.replace(0, np.nan)

        # 严格遵守铁律3与铁律2：计算边际变化与均值衰竭判定，反接飞刀！
        spread_ma3 = collateral_spread.rolling(3).mean()
        spread_diff = collateral_spread.diff()
        
        vix_ma3 = vix.rolling(3).mean()
        vix_diff = vix.diff()

        # 触发条件 1: 多头信号 (流动性黑洞与抵押品挤兑衰竭点)
        # 极值条件: 抵押品稀缺极化 (Z < -1.5) 且 恐慌飙升 (Z > 1.5)
        # 衰竭条件: 挤兑开始缓解 (利差边际回升) 且 恐慌同步回落
        long_cond = (
            (spread_z < -1.5) &
            (vix_z > 1.5) &
            (collateral_spread > spread_ma3) &
            (spread_diff > 0) &
            (vix < vix_ma3) &
            (vix_diff < 0)
        )

        # 触发条件 2: 空头信号 (微观环境极其宽松，避险情绪彻底消失后的逆转)
        # 极值条件: 抵押品溢价消失甚至转强 (Z > 1.5) 且 极度贪婪无恐慌 (Z < -1.5)
        # 衰竭条件: 风险偏好突然破裂，恐慌低位爆拉 且 流动性边际变紧收缩
        short_cond = (
            (spread_z > 1.5) &
            (vix_z < -1.5) &
            (collateral_spread < spread_ma3) &
            (spread_diff < 0) &
            (vix > vix_ma3) &
            (vix_diff > 0)
        )

        # 只在严苛条件满足的当天输出脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        # 清除由于前置数据不足产生的无效计算期
        signal.iloc[:window] = 0.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"