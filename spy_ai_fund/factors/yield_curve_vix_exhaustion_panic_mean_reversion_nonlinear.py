import numpy as np
import pandas as pd

class YieldCurveVixExhaustionFactor:
    """收益率曲线牛陡与VIX恐慌衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 针对美股市场长牛且均值回归的物理属性，最危险的恐慌通常来自经济衰退预期引发的"美债曲线急剧牛陡"(短端利率暴跌，倒挂迅速解除)，伴随VIX飙升。此时绝对不能接飞刀。正确的抄底时刻必须等待VIX创新高后单日明显回落(恐慌衰竭)。恐慌初期发酵阶段则输出一次性的看空脉冲。
    数据: vixcls (VIX隐含波动率), t10y2y (10年-2年美债利差)
    输出: +1.0 表示恐慌极值见顶回落(极佳抄底买点)，-1.0 表示恐慌刚开始发酵且趋势恶化，0.0 为常态无动作。
    触发条件: 抄底需曲线急陡动量(Z>1.5)与VIX极值(Z>1.5)叠加后VIX回落；做空需恐慌动量刚突破且VIX抬升(脉冲化)。预期Trigger Rate约为6%-10%。
    """

    def __init__(self):
        self.name = 'yield_curve_vix_exhaustion_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 常态下必须休眠, 返回全 0
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖数据是否存在
        if 'vixcls' not in data.columns or 't10y2y' not in data.columns:
            return signal
            
        # 缺失值前向填充，防漏
        vix = data['vixcls'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # --- 1. 计算 VIX 的极值状态 ---
        # 采用252交易日作为长期滚动基准评估Z-Score
        vix_mean_252 = vix.rolling(window=252, min_periods=60).mean()
        vix_std_252 = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean_252) / vix_std_252
        
        # --- 2. 计算收益率曲线边际变化的极值状态 ---
        # 必须使用 .diff() 衡量预期改变(边际变化)，绝对禁止使用绝对值交叉
        # 5日变动代表短期的陡峭化急剧程度
        t10_2_diff5 = t10y2y.diff(5)
        t10_2_mom_mean = t10_2_diff5.rolling(window=252, min_periods=60).mean()
        t10_2_mom_std = t10_2_diff5.rolling(window=252, min_periods=60).std()
        t10_2_mom_z = (t10_2_diff5 - t10_2_mom_mean) / t10_2_mom_std
        
        # --- 3. 极端状态定义 ---
        # 利差急剧走阔 (Z > 1.5) 且 VIX 处于历史高位 (Z > 1.5)
        panic_state = (t10_2_mom_z > 1.5) & (vix_z > 1.5)
        
        # --- 4. 恐慌衰竭 (抄底买点: +1.0) ---
        # 二阶导数铁律: 高位接飞刀必死，必须等待恐慌边际减弱
        # 当天VIX下跌且VIX跌破过去3日均值
        vix_exhaustion = (vix.diff(1) < 0) & (vix < vix.rolling(window=3).mean())
        
        # 触发条件: 过去5天内只要发生过极度恐慌，且今天刚刚出现恐慌衰竭
        buy_trigger_base = panic_state.rolling(window=5, min_periods=1).max().shift(1).fillna(0) > 0
        buy_trigger = buy_trigger_base & vix_exhaustion
        
        # --- 5. 恐慌发酵阶段 (趋势恶化看空: -1.0) ---
        # 曲线开始走阔动量初现(Z>1.0)，波动率抬升但未达到爆表区间(0.5 < Z <= 1.5)
        panic_brewing = (t10_2_mom_z > 1.0) & (vix.diff(3) > 0) & (vix_z > 0.5) & (vix_z <= 1.5)
        
        # 脉冲化处理: 仅在状态刚满足的当天输出，绝对禁止连续输出
        sell_trigger = panic_brewing & (~panic_brewing.shift(1).fillna(False))
        
        # 处理可能的极小概率冲突，买点优先(衰竭确认)
        sell_trigger = sell_trigger & (~buy_trigger)
        
        # --- 6. 赋值离散信号 ---
        signal.loc[sell_trigger] = -1.0
        signal.loc[buy_trigger] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"