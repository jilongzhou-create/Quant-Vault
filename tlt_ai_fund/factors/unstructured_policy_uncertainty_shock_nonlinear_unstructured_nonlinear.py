import numpy as np
import pandas as pd

class UnstructuredMacroShockNonlinearFactor:
    """Unstructured Macro Shock Nonlinear Factor (unstructured/nonlinear)

    逻辑: 结合非结构化的经济政策不确定性(EPU)与政策敏感的前瞻指标(2年期美债收益率), 通过非线性交叉放大(x * |x|)构建"不确定性-利率冲击指数"。
          利用"极值+衰竭"的二阶导数抄底铁律防飞刀：
          做多TLT：当"鹰派恐慌(不确定性高企+短端利率暴涨)"达到极值(Z>2.0)，且短端利率动量衰竭(开始回落)时，精准狙击美联储紧缩预期见顶瞬间。
          做空TLT：当"鸽派狂热(不确定性急剧变化+短端利率暴跌)"达到极值(Z<-2.0)，且短端利率触底反弹时，捕捉降息预期Price-in过度的修正瞬间。
    数据: usepuindxd (Economic Policy Uncertainty), dgs2 (2-year Yield), t10y2y (Yield Curve)
    触发: 冲击指数前向Z-Score > 2.0且当日短端利率反转 + 收益率曲线未出现背离
    输出: 脉冲信号 [-1.0, 1.0], 目标 Trigger Rate 5%-15%
    """

    def __init__(self):
        self.name = 'unstructured_macro_shock_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 严格遵守初始化为0.0，满足零值休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        required_cols = ['usepuindxd', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 向前填充缺失值
        epu = data['usepuindxd'].ffill()
        dgs2 = data['dgs2'].ffill()
        curve = data['t10y2y'].ffill()
        
        # 1. 边际变化计算 (三大铁律之三: Marginal Change Only)
        # 使用3日窗口捕捉短期的边际动量脉冲
        epu_diff = epu.diff(3)
        dgs2_diff = dgs2.diff(3)
        curve_diff = curve.diff(3)
        
        # 2. 前向 Z-Score 计算 (避免极端值拉大当期标准差导致自我抑制)
        # 63天为一个宏观经济季度
        epu_mean = epu_diff.rolling(63).mean().shift(1)
        epu_std = epu_diff.rolling(63).std().shift(1) + 1e-6
        epu_z = (epu_diff - epu_mean) / epu_std
        
        dgs2_mean = dgs2_diff.rolling(63).mean().shift(1)
        dgs2_std = dgs2_diff.rolling(63).std().shift(1) + 1e-6
        dgs2_z = (dgs2_diff - dgs2_mean) / dgs2_std
        
        # 3. 非线性特征交叉 (当前挖掘方法: nonlinear)
        # 经济学含义: 只有在政策不确定性(EPU)极高时，利率的冲击才具有真实的破坏力。
        # 使用 EPU 的绝对波动(无视方向)去非线性放大 2年期美债收益率的方向性冲击
        shock_index = epu_z.abs() * dgs2_z * dgs2_z.abs()
        
        # 对合成的非线性冲击指数进行二次 Z-Score 映射，以界定标准触发阈值
        shock_mean = shock_index.rolling(63).mean().shift(1)
        shock_std = shock_index.rolling(63).std().shift(1) + 1e-6
        shock_z = (shock_index - shock_mean) / shock_std
        
        # 4. 极值条件提取 (近3日内是否爆发过极端冲击)
        # 阈值 2.0 代表 95% 置信区间尾部事件
        bull_extreme = shock_z.rolling(3).max() > 2.0  # 鹰派恐慌极值 (Yields 狂飙)
        bear_extreme = shock_z.rolling(3).min() < -2.0 # 鸽派狂热极值 (Yields 暴跌)
        
        # 5. 衰竭条件 (三大铁律之二: Anti-Catch-Falling-Knife)
        # 必须等待最核心的资产(2年期美债)出现反向拐点，确认趋势动能衰竭
        bull_exhaust = dgs2.diff(1) < 0  # 鹰派衰竭: 收益率冲顶后开始回落
        bear_exhaust = dgs2.diff(1) > 0  # 鸽派衰竭: 收益率砸底后开始反弹
        
        # 6. 曲线形态验证
        # 做多TLT时，短端下行带来的曲线形态应不至于发生极端的熊市平坦化 (>-0.03 = 过滤异常噪音)
        bull_curve = curve_diff > -0.03
        bear_curve = curve_diff < 0.03
        
        # 7. 脉冲信号综合判定
        bull_trigger = bull_extreme & bull_exhaust & bull_curve
        bear_trigger = bear_extreme & bear_exhaust & bear_curve
        
        # 消除缺失值传播
        bull_trigger = bull_trigger.fillna(False)
        bear_trigger = bear_trigger.fillna(False)
        
        # 输出脉冲信号
        signal[bull_trigger] = 1.0
        signal[bear_trigger] = -1.0
        
        # 冲突兜底处理
        conflict = bull_trigger & bear_trigger
        signal[conflict] = 0.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"