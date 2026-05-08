import numpy as np
import pandas as pd

class UnstructuredMacroOptionsShockFactor:
    """宏观政策恐慌突变因子 (unstructured/options)

    逻辑: 结合基于新闻文本的经济政策不确定性(usepuindxd)与期权隐含波动率(vixcls)，构建综合宏观恐慌突变指数。基于期权市场与非结构化文本的共振，当恐慌突变极度飙升并开始回落时(二阶衰竭)，市场将Price-in美联储流动性救市及降息预期，由于美债是正向Carry的避险资产，此时做多美债；反之当极度自满(突变指标暴跌)且动能衰竭时，市场容易遭遇通胀或紧缩冲击，此时做空美债。
    数据: usepuindxd (非结构化文本指标), vixcls (期权隐含波动率)
    触发: 动量突变的 63日 Z-Score > 1.25 且动量开始回落(弱于3日均值) -> +1.0；Z-Score < -1.25 且动量开始反转上升 -> -1.0
    输出: 严格脉冲型信号，[-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'unstructured_macro_options_shock'
        self.window = 63             # 63个交易日(一个季度)，用于计算中期极值
        self.momentum_days = 5       # 5日变化量，捕捉边际突变
        self.smooth_days = 3         # 3日均值，用于二阶导数的衰竭判定
        self.z_thresh = 1.25         # 1.25个标准差，大致对应 10%-15% 的触发概率，满足5%-15%铁律

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始默认信号为0.0 (零值休眠铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        if 'usepuindxd' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        # 前向填充缺失值，避免对齐问题
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        
        # 1. 预平滑处理，过滤单日极端噪音
        epu_smooth = epu.rolling(window=self.smooth_days).mean()
        vix_smooth = vix.rolling(window=self.smooth_days).mean()
        
        # 2. 边际变化铁律：绝对禁止看绝对值，使用 5 日动量捕捉情绪的阶跃(Shock)
        epu_diff = epu_smooth.diff(self.momentum_days)
        vix_diff = vix_smooth.diff(self.momentum_days)
        
        # 3. 标准化：计算两者突变动能的 63 日 Z-Score，统一量纲
        epu_z = (epu_diff - epu_diff.rolling(self.window).mean()) / epu_diff.rolling(self.window).std().replace(0, 1e-5)
        vix_z = (vix_diff - vix_diff.rolling(self.window).mean()) / vix_diff.rolling(self.window).std().replace(0, 1e-5)
        
        # 4. 构建综合宏观恐慌突变指数 (Macro Shock Index)
        macro_shock = epu_z + vix_z
        
        # 对综合突变指数再次标准化以确定最终触发的极端阈值
        shock_z = (macro_shock - macro_shock.rolling(self.window).mean()) / macro_shock.rolling(self.window).std().replace(0, 1e-5)
        
        # 5. 二阶导数铁律 (Anti-Catch-Falling-Knife)
        # 突变动能开始放缓 (当前突变值低于过去3天均值，确认山峰翻越)
        shock_falling = macro_shock < macro_shock.rolling(self.smooth_days).mean()
        # 突变动能开始抬头反弹 (当前突变值高于过去3天均值，确认山谷翻越)
        shock_rising = macro_shock > macro_shock.rolling(self.smooth_days).mean()
        
        # 6. 信号生成
        # Long TLT: 情绪极度飙升(恐慌)，但飙升动能开始衰竭 -> 资产即将出清，预期联储转鸽救市 -> +1.0
        long_cond = (shock_z > self.z_thresh) & shock_falling
        
        # Short TLT: 情绪极度下挫(自满)，但下挫动能衰竭 -> 市场无视过热风险，极易遭鹰派通胀打击 -> -1.0
        short_cond = (shock_z < -self.z_thresh) & shock_rising
        
        signal.loc[long_cond.fillna(False)] = 1.0
        signal.loc[short_cond.fillna(False)] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, thresh={self.z_thresh})"