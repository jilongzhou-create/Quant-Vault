import numpy as np
import pandas as pd

class MacroVolRegimeReversalFactor:
    """波动率极值与拥挤反转 (volatility/nonlinear)

    逻辑: 结合黄金(或VIX)与避险货币(日元)的波动率构建跨资产双重恐慌指数。卫星因子必须是纯粹的脉冲信号，遵循"极值+衰竭"铁律。
          当恐慌极度拥挤(Z>1.5)并开始衰竭时，或极度自满(Z<-1.5)并爆发时，我们引入期限利差(t10y2y)作为宏观状态过滤器：
          - 倒挂期(紧缩周期)：恐慌衰竭意味着加息预期见顶，利好美债(+1.0)；自满被打破意味着紧缩超预期，利空美债(-1.0)。
          - 正常期(增长周期)：恐慌衰竭意味着避险情绪消退、经济复苏，利空美债(-1.0)；自满被打破意味着突发增长冲击，利好美债(+1.0)。
          这种基于二阶导数和宏观状态的高阶非线性交叉，彻底解决单边因子在不同宏观周期下的失效和内部摩擦问题。
    数据: gvzcls(黄金波动率, 缺失用vixcls代), dexjpus(日元汇率), t10y2y(期限利差)
    触发: 恐慌指数 Z-Score > 1.5 且开始回落，或 Z-Score < -1.5 且开始飙升。
    输出: 脉冲信号 [-1.0, 1.0]。目标 Trigger Rate 5%-15%。常态下休眠为 0.0。
    """

    def __init__(self):
        self.name = 'macro_vol_regime_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始状态：完全休眠 (满足零值铁律)
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        required_cols = ['dexjpus', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 核心避险资产波动率：黄金 (若早期数据缺失则用VIX替代)
        if 'gvzcls' in data.columns and 'vixcls' in data.columns:
            base_vol = data['gvzcls'].fillna(data['vixcls']).ffill()
        elif 'gvzcls' in data.columns:
            base_vol = data['gvzcls'].ffill()
        elif 'vixcls' in data.columns:
            base_vol = data['vixcls'].ffill()
        else:
            return signal
            
        jpy = data['dexjpus'].ffill()
        curve = data['t10y2y'].ffill()
        
        # 计算日元波动率 (20日年化，捕捉汇率避险情绪)
        # 使用对数收益率
        jpy_ret = np.log(jpy / jpy.shift(1))
        jpy_vol = jpy_ret.rolling(window=20).std() * np.sqrt(252)
        
        # 计算核心指标的 252 日 Z-Score
        base_vol_mean = base_vol.rolling(252).mean()
        base_vol_std = base_vol.rolling(252).std().replace(0, np.nan)
        vol_z = (base_vol - base_vol_mean) / base_vol_std
        
        jpy_vol_mean = jpy_vol.rolling(252).mean()
        jpy_vol_std = jpy_vol.rolling(252).std().replace(0, np.nan)
        jpy_z = (jpy_vol - jpy_vol_mean) / jpy_vol_std
        
        # 合成双重跨资产恐慌指数 (Cross-Asset Panic Index)
        panic_idx = vol_z + jpy_z
        
        # 边际变化 (边际变化铁律：捕捉预期反转瞬间)
        panic_diff = panic_idx.diff()
        vol_diff = base_vol.diff()
        
        # 1. 恐慌极度拥挤且开始衰竭 (反接飞刀铁律：极值 + 衰竭)
        panic_extreme = panic_idx > 1.5
        panic_exhaust = (panic_diff < 0) & (vol_diff < 0) & (panic_idx < panic_idx.rolling(3).mean())
        
        # 2. 极度自满且波动率突然爆发 (反接飞刀铁律反向应用)
        complacency_extreme = panic_idx < -1.5
        complacency_breakout = (panic_diff > 0) & (vol_diff > 0) & (panic_idx > panic_idx.rolling(3).mean())
        
        # 宏观状态过滤器 (Regime Filter: 防止因子在特定周期产生负边际贡献)
        inverted_curve = curve < 0
        normal_curve = curve >= 0
        
        # 信号生成逻辑 (交叉映射)
        # 看多美债 (+1.0)
        # A. 倒挂期(紧缩/通胀主导) + 恐慌衰竭 -> 加息风险见顶，长债反弹
        # B. 正常期(增长主导) + 自满爆发 -> 突发经济/流动性冲击，避险资金涌入美债
        long_cond = (panic_extreme & panic_exhaust & inverted_curve) | \
                    (complacency_extreme & complacency_breakout & normal_curve)
                    
        # 看空美债 (-1.0)
        # A. 倒挂期(紧缩/通胀主导) + 自满爆发 -> 通胀粘性或强于预期，市场醒悟，抛售美债
        # B. 正常期(增长主导) + 恐慌衰竭 -> 危机解除，复苏开启，避险资金流出美债
        short_cond = (complacency_extreme & complacency_breakout & inverted_curve) | \
                     (panic_extreme & panic_exhaust & normal_curve)
                     
        # 最终赋值 (仅在触发条件满足时赋值为1/-1，其余保持为0)
        signal.loc[long_cond.fillna(False)] = 1.0
        signal.loc[short_cond.fillna(False)] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"