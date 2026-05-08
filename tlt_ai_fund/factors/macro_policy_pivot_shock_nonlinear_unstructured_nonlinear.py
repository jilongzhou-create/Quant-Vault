import numpy as np
import pandas as pd

class MacroPolicyPivotShockNonlinearFactor:
    """政策预期突变非线性交叉因子 (unstructured/nonlinear)

    逻辑: 捕捉宏观恐慌/加息预期见顶衰竭时，美联储政策预期突变导致的收益率曲线剧烈形变。
          对于长端美债(TLT)，这是狙击手级别的脉冲交易机会。避险情绪爆发引发短端利率下行和牛陡时做多；加息预期极端飙升引发熊平时做空。
    数据: vixcls (波动率/恐慌), dgs2 (短端利率/政策敏感), t10y2y (收益率曲线形态)
    触发: 
      做多: VIX Z-Score > 2.5 且开始跌破3日均值(恐慌衰竭) + dgs2降息预期(边际下行) + t10y2y牛陡(边际变陡)
      做空: dgs2上行动量 Z-Score > 2.5 且单日上行势头弱于3日均值(加息恐慌衰竭) + t10y2y熊平(边际变平) + VIX未进入极度恐慌
    输出: +1.0 看多美债(TLT), -1.0 看空美债, 其余时间常态为 0.0 脉冲休眠
    """

    def __init__(self):
        self.name = 'macro_policy_pivot_shock_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化零值休眠信号
        signal = pd.Series(0.0, index=data.index)
        
        # 检查依赖特征
        required_cols = ['vixcls', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 提取时间序列并前向填充缺失值
        vix = data['vixcls'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # ---------------------------------------------------------
        # 1. 波动率与恐慌情绪 (二阶导数与衰竭逻辑)
        # ---------------------------------------------------------
        # 计算 VIX 的 63个交易日(约一季度)的 Z-Score
        vix_mean = vix.rolling(window=63, min_periods=21).mean()
        vix_std = vix.rolling(window=63, min_periods=21).std()
        vix_zscore = (vix - vix_mean) / vix_std.replace(0, 1e-5)
        
        # VIX 二阶衰竭条件: 极值后跌破3日均值，证明脉冲势头放缓 (反接飞刀铁律)
        vix_exhaustion = vix < vix.rolling(window=3).mean()
        
        # ---------------------------------------------------------
        # 2. 短端利率边际变化 (前瞻政策预期)
        # ---------------------------------------------------------
        # 边际变化铁律: 使用 5日动量而非绝对水位
        dgs2_diff5 = dgs2.diff(5)
        dgs2_1d = dgs2.diff(1)
        
        # dgs2 动量的极值 Z-Score (加息/降息预期的急速跳跃)
        dgs2_diff_mean = dgs2_diff5.rolling(window=63, min_periods=21).mean()
        dgs2_diff_std = dgs2_diff5.rolling(window=63, min_periods=21).std()
        dgs2_diff_zscore = (dgs2_diff5 - dgs2_diff_mean) / dgs2_diff_std.replace(0, 1e-5)
        
        # dgs2 二阶衰竭条件: 单日上行动量小于过去3日平均，表明鹰派预期冲击初步见顶
        dgs2_bear_exhaustion = dgs2_1d < dgs2_1d.rolling(window=3).mean()
        
        # ---------------------------------------------------------
        # 3. 期限结构非线性交叉 (动量变动而非绝对倒挂)
        # ---------------------------------------------------------
        # 边际变化铁律: 只看曲线的动量变动
        t10y2y_diff5 = t10y2y.diff(5)
        
        # ---------------------------------------------------------
        # 4. 生成狙击手脉冲信号 (多维共振)
        # ---------------------------------------------------------
        
        # 【做多脉冲】 避险极端且衰竭 + 短端利率边际下行(降息预期) + 曲线牛陡
        long_cond = (
            (vix_zscore > 2.5) & 
            vix_exhaustion & 
            (dgs2_diff5 < 0.0) & 
            (t10y2y_diff5 > 0.0)
        )
        
        # 【做空脉冲】 鹰派加息突变且势头放缓 + 曲线熊平 + 未爆发全面流动性恐慌
        short_cond = (
            (dgs2_diff_zscore > 2.5) & 
            dgs2_bear_exhaustion & 
            (t10y2y_diff5 < 0.0) & 
            (vix_zscore < 1.5)
        )
        
        # 赋值并只在条件触发时偏离 0.0
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"