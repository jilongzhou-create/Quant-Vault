import numpy as np
import pandas as pd

class EpuPivotSteepeningFactor:
    """Unstructured EPU Pivot Steepening Factor (unstructured/nonlinear)

    逻辑: 捕捉由于极端经济政策不确定性(EPU)消退所引发的美联储货币政策预期突变。
          当政策不确定性飙升后开始回落时(恐慌衰竭)，如果伴随短端利率(2年期)剧烈下行且收益率曲线急剧变陡(Bull Steepening)，
          说明市场正在进行极端的鸽派降息定价(Price-in)，此时做多美债(TLT)。反之做空。
          这是一个严格的脉冲因子，仅在预期重定价的极短期窗口内触发。
    数据: dgs2 (2年期美债收益率), t10y2y (10年-2年利差), usepuindxd (美国经济政策不确定性指数)
    触发: 
          看多(降息突变): dgs2 5日降幅 Z-Score < -1.5 AND t10y2y 5日涨幅 Z-Score > 1.5 AND 
                        EPU处于高位(Z > 1.0) AND EPU开始衰竭回落(<3日均值)
          看空(加息突变): dgs2 5日涨幅 Z-Score > 1.5 AND t10y2y 5日降幅 Z-Score < -1.5 AND 
                        EPU处于高位(Z > 1.0) AND EPU开始衰竭回落(<3日均值)
    输出: +1.0 (鸽派降息突变，看多TLT) / -1.0 (鹰派加息突变，看空TLT) / 0.0 (常态休眠)
    """

    def __init__(self):
        self.name = 'unstructured_epu_pivot_steepening_nonlinear'
        self.lookback_window = 252 # 1年基准期用于计算经济学意义上的Z-Score
        self.momentum_window = 5   # 5天(一周)用于捕捉边际突变
        
    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全 0 信号，遵守脉冲休眠铁律
        signal = pd.Series(0.0, index=data.index)
        
        # 检查必需字段是否存在
        required_cols = ['dgs2', 't10y2y', 'usepuindxd']
        if not all(col in data.columns for col in required_cols):
            return signal
            
        # 提取数据并做前向填充以处理节假日缺失值，避免差分计算中断 (禁止 look-ahead)
        df = data[required_cols].ffill()
        
        # 铁律3: 边际变化 (Marginal Change)
        # 绝对禁止使用利率的绝对水位，必须使用 5 日动量变化捕捉突变
        dgs2_diff = df['dgs2'].diff(self.momentum_window)
        t10y2y_diff = df['t10y2y'].diff(self.momentum_window)
        
        # 计算各指标的滚动 Z-Score 以衡量极端程度
        dgs2_diff_mean = dgs2_diff.rolling(self.lookback_window, min_periods=20).mean()
        dgs2_diff_std = dgs2_diff.rolling(self.lookback_window, min_periods=20).std()
        dgs2_z = (dgs2_diff - dgs2_diff_mean) / (dgs2_diff_std + 1e-8)
        
        t10y2y_diff_mean = t10y2y_diff.rolling(self.lookback_window, min_periods=20).mean()
        t10y2y_diff_std = t10y2y_diff.rolling(self.lookback_window, min_periods=20).std()
        t10y2y_z = (t10y2y_diff - t10y2y_diff_mean) / (t10y2y_diff_std + 1e-8)
        
        epu_mean = df['usepuindxd'].rolling(self.lookback_window, min_periods=20).mean()
        epu_std = df['usepuindxd'].rolling(self.lookback_window, min_periods=20).std()
        epu_z = (df['usepuindxd'] - epu_mean) / (epu_std + 1e-8)
        
        # 铁律2: 二阶导数/衰竭条件 (Anti-Catch-Falling-Knife)
        # 政策不确定性必须处于高位且已经开始回落，禁止在不确定性发散上升期接飞刀
        epu_exhaustion = df['usepuindxd'] < df['usepuindxd'].rolling(3).mean()
        cond_epu_ready = (epu_z > 1.0) & epu_exhaustion
        
        # --- 多头信号条件 (鸽派突变 / Bull Steepening) ---
        # 1. 2年期短端极速暴跌 (降息预期骤升)
        # 2. 10年-2年利差极速扩大 (短端下行快于长端，经典的宽松早期形态)
        cond_dovish_plunge = dgs2_z < -1.5
        cond_bull_steepen = t10y2y_z > 1.5
        buy_mask = cond_dovish_plunge & cond_bull_steepen & cond_epu_ready
        
        # --- 空头信号条件 (鹰派突变 / Bear Flattening) ---
        # 1. 2年期短端极速飙升 (加息预期骤升)
        # 2. 10年-2年利差极速缩小/倒挂加深 (短端上行快于长端，经典的紧缩形态)
        cond_hawkish_spike = dgs2_z > 1.5
        cond_bear_flatten = t10y2y_z < -1.5
        sell_mask = cond_hawkish_spike & cond_bear_flatten & cond_epu_ready
        
        # 铁律1: 零值休眠 (Sniper Pulse)
        # 仅在非线性交叉条件完全满足的瞬间触发 ±1.0 信号
        signal.loc[buy_mask] = 1.0
        signal.loc[sell_mask] = -1.0
        
        # 清理由于计算前几天可能产生的 NaN
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(lookback={self.lookback_window}, momentum={self.momentum_window})"