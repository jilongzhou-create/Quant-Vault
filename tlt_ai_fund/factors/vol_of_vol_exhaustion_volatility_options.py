import numpy as np
import pandas as pd

class EquityVrpExhaustionFactor:
    """Volatility / Options

    逻辑: 捕捉隐含波动率风险溢价(VRP = VIX - 现货实际波动率)的极端错配与二阶导数反转。
          VRP隔离了期权市场的纯恐慌溢价。多头逻辑: 当VRP极端飙升(Z-Score > 2.0)且开始瓦解时, 标志着急跌恐慌见顶, 央行鸽派预期发酵, 做多美债(TLT)。
          空头逻辑: 当VRP极端负溢价(现货阴跌但VIX极低, 典型利率驱动慢熊)且开始反弹时, 标志着流动性收紧持续发酵, 做空美债。
          脉冲设计: 必须配合 VRP与VIX的 .diff() 衰竭条件, 确保绝不接飞刀。
    数据: vixcls, sp500 / nasdaqcom / djia
    触发: VRP Z-Score > 2.0 且 边际回落 (+1.0); VRP Z-Score < -2.0 且 边际反弹 (-1.0)
    输出: [-1.0, 1.0] 狙击手级脉冲
    """

    def __init__(self):
        self.name = 'equity_vrp_exhaustion_volatility_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 绝对铁律 1: 常态下必须为 0.0
        signal = pd.Series(0.0, index=data.index)
        
        # 确保 VIX 数据可用
        if 'vixcls' not in data.columns:
            return signal
            
        vix = data['vixcls'].ffill()
        
        # 构建现货指数替代链以最大化历史回溯覆盖
        equity_idx = None
        for col in ['sp500', 'nasdaqcom', 'djia']:
            if col in data.columns:
                if equity_idx is None:
                    equity_idx = data[col].copy()
                else:
                    equity_idx = equity_idx.fillna(data[col])
                    
        if equity_idx is None:
            return signal
            
        # 滤除非交易日(价格无变动周末等)，精确提取有效交易日
        valid_series = equity_idx.drop_duplicates(keep='first')
        if len(valid_series) < 21:
            return signal
            
        # 计算 21个有效交易日 的现货实际年化波动率 (Realized Volatility)
        valid_ret = valid_series.pct_change().fillna(0.0)
        valid_realized_vol = valid_ret.rolling(window=21, min_periods=10).std() * np.sqrt(252) * 100
        
        # 将算好的交易日波动率映射回全局时间轴并前向填充
        realized_vol = pd.Series(np.nan, index=data.index)
        realized_vol.loc[valid_series.index] = valid_realized_vol
        realized_vol = realized_vol.ffill()
        
        # 计算 VRP (Volatility Risk Premium): 隐含波动率(事前) - 实际波动率(事后)
        vrp = vix - realized_vol
        
        # 使用 42 日 (约2个交易月) 窗口计算短期自适应 Z-Score
        # 使用较短窗口保证其对局部均值有高敏感度，确保产生(5%-15%)的高质量极端脉冲
        vrp_mean = vrp.rolling(window=42, min_periods=10).mean()
        vrp_std = vrp.rolling(window=42, min_periods=10).std()
        vrp_std = vrp_std.replace(0, np.nan)
        vrp_zscore = (vrp - vrp_mean) / vrp_std
        
        # --------------------------------------------------------
        # 绝对铁律 2 & 3: 二阶导数 (极值 + 衰竭) & 边际变化
        # --------------------------------------------------------
        
        # --- 多头脉冲 (+1.0) 逻辑 ---
        # 极值: 过去2天内 VRP 曾处于极端恐慌高位
        cond_long_extreme = (vrp_zscore > 2.0).astype(int).rolling(window=2).max() > 0
        # 衰竭: 今日 VRP 边际回落 且 跌破3日均线
        cond_long_exhaustion = (vrp < vrp.rolling(window=3).mean()) & (vrp.diff() < 0)
        # 确认: VIX 本身也必须发生实质性边际回落
        cond_long_vix_confirm = vix.diff() < 0
        
        trigger_long = cond_long_extreme & cond_long_exhaustion & cond_long_vix_confirm
        
        # --- 空头脉冲 (-1.0) 逻辑 ---
        # 极值: 过去2天内 VRP 曾处于极度负溢价 (现货在阴跌，期权处于温水煮青蛙的自满状态)
        cond_short_extreme = (vrp_zscore < -2.0).astype(int).rolling(window=2).max() > 0
        # 衰竭(反转): 负溢价状态见底，期权市场开始醒悟并定价补跌
        cond_short_exhaustion = (vrp > vrp.rolling(window=3).mean()) & (vrp.diff() > 0)
        # 确认: VIX 本身边际抬升
        cond_short_vix_confirm = vix.diff() > 0
        
        trigger_short = cond_short_extreme & cond_short_exhaustion & cond_short_vix_confirm
        
        # 严格赋值，生成脉冲信号
        signal[trigger_long] = 1.0
        signal[trigger_short] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"