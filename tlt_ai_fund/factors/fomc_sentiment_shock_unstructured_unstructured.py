import numpy as np
import pandas as pd

class UnstructuredPolicyPivotShockFactor:
    """政策预期突变脉冲因子 (unstructured)

    逻辑: 捕捉美联储官方(FOMC情绪)或市场隐含(短端利率)的政策预期极端突变。这会导致债券市场重新定价。
          为避免接飞刀，必须严格等待突变冲击过后，短端收益率动量放缓(衰竭)且恐慌情绪(VIX)回落时才入场，确保胜率和IC。
    数据: fomc_sentiment, dgs2, t10y2y, vixcls
    触发: (FOMC情绪5日边际变化Z-Score>2.0 OR 2年期收益率急跌/暴涨Z-Score极值) + 收益率运动停滞(衰竭) + VIX回落
    输出: 严格脉冲型 [-1.0, 1.0]
    """

    def __init__(self):
        self.name = 'unstructured_policy_pivot_shock'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 严格遵守铁律：初始信号必须为 0.0 (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需的核心字段是否存在
        required_cols = ['fomc_sentiment', 'dgs2', 't10y2y', 'vixcls']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 前向填充以处理非交易日或缺失值
        df = data[required_cols].ffill()
        
        # --- 条件A: FOMC 情绪突变 (官方预期突变) ---
        # 遵守边际变化铁律: 绝对禁止使用绝对值，使用 5 日变化量捕捉会议前后的情绪跳跃
        sent_mom = df['fomc_sentiment'].diff(5)
        # 使用 252 日滚动标准差，设置 0.05 的下限防止平缓期(无会议期间)除以零导致异动
        sent_vol = sent_mom.rolling(252).std().clip(lower=0.05)
        sent_z = (sent_mom - sent_mom.rolling(252).mean()) / sent_vol
        
        # --- 条件B: 市场隐含突变 (短端收益率与曲线形态的极值) ---
        # 2年期国债收益率(dgs2)对政策最敏感，10年-2年利差(t10y2y)变陡/变平确认降息/加息周期
        dgs2_mom = df['dgs2'].diff(5)
        curve_mom = df['t10y2y'].diff(5)
        
        dgs2_vol = dgs2_mom.rolling(252).std() + 1e-5
        curve_vol = curve_mom.rolling(252).std() + 1e-5
        
        dgs2_z = (dgs2_mom - dgs2_mom.rolling(252).mean()) / dgs2_vol
        curve_z = (curve_mom - curve_mom.rolling(252).mean()) / curve_vol
        
        # --- 条件C: 衰竭与确认条件 (反接飞刀/二阶导数铁律) ---
        # 短端收益率的单日变化，用于判断极值后的动量是否已经放缓
        dgs2_daily = df['dgs2'].diff(1)
        
        # VIX 情绪衰竭：股市恐慌情绪必须已经见顶回落 (当前 VIX 低于过去 3 日均值)
        vix_exhaustion = df['vixcls'] < df['vixcls'].rolling(3).mean()
        
        # --- 信号生成逻辑聚合 ---
        
        # 做多 TLT (看多美债) 条件：
        # 极端鸽派突变：FOMC 情绪飙升(转鸽)，或市场自发走出短端收益率急跌+曲线牛陡
        cond_dovish_extreme = (sent_z > 2.0) | ((dgs2_z < -2.0) & (curve_z > 1.5))
        # 衰竭条件：收益率停止急速单边下跌 (今日下跌幅度缓于 2 个基点，或已经开始微弹)
        cond_dovish_exhaust = (dgs2_daily >= -0.02)
        
        buy_cond = cond_dovish_extreme & cond_dovish_exhaust & vix_exhaustion
        
        # 做空 TLT (看空美债) 条件：
        # 极端鹰派突变：FOMC 情绪骤降(转鹰)，或市场自发走出短端收益率暴涨+曲线熊平
        cond_hawkish_extreme = (sent_z < -2.0) | ((dgs2_z > 2.0) & (curve_z < -1.5))
        # 衰竭条件：收益率停止急速单边上涨 (今日上涨幅度缓于 2 个基点，或已经开始微调)
        cond_hawkish_exhaust = (dgs2_daily <= 0.02)
        
        sell_cond = cond_hawkish_extreme & cond_hawkish_exhaust & vix_exhaustion
        
        # --- 赋值触发信号 ---
        signal[buy_cond] = 1.0
        signal[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"