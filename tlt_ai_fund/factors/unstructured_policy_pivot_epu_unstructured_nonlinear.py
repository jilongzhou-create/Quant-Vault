import numpy as np
import pandas as pd

class UnstructuredPolicyShockNonlinearFactor:
    """非结构化政策恐慌衰竭因子 (unstructured/nonlinear)

    逻辑: 捕捉基于新闻文本挖掘的经济政策不确定性(EPU)极值反转，并交叉验证短端利率(DGS2)的政策预期突变。
          当政策不确定性飙升至极值并见顶回落，且伴随2年期美债收益率快速下行、收益率曲线变陡时，
          代表"不确定性冲击倒逼美联储降息"的逻辑正被市场积极Price-in (Bull Steepening)，此时生成看多美债脉冲。
          本因子严格遵循零值休眠与二阶导数反转铁律，避免在波动率恶化主跌浪中接飞刀。
    数据: usepuindxd (经济政策不确定性), dgs2 (2年期美债), t10y2y (期限利差)
    触发: EPU Z-Score > 1.0 + 近3日EPU动量衰竭回落 + DGS2下行且曲线陡峭化 -> 脉冲+1.0
    输出: 狙击手级别的脉冲信号 [-1.0, 1.0]，非触发日严格为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_policy_shock_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠 (Sniper Pulse)
        signal = pd.Series(0.0, index=data.index)
        
        # 缺失列检查，保护计算过程
        req_cols = ['usepuindxd', 'dgs2', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return signal
                
        # 前向填充处理非交易日或缺失数据
        epu = data['usepuindxd'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 1. 非结构化数据平滑与高频Z-Score计算 (窗口=252天)
        # EPU本身带有较大日度噪音，使用5日均值提取核心政策情绪趋势
        epu_smooth = epu.rolling(window=5).mean()
        epu_mean = epu_smooth.rolling(window=252).mean()
        epu_std = epu_smooth.rolling(window=252).std()
        
        # 计算政策不确定性水位的相对极端程度
        epu_z = (epu_smooth - epu_mean) / (epu_std + 1e-8)
        
        # 2. 铁律2: 二阶导数与衰竭 (Anti-Catch-Falling-Knife)
        # 绝对禁止在高不确定性期间单边做多！必须等待情绪恶化停止并开始反转
        # 看多衰竭：政策恐慌处于高位(Z>1.0)，且最近3天已经开始降温(diff<0)
        epu_peak_exhaustion = (epu_z > 1.0) & (epu_smooth.diff(3) < 0)
        
        # 看空衰竭：政策过度自满(Z<-1.0)，且最近3天不确定性开始抬头(diff>0)
        epu_trough_exhaustion = (epu_z < -1.0) & (epu_smooth.diff(3) > 0)
        
        # 3. 铁律3: 边际变化与 FICC经济学交叉验证
        # 仅有情绪衰竭不够，必须有实际资金流(短端利率)的价格验证
        
        # 降息预期印证: 短端收益率3天内快速下行超过5个基点 (-0.05)
        dgs2_falling = dgs2.diff(3) < -0.05
        
        # 曲线陡峭化印证 (Bull Steepening): 短端下行快于长端，期限利差走阔超过2个基点
        # 这是美联储开启降息周期时最典型的收益率曲线形变
        curve_steepening = t10y2y.diff(3) > 0.02
        
        # 加息/过热预期印证: 短端收益率快速上行超过5个基点
        dgs2_rising = dgs2.diff(3) > 0.05
        
        # 曲线平坦化印证 (Bear Flattening): 短端上行快于长端
        curve_flattening = t10y2y.diff(3) < -0.02
        
        # 4. 组合触发逻辑 (目标 Trigger Rate 5% - 15%)
        # 多头脉冲: 恐慌见顶衰竭 + 降息预期升温 + 牛陡
        bull_cond = epu_peak_exhaustion & dgs2_falling & curve_steepening
        
        # 空头脉冲: 自满结束抬头 + 紧缩预期升温 + 熊平
        bear_cond = epu_trough_exhaustion & dgs2_rising & curve_flattening
        
        signal[bull_cond] = 1.0
        signal[bear_cond] = -1.0
        
        # 补充动量突破逻辑: 当短端利率发生极其剧烈的单边重定价(>10个基点)时，
        # 即便曲线形态因长端跟随而未显著形变，也构成强烈的政策转向共振
        bull_cond_alt = epu_peak_exhaustion & (dgs2.diff(3) < -0.10)
        bear_cond_alt = epu_trough_exhaustion & (dgs2.diff(3) > 0.10)
        
        signal[bull_cond_alt] = 1.0
        signal[bear_cond_alt] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"