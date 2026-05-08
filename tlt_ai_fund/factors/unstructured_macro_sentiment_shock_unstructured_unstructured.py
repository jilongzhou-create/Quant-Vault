import numpy as np
import pandas as pd

class UnstructuredMacroSentimentShockFactor:
    """非结构化宏观情绪突变脉冲因子 (unstructured/unstructured)

    逻辑: 采用海量新闻衍生指标(EPU)与美联储文本情绪(FOMC)捕捉宏观预期的极端反转。当不确定性因冲击飙升至极值(Z>2.5)并首现动能衰竭时，市场恐慌见顶，往往Price-in绝大部分利空，长债迎避险修复；当联储语调发生极端边际跃迁时，产生直接重定价脉冲。由于只抓爆发与衰竭点，避免了常态噪音。
    数据: usepuindxd (美国经济政策不确定性), fomc_sentiment (FOMC议息文本情绪得分)
    触发: 
      - 看多脉冲: (EPU 5日变化Z-Score > 2.5 且跌破3日均值开始衰竭) 或 (FOMC 5日鸽派突变Z-Score > 2.5 且确认为鸽派突变瞬间)。
      - 看空脉冲: (EPU 5日变化Z-Score < -2.5 且突破3日均值反向抬头) 或 (FOMC 5日鹰派突变Z-Score < -2.5 且确认为鹰派突变瞬间)。
      为满足 Target Trigger Rate，触发后释放 3 天 (当天+跟随2天) 的极短生命周期。
    输出: +1.0 (恐慌衰竭/极度转鸽) 或 -1.0 (极度自满打破/极度转鹰)，常态休眠 0.0。
    """

    def __init__(self, window=5, z_window=252, z_thresh=2.5, hold_days=2):
        self.name = 'unstructured_macro_sentiment_shock'
        self.window = window
        self.z_window = z_window
        self.z_thresh = z_thresh
        self.hold_days = hold_days

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 铁律1: 零值休眠，初始化基底为 0.0
        signal = pd.Series(0.0, index=data.index)
        bull_mask = pd.Series(False, index=data.index)
        bear_mask = pd.Series(False, index=data.index)

        # ====== 模块1: EPU 经济政策不确定性突变 (非结构化新闻数据) ======
        if 'usepuindxd' in data.columns:
            epu = data['usepuindxd'].ffill()
            
            # 铁律3: 边际变化 (严格禁止绝对值水位判断)
            epu_chg = epu.diff(self.window)
            
            # 计算边际变化的 Z-Score 确定异常波动
            epu_roll_mean = epu_chg.rolling(self.z_window).mean()
            epu_roll_std = epu_chg.rolling(self.z_window).std()
            epu_z = (epu_chg - epu_roll_mean) / (epu_roll_std + 1e-6)
            
            # 铁律2: 二阶导数 (要求动能衰竭与反转，禁接飞刀)
            epu_chg_ma3 = epu_chg.rolling(3).mean()
            epu_exhaustion_down = epu_chg < epu_chg_ma3  # 向上飙升见顶回落
            epu_exhaustion_up = epu_chg > epu_chg_ma3    # 向下急跌企稳抬头
            
            # 看多条件: 极度恐慌 + 开始回落
            epu_bull = (epu_z > self.z_thresh) & epu_exhaustion_down
            # 看空条件: 极度自满 + 突发不确定性
            epu_bear = (epu_z < -self.z_thresh) & epu_exhaustion_up
            
            bull_mask = bull_mask | epu_bull
            bear_mask = bear_mask | epu_bear

        # ====== 模块2: FOMC 情绪得分跳跃 (非结构化央行文本) ======
        if 'fomc_sentiment' in data.columns:
            fomc = data['fomc_sentiment'].ffill()
            
            # 铁律3: 边际变化 (捕捉超预期动作)
            fomc_chg = fomc.diff(self.window)
            
            # 计算变化的 Z-Score
            fomc_roll_mean = fomc_chg.rolling(self.z_window).mean()
            fomc_roll_std = fomc_chg.rolling(self.z_window).std()
            fomc_z = (fomc_chg - fomc_roll_mean) / (fomc_roll_std + 1e-6)
            
            # 仅在边际跳跃真正发生时触发，抛弃之后高位的粘滞噪音
            # 阶梯数据的突变日，必定伴随单日 diff 发生非零位移
            is_jump = fomc.diff(1).abs() > 0
            
            # 看多条件: 极端转鸽极值 + 确认当日突变 + 水位进入实质性宽松期
            fomc_bull = (fomc_z > self.z_thresh) & (fomc > 0) & is_jump
            # 看空条件: 极端转鹰极值 + 确认当日突变 + 水位进入实质性紧缩期
            fomc_bear = (fomc_z < -self.z_thresh) & (fomc < 0) & is_jump

            bull_mask = bull_mask | fomc_bull
            bear_mask = bear_mask | fomc_bear

        # ====== 信号合成与目标 Trigger Rate 控制 ======
        raw_bull = pd.Series(0.0, index=data.index)
        raw_bull[bull_mask] = 1.0
        
        raw_bear = pd.Series(0.0, index=data.index)
        raw_bear[bear_mask] = -1.0

        # 向前延展极短时间以捕捉动量释放的红利周期，且把 Trigger rate 拉高到 5% 级别
        bull_pulse = raw_bull.replace(0.0, np.nan).ffill(limit=self.hold_days).fillna(0.0)
        bear_pulse = raw_bear.replace(0.0, np.nan).ffill(limit=self.hold_days).fillna(0.0)

        # 极端情况下若多空重叠，偏向做空(恐慌或紧缩发力时杀伤力往往更大)
        signal[bull_pulse == 1.0] = 1.0
        signal[bear_pulse == -1.0] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, z_window={self.z_window}, z_thresh={self.z_thresh})"