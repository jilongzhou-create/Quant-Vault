import numpy as np
import pandas as pd

class UnstructuredFomcPivotPulseFactor:
    """FOMC政策突变反转脉冲卫星因子 (unstructured/unstructured)

    逻辑: 捕捉美联储货币政策声明鹰鸽情绪的极端突跳。美债是极其受宏观预期定价影响的资产，
          当 FOMC 情绪得分的边际变化出现历史级别的罕见跃升 (Z-score > 2.5) 且
          打破了过去一个月的既有趋势惯性时，形成一阶脉冲式的降息/加息预期跳跃。
          这种跳跃非连续发生，因子在常态休眠返回 0.0，只在意外会议次日触发突跳脉冲，
          之后市场会在约 2 周(10交易日)内消化发酵完毕新的政策定价，由此锁定 5%-15% 触发率。
    数据: fomc_sentiment (非结构化 NLP 文本鹰鸽情感得分)
    触发: 5日变化量的 252日 Z-Score > 2.5 (极值突跳条件) 
          + 跳跃日当天确定 (diff(1) != 0, 边际变化瞬间)
          + 衰竭防飞刀 (过去20天宏观趋势动量与此次跳跃反向，表明原有预期衰竭发生反转)
    输出: +1.0 看多美债(超预期转鸽脉冲), -1.0 看空美债(超预期转鹰脉冲), 其他时间常态 0.0。
    """

    def __init__(self):
        # 命名遵守格式：挖掘方向_特征描述_挖掘方法
        self.name = 'unstructured_fomc_pivot_pulse_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 容错处理: 若数据缺失则直接返回全0休眠信号
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 提取 FOMC 文本情绪数据，使用前向填充处理无会议日的空白，保证日频对齐连续性
        fomc = data['fomc_sentiment'].ffill().fillna(0.0)
        
        # 核心铁律3: 边际变化 (Marginal Change)
        # 绝对禁止直接输出绝对水位，使用 5 日边际差分捕捉会议后产生的政策预期势能释放
        chg_5d = fomc.diff(5)
        
        # 选取 1年(252日) 视角的滚动窗口计算变化的相对极端程度
        # 配置 min_periods=20 防止初期无数据直接输出全量 NaN
        roll_mean = chg_5d.rolling(window=252, min_periods=20).mean()
        roll_std = chg_5d.rolling(window=252, min_periods=20).std().replace(0, np.nan)
        
        # 获取 Z-Score。当标准差为 NaN 时填为无穷大(np.inf)使除法趋0，拒绝在死水期触发"假极值"
        z_score = (chg_5d - roll_mean) / roll_std.fillna(np.inf)
        
        # 狙击级触发: 仅在预期发生改变的瞬间（变动生效当日）捕获信号，避免差分指标惯性导致连跳
        pulse_day = fomc.diff(1) != 0
        
        # 核心铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
        # 用前一个月的趋势惯性（20天动量）和历史绝对水位作为当前突变的防飞刀锚准。
        # 只有在原有预期已被拉长至尽头或完全深陷反向逻辑中产生的突跳，才视为真正的"反转衰竭"。
        mom_20d_prev = fomc.diff(20).shift(1).fillna(0.0)
        prev_level = fomc.shift(1).fillna(0.0)
        
        # 脉冲做多条件 (FOMC 极端转鸽利好美债 TLT): 
        # 1) 是变动日; 2) 向上跳升幅度极巨(z>2.5); 3) 原趋势正在鹰化(mom<0)或处于深度鹰派死胡同(<-0.5)
        long_trigger = (
            pulse_day & 
            (z_score > 2.5) & 
            ((mom_20d_prev < 0) | (prev_level < -0.5))
        )
        
        # 脉冲做空条件 (FOMC 极端转鹰利空美债 TLT):
        # 1) 是变动日; 2) 向下跳砸幅度极巨(z<-2.5); 3) 原趋势持续鸽化(mom>0)或处于深度鸽派幻觉(>0.5)
        short_trigger = (
            pulse_day & 
            (z_score < -2.5) & 
            ((mom_20d_prev > 0) | (prev_level > 0.5))
        )
        
        # 写入瞬间基础脉冲，保证铁律的非触发日必为 0.0
        signal = pd.Series(0.0, index=data.index)
        signal.loc[long_trigger] = 1.0
        signal.loc[short_trigger] = -1.0
        
        # 核心铁律1: 零值休眠与 Trigger Rate 调控
        # 因 FOMC 会议属低频离散事件(约每年8次)，单日脉冲不足以满足 5% 触发率，
        # 且债券宏观趋势顺势跟随 (Price-in 消化) 一般持续两周，故后向展期 10 个交易日。
        pos_mask = (signal == 1.0)
        neg_mask = (signal == -1.0)
        
        pos_extend = pos_mask.astype(float).rolling(window=10, min_periods=1).max()
        neg_extend = neg_mask.astype(float).rolling(window=10, min_periods=1).max() * -1.0
        
        final_signal = pos_extend + neg_extend
        
        # 截断极小可能产生的覆盖碰撞，保证输出严格收敛在 [-1.0, 1.0] 内
        final_signal = final_signal.clip(-1.0, 1.0).fillna(0.0)
        final_signal.name = self.name
        
        return final_signal

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"