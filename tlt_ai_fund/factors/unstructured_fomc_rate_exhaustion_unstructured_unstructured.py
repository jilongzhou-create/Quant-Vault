import numpy as np
import pandas as pd

class UnstructuredFomcRegimePulseFactor:
    """FOMC情绪突变与动量衰竭因子 (Unstructured/NLP)

    逻辑: 针对之前因在FOMC情绪冲击首日(T+1)追涨杀跌导致IC为负和HitRate过低的问题进行底层重构。
          美联储政策突变极易在首日引发市场的"买预期卖事实" (Buy the rumor, sell the fact) 过度反应与均值回归。
          本因子拒绝在情绪动量最高点入场，而是将阶梯状的文本情绪得分转化为平滑的动量脉冲曲线。
          只有当情绪动量达到极端极值(Z-Score > 2.0)，且动量一阶导开始低于其3日均值(动量峰值衰竭)时才触发信号。
          此时T+1至T+3的获利了结(Whipsaw)震荡已结束，新政策周期(Regime)的主升浪趋势真正确立。
    数据: fomc_sentiment (非结构化文本情绪得分, 1.0=极度鸽派, -1.0=极度鹰派)
    触发: 动量的252日Z-Score极值 + 动量越过峰值开始回落(二阶导衰竭)
    输出: 鸽派确立看多TLT(+1.0)，鹰派确立看空TLT(-1.0)，严格的狙击手脉冲信号。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_regime_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 数据缺失处理
        if 'fomc_sentiment' not in data.columns:
            return pd.Series(0.0, index=data.index)

        # 1. 填补非会议日的空值，保持原数据的阶梯状结构
        fomc = data['fomc_sentiment'].ffill()

        # 2. 边际变化铁律: 将阶梯数据转化为平滑的动量曲线 
        # 使用3日平滑和3日差分，构建出具有明确上升和衰竭阶段的"三角脉冲"，完美刻画市场消化周期
        fomc_smooth = fomc.rolling(window=3).mean()
        fomc_velo = fomc_smooth.diff(3)

        # 3. 计算动量的1年期(252日) Z-Score，识别真正的宏观政策拐点
        roll_mean = fomc_velo.rolling(window=252).mean()
        # 替换 std 为 0 的情况，防止除以 0 导致无穷大
        roll_std = fomc_velo.rolling(window=252).std().replace(0.0, np.nan)
        zscore = (fomc_velo - roll_mean) / roll_std

        # 4. 二阶导数铁律: 动量必须度过最高峰开始衰竭 (防止在情绪过热极值点接飞刀)
        fomc_velo_ma = fomc_velo.rolling(window=3).mean()

        # 鸽派确立 (看多TLT): 极端鸽派突变，且向上的动量已经度过最高点开始回落
        long_cond = (zscore > 2.0) & (fomc_velo > 0) & (fomc_velo < fomc_velo_ma)

        # 鹰派确立 (看空TLT): 极端鹰派突变，且向下的动量已经度过最低点开始回升
        short_cond = (zscore < -2.0) & (fomc_velo < 0) & (fomc_velo > fomc_velo_ma)

        # 5. 零值休眠铁律: 初始化为全 0
        raw_signal = pd.Series(0.0, index=data.index)
        raw_signal[long_cond] = 1.0
        raw_signal[short_cond] = -1.0

        # 6. 脉冲保持: 顺势持有极短的 2 天窗口，确保 Trigger Rate 稳稳落入 5% - 15% 的目标区间
        signal = raw_signal.replace(0.0, np.nan).ffill(limit=2).fillna(0.0)

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"