import numpy as np
import pandas as pd

class EpuPanicExhaustionFactor:
    """Epu Panic Exhaustion (panic_mean_reversion/unstructured)

    逻辑: 专门捕捉基于非结构化新闻文本计算的经济政策不确定性(EPU)的极值反转现象。
          美股SPY的长牛物理属性决定了：当不确定性指数在近期飙升至年度高位(极度恐慌)，且短期动量刚刚转弱(新闻热度见顶回落，即均线死叉)，代表恐慌见顶衰竭，这提供了一个高胜率的抄底买点，输出看多(+1.0)。
          相反，当市场处于未见极值的平静期，但短期不确定性刚刚抬头(轻度恐慌缓慢攀升，即短均线上穿长均线)，代表钝刀割肉的下跌趋势刚启动，输出看空(-1.0)。
    数据: [usepuindxd] (Daily US Economic Policy Uncertainty Index, NLP-based)
    输出: +1.0 表示强烈看多的抄底脉冲, -1.0 表示看跌趋势起点的脉冲。
    触发条件: 过去15天内极大值(Z>1.5) + 短期动量下穿(死叉)时输出+1；非极值状态下短期动量上穿(金叉)时输出-1。预期 Trigger Rate 8%-12%。
    """

    def __init__(self):
        self.name = 'epu_panic_exhaustion_unstructured'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 如果缺少所需核心字段，则全量返回0.0的Series
        if 'usepuindxd' not in data.columns:
            return pd.Series(0.0, index=data.index, name=self.name)

        # 确保数据为正且向下填充缺失值 (EPU理论上大于0，clip确保对数计算安全)
        epu = data['usepuindxd'].clip(lower=1.0).ffill()
        
        # 经济政策不确定性指数的分布高度右偏，进行对数转换使其更贴近正态分布，平滑长尾极值
        log_epu = np.log(epu)

        # 构建短、中、长三条移动平均线，用以过滤每日新闻的高频噪音，提取趋势
        ma3 = log_epu.rolling(window=3).mean()
        ma10 = log_epu.rolling(window=10).mean()
        ma20 = log_epu.rolling(window=20).mean()

        # 计算年度滚动 Z-Score (252个交易日)，赋予阈值明确的经济学极值意义
        mean_252 = log_epu.rolling(window=252).mean()
        std_252 = log_epu.rolling(window=252).std()
        z_score = (log_epu - mean_252) / std_252

        # ----------------------------------------------------
        # 条件1: 极度恐慌 + 衰竭 -> 抄底买入脉冲 (+1.0)
        # ----------------------------------------------------
        # 一阶条件：过去15天内，不确定性至少有一次突破年度均值上方1.5个标准差 (确认处于年度极端高波恐慌状态)
        panic_extreme = z_score.rolling(window=15).max() >= 1.5
        
        # 二阶导数条件：今日短期均线向下击穿中期均线 (边际变化: 恐慌新闻热度确实开始消退衰竭)
        exhaustion_cross = (ma3 < ma10) & (ma3.shift(1) >= ma10.shift(1))
        
        buy_signal = panic_extreme & exhaustion_cross

        # ----------------------------------------------------
        # 条件2: 轻度恐慌缓慢发酵 -> 避险卖出脉冲 (-1.0)
        # ----------------------------------------------------
        # 一阶条件：近15天最高Z-Score未触及1.0的恐慌线 (确认处于常态或平静期，排除暴跌过程中的噪音或二次探底导致的轧空)
        not_extreme = z_score.rolling(window=15).max() < 1.0
        
        # 边际恶化条件：短期不确定性刚刚上穿中期基准 (趋势转坏，轻度恐慌刚开始抬头)
        slow_worsening = (ma3 > ma20) & (ma3.shift(1) <= ma20.shift(1))
        
        sell_signal = not_extreme & slow_worsening

        # ----------------------------------------------------
        # 信号合成 (休眠铁律: 默认0.0)
        # ----------------------------------------------------
        signal = pd.Series(0.0, index=data.index)
        signal[buy_signal] = 1.0
        signal[sell_signal] = -1.0
        
        signal.name = self.name
        return signal