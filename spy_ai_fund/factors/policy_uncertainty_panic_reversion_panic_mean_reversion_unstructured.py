import numpy as np
import pandas as pd

class PolicyUncertaintyPanicReversionFactor:
    """政策不确定性恐慌均值回归因子 (panic_mean_reversion/unstructured)

    逻辑: 捕捉美国经济政策不确定性(USEPU)作为非结构化恐慌指标的衰竭脉冲。根据SPY均值回归物理属性，当政策恐慌处于极值(Z-Score>1.2)且边际下降时，说明风险溢价出尽，触发强烈看多抄底信号；当不确定性刚从均值上方(0.5~1.2)加速上升时，说明轻度恐慌爆发且趋势恶化，触发看空信号。绝对不接飞刀，必须等恐慌转降才抄底。
    数据: [usepuindxd] (每日经济政策不确定性指数，基于非结构化新闻提取)
    输出: 1.0 强烈看多(恐慌极值回落), -1.0 趋势看空(轻微恐慌初步攀升), 0.0 常态休眠
    触发条件: 狙击手脉冲型，看多需Z-Score>1.2且当天开始衰竭(diff<0)；看空需0.5<Z-Score<=1.2且连续上升高于5日均线。预期Trigger Rate 10%左右。
    """

    def __init__(self):
        self.name = 'policy_uncertainty_panic_reversion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 铁律: 数据不存在时安全返回0.0
        if 'usepuindxd' not in data.columns:
            signal.name = self.name
            return signal
            
        # 填充缺失值
        usepu = data['usepuindxd'].ffill()
        
        # 计算长期 (126个交易日，约半年) 基准均值与标准差
        mean_126 = usepu.rolling(window=126, min_periods=21).mean()
        std_126 = usepu.rolling(window=126, min_periods=21).std()
        
        # 防止除以零导致无限值
        std_126 = std_126.replace(0, np.nan).bfill().fillna(1.0)
        
        # 统计学极值: 计算 Z-Score 衡量恐慌偏离度
        z_score = (usepu - mean_126) / std_126
        
        # 边际变化(二阶导数铁律) - 判断恐慌预期是加剧还是衰竭
        usepu_diff = usepu.diff()
        usepu_5ma = usepu.rolling(window=5, min_periods=1).mean()
        
        # ====================================================================
        # 多头信号 (极度恐慌 + 衰竭) -> 顺应SPY长牛抄底属性
        # 条件: 政策不确定性处于极端高位(Z > 1.2), 并且今天边际回落(预期改善)
        # 禁止直接用绝对值做多，必须叠加 diff < 0 等待衰竭，防接飞刀!
        # ====================================================================
        buy_cond = (z_score > 1.2) & (usepu_diff < 0)
        
        # ====================================================================
        # 空头信号 (轻微恐慌恶化) -> 钝刀子割肉期
        # 条件: 政策不确定性刚突破中等偏离(0.5 < Z <= 1.2)
        # 且正加速上升(当前值不仅上升且高于5日均线)，代表风险刚开始发酵
        # ====================================================================
        sell_cond = (z_score > 0.5) & (z_score <= 1.2) & (usepu_diff > 0) & (usepu > usepu_5ma)
        
        # 写入脉冲信号
        signal.loc[buy_cond] = 1.0
        signal.loc[sell_cond] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"