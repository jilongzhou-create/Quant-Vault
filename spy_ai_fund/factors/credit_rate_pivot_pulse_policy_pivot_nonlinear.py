import numpy as np
import pandas as pd

class PolicyPivotCreditPulseFactor:
    """Policy Pivot & Credit Nonlinear Pulse (policy_pivot/nonlinear)

    逻辑: 结合短期利率预期(DGS2)、收益率曲线变陡(T10Y2Y)与信用利差(HY OAS)脉冲。
          1. 当2年期国债收益率急跌+收益率曲线急剧变陡+信用利差稳定时，表明市场抢跑“软着陆降息”，释放强流动性冲量，看多(+1.0)。
          2. 当信用利差急剧飙升但尚未达到极值时，表明轻微至中度恐慌蔓延，看空(-1.0)。
          3. 当信用利差处于历史高位(Z>2.0)且当日回落时，标志着恐慌极值衰竭，迎来抄底买点，看多(+1.0)。
          4. 2年期国债收益率急升且信用未见改善，表明鹰派紧缩冲击，看空(-1.0)。
    数据: dgs2, t10y2y, bamlh0a0hym2
    输出: +1.0 看多, -1.0 看空
    触发条件: 3日动量Z-Score极值状态及恐慌衰竭点组合触发，常态返回0.0。预期Trigger Rate: 8%-12%。
    """

    def __init__(self):
        self.name = 'policy_pivot_credit_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 检查依赖列是否存在
        req_cols = ['bamlh0a0hym2', 'dgs2', 't10y2y']
        for col in req_cols:
            if col not in data.columns:
                return pd.Series(0.0, index=data.index)
        
        # 向前填充缺失值（处理节假日/交易日不对齐问题）
        df = data[req_cols].ffill()
        
        # 设置经济学滚动窗口: 252为约1个交易年, 3为捕捉极短冲量
        window = 252
        diff_win = 3
        
        hy = df['bamlh0a0hym2']
        dgs2 = df['dgs2']
        t10y2y = df['t10y2y']
        
        # 计算短期动量变化
        hy_diff = hy.diff(diff_win)
        dgs2_diff = dgs2.diff(diff_win)
        t10y2y_diff = t10y2y.diff(diff_win)
        
        # 安全计算滚动标准差 (防除零)
        hy_std = hy_diff.rolling(window).std().replace(0, np.nan)
        dgs2_std = dgs2_diff.rolling(window).std().replace(0, np.nan)
        t10y2y_std = t10y2y_diff.rolling(window).std().replace(0, np.nan)
        hy_lvl_std = hy.rolling(window).std().replace(0, np.nan)
        
        # 计算动量变化的Z-Score
        hy_z = (hy_diff - hy_diff.rolling(window).mean()) / hy_std
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(window).mean()) / dgs2_std
        t10y2y_z = (t10y2y_diff - t10y2y_diff.rolling(window).mean()) / t10y2y_std
        
        # 计算信用利差绝对水位的Z-Score (用于捕捉恐慌极值点)
        hy_lvl_z = (hy - hy.rolling(window).mean()) / hy_lvl_std
        
        # 默认信号输出为0.0
        signal = pd.Series(0.0, index=df.index)
        
        # ==========================================
        # 负向脉冲逻辑 (看空 -1.0)
        # ==========================================
        
        # 1. 鹰派冲击: 短端利率急剧上升 (Z > 1.5) 且 信用利差并未显著收窄对冲风险 (Z > -0.5)
        sell_hawkish = (dgs2_z > 1.5) & (hy_z > -0.5)
        
        # 2. 恐慌初期蔓延: 信用利差急剧扩大 (Z > 1.5)，但绝对值尚未达到恐慌极值地带
        sell_panic = (hy_z > 1.5) & (hy_lvl_z < 2.0)
        
        signal.loc[sell_hawkish | sell_panic] = -1.0
        
        # ==========================================
        # 正向脉冲逻辑 (看多 +1.0)
        # ==========================================
        
        # 1. 软着陆式抢跑降息: 短端利率急剧下挫 (Z < -1.5) + 曲线出现经典牛陡 (Z > 0.5) + 且非信用爆雷导致 (Z < 0.5)
        buy_pivot = (dgs2_z < -1.5) & (t10y2y_z > 0.5) & (hy_z < 0.5)
        
        # 2. 恐慌极值与衰竭(二阶导防接飞刀): 信用利差水位极度恐慌 (Z > 2.0) + 但今日边际动量转负(恐慌开始回落)
        buy_exhaustion = (hy_lvl_z > 2.0) & (hy.diff(1) < 0.0)
        
        # 多头信号执行 (覆写相同日期可能冲突的空头信号，抄底有最高优先级)
        signal.loc[buy_pivot | buy_exhaustion] = 1.0
        
        return signal.fillna(0.0)

    def __repr__(self):
        return f"{self.__class__.__name__}()"