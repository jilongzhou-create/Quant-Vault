import numpy as np
import pandas as pd

class LiquidityPivotNonlinearFactor:
    """流动性转向与信用状态非线性交叉因子 (policy_pivot/nonlinear)

    逻辑: 捕捉政策利率预期剧变与企业信用环境的非线性组合。短端利率急跌且信用利差收窄代表软着陆的宽松冲量(强看多)；短端利率急涨且信用承压代表紧缩冲击(看空)；短端利率急跌但信用利差飙升代表硬着陆衰退恐慌(看空)。
    数据: [dgs2, bamlh0a0hym2]
    输出: [-1.0, 0.0, 1.0] 的脉冲信号
    触发条件: 5日动量的252日Z-Score分别达到极端阈值且实现经济学逻辑交叉，预期Trigger Rate在 8%-12% 之间。
    """

    def __init__(self):
        self.name = 'liquidity_pivot_nonlinear'
        self.window = 252
        self.momentum_days = 5

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 默认返回全 0.0 的 Series
        signal = pd.Series(0.0, index=data.index)
        
        # 检查所需数据是否缺失
        if 'dgs2' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            signal.name = self.name
            return signal
            
        # 提取所需列并前向填充以防单个字段某天休市缺失
        df = data[['dgs2', 'bamlh0a0hym2']].ffill()
        
        # 铁律8: 边际变化铁律。禁止绝对值，计算5日动量变化捕捉短期的瞬间冲量
        dgs2_diff = df['dgs2'].diff(self.momentum_days)
        hy_diff = df['bamlh0a0hym2'].diff(self.momentum_days)
        
        # 铁律5 & 7: 经济学含义与防极值陷阱。使用滚动一年的 Z-Score 识别“极值突变”状态
        dgs2_z = (dgs2_diff - dgs2_diff.rolling(self.window).mean()) / dgs2_diff.rolling(self.window).std()
        hy_z = (hy_diff - hy_diff.rolling(self.window).mean()) / hy_diff.rolling(self.window).std()
        
        # 场景1: 软着陆预防式宽松冲量 -> 看多 (+1.0)
        # 逻辑: 短端利率(降息预期)急剧下移，同时信用利差也在收窄(排除了经济硬着陆危机)
        bull_cond = (dgs2_z < -1.2) & (hy_z < -0.2)
        
        # 场景2: 紧缩超预期恶化冲击 -> 看空 (-1.0)
        # 逻辑: 短端利率急剧攀升(预期加息/higher-for-longer)，企业信用开始承压扩大
        hawk_shock_cond = (dgs2_z > 1.2) & (hy_z > 0.5)
        
        # 场景3: 衰退恐慌 / 流动性危机 -> 看空 (-1.0)
        # 逻辑: 市场在疯狂抢跑降息(利率急跌)，但是因为高收益债信用利差爆雷狂飙，说明遇到了极端的信用抛售
        recession_panic_cond = (dgs2_z < -1.0) & (hy_z > 1.5)
        
        # 触发脉冲信号
        signal.loc[bull_cond] = 1.0
        signal.loc[hawk_shock_cond] = -1.0
        signal.loc[recession_panic_cond] = -1.0
        
        # 过滤掉由于前期窗口不足 252 天产生的 NaN，并强制填充 0.0 保障底层安全
        signal = signal.fillna(0.0)
        signal.name = self.name
        
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}(window={self.window}, momentum_days={self.momentum_days})"