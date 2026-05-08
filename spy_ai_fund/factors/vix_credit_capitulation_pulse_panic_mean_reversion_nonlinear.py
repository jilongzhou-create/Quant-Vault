import numpy as np
import pandas as pd

class VixCreditCapitulationPulseFactor:
    """VixCreditCapitulationPulseFactor (panic_mean_reversion/nonlinear)

    逻辑: 捕捉美股极度恐慌见顶衰竭瞬间的抄底机会，以及信用环境隐秘恶化初期的做空机会。
          多头(防接飞刀): 当VIX处于近一季度极高位(>80%分位)且今日出现明确回落(跌破3日均值)，同时高收益债信用利差停止走阔时，确立恐慌衰竭点，输出强看多(+1.0)。
          空头(温水煮青蛙): 当高收益债信用利差在5天内显著走阔(>20bp)，且VIX尚未触及极度恐慌(分位<70%)但已处于缓慢上升通道时，确立趋势恶化，输出强看空(-1.0)。
    数据: vixcls (VIX指数), bamlh0a0hym2 (ICE BofA US High Yield Index Option-Adjusted Spread)
    输出: +1.0 (恐慌极值回落，抄底), -1.0 (信用恶化且恐慌起步，看空)
    触发条件: 满足上述非线性一阶导/二阶导衰竭及交叉条件时触发。预期 Trigger Rate: 5%-12%。
    """

    def __init__(self):
        self.name = 'vix_credit_capitulation_pulse'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        # 检查依赖数据是否存在
        if 'vixcls' not in data.columns or 'bamlh0a0hym2' not in data.columns:
            return signal

        vix = data['vixcls'].ffill()
        hy = data['bamlh0a0hym2'].ffill()

        # 计算过去63天(约一个季度)的滚动最小最大值，构建局部相对分位(0~1)，识别相对恐慌极值状态
        vix_63_min = vix.rolling(window=63, min_periods=21).min()
        vix_63_max = vix.rolling(window=63, min_periods=21).max()
        vix_pct = (vix - vix_63_min) / (vix_63_max - vix_63_min + 1e-6)

        hy_63_min = hy.rolling(window=63, min_periods=21).min()
        hy_63_max = hy.rolling(window=63, min_periods=21).max()
        hy_pct = (hy - hy_63_min) / (hy_63_max - hy_63_min + 1e-6)

        # --------------------------------------------------------------------------------
        # 多头逻辑 (+1.0) : 极端恐慌 + 二阶导数衰竭 (绝对禁止直接接飞刀!)
        # 1. 股市恐慌处于相对历史高位: vix_pct > 0.80
        # 2. 股市恐慌开始消退(见顶回落): 今日VIX下降 且 跌破近3日均值
        # 3. 信用市场恐慌未进一步加剧: 今日高收益债利差下降或持平
        # --------------------------------------------------------------------------------
        long_cond = (
            (vix_pct > 0.80) & 
            (vix.diff(1) < 0) & 
            (vix < vix.rolling(window=3).mean()) & 
            (hy.diff(1) <= 0)
        )

        # --------------------------------------------------------------------------------
        # 空头逻辑 (-1.0) : 趋势恶化初段 (钝刀割肉)
        # 1. 信用环境显著恶化: 高收益债利差5日内走阔超 20bp (0.20%)
        # 2. 股市尚未陷入极端恐慌(防空在底部): vix_pct < 0.70 且 hy_pct < 0.95
        # 3. 股市风险悄然积聚: VIX 3日内上升超 1.0 点 且 今日依然上涨
        # --------------------------------------------------------------------------------
        short_cond = (
            (hy.diff(5) > 0.20) & 
            (vix_pct < 0.70) & 
            (vix.diff(3) > 1.0) & 
            (vix.diff(1) > 0) & 
            (hy_pct < 0.95)
        )

        # 写入脉冲信号
        signal.loc[long_cond] = 1.0
        signal.loc[short_cond] = -1.0

        # 清理异常值和缺失期
        signal = signal.fillna(0.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"