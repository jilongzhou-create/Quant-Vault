import numpy as np
import pandas as pd

class DollarLiquidityPanicExhaustionFactor:
    """美元流动性恐慌极值与均值回归因子 (panic_mean_reversion/nonlinear)

    逻辑: 强美元(dtwexbgs)引发全球流动性挤兑，配合高VIX(vixcls)代表极端系统性恐慌。当两者达到历史极高位并同步出现衰竭(拐头向下)时，触发生死存亡后的长牛均值回归（强看多）。若美元和VIX在常态下同步温和走高，则为温水煮青蛙式的流动性收紧，输出看空。
    数据: dtwexbgs(广义美元指数), vixcls(VIX指数)
    输出: +1.0 表示流动性危机见顶衰竭(强烈看多), -1.0 表示流动性温和收紧发酵(恶化看空), 0.0 处于常态无信号
    触发条件: 极值Z-score>1.0且相加>2.5并伴随动量回落为+1.0；常态下短期急升为-1.0。预期 Trigger Rate 5%-15%
    """

    def __init__(self):
        self.name = 'dollar_liquidity_panic_exhaustion'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)

        if 'dtwexbgs' not in data.columns or 'vixcls' not in data.columns:
            return signal
            
        # 填充缺失值，周末或节假日汇率可能有NaN
        dollar = data['dtwexbgs'].ffill()
        vix = data['vixcls'].ffill()

        # 计算 252 日 (约一年) 的长期 Z-Score 来识别宏观系统性状态
        dollar_mean = dollar.rolling(window=252).mean()
        dollar_std = dollar.rolling(window=252).std()
        dollar_z = (dollar - dollar_mean) / (dollar_std + 1e-8)

        vix_mean = vix.rolling(window=252).mean()
        vix_std = vix.rolling(window=252).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)

        # === 看多逻辑：极度恐慌 + 边际衰竭 (抄底接飞刀防护) ===
        # 极值条件：两者都高于长期均值1个标准差以上，且综合恐慌程度极高 (总和>2.5)
        is_extreme_panic = (dollar_z > 1.0) & (vix_z > 1.0) & ((dollar_z + vix_z) > 2.5)
        
        # 衰竭条件：二阶导数向下，恐慌情绪和流动性压力双双边际缓解
        # 美元指数不再创新高，且回落至前3日均值下方 (汇率的衰竭)
        dollar_exhausted = (dollar.diff(1) < 0) & (dollar < dollar.rolling(window=3).mean())
        # VIX 显著见顶回落 (恐慌情绪的衰竭)
        vix_exhausted = (vix.diff(1) < 0) & (vix < vix.rolling(window=3).mean())

        buy_condition = is_extreme_panic & dollar_exhausted & vix_exhausted

        # === 看空逻辑：温水煮青蛙，轻度恐慌恶化 ===
        # 没有达到极值洗盘的状态，处于历史中等偏上水平
        is_mild_stress = (vix_z > 0.0) & (vix_z <= 1.25) & (dollar_z > 0.0) & (dollar_z <= 1.25)
        
        # 短期内两资产同步急升，流动性正在隐秘抽干且引发市场不安
        # 美元5日涨幅超0.5% (对于加权汇率已是明显上涨)，且当天继续升值发酵
        dollar_surging = (dollar.pct_change(5) > 0.005) & (dollar.diff(1) > 0)
        # VIX 5日内抬升超过2.0点，且当天继续小幅抬升
        vix_creeping = (vix.diff(5) > 2.0) & (vix.diff(1) > 0)

        sell_condition = is_mild_stress & dollar_surging & vix_creeping

        # 赋值脉冲信号
        signal[buy_condition] = 1.0
        signal[sell_condition] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"