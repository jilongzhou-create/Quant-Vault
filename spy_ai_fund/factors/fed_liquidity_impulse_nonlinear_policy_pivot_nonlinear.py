import numpy as np
import pandas as pd

class FedLiquidityImpulseNonlinearFactor:
    """美联储流动性冲量非线性因子 (policy_pivot/nonlinear)

    逻辑: 捕捉美债短端利率(DGS2)剧烈变化导致收益率曲线骤然变陡/变平的时刻。
          真正的买点发生在短端利率急跌(市场抢跑降息)驱动的曲线变陡(Bull Steepening)，
          且同时信用利差(BAA10YM)未显著恶化(排除硬着陆带来的恐慌性利率下行)。
    数据: dgs2 (2年期美债), t10y2y (期限利差), baa10ym (企业债信用利差)
    输出: +1.0 表示鸽派宽松预期突变(多)，-1.0 表示鹰派紧缩预期突变(空)
    触发条件: DGS2 5日单边变动超12bps(约计入半次基准利率变动预期) + 曲线变陡/变平 + 当日动量延续。预期 Trigger Rate 5%-15%。
    """

    def __init__(self):
        self.name = 'fed_liquidity_impulse_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['dgs2', 't10y2y', 'baa10ym']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            return signal
            
        # 1. 数据对齐与前向填充
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        baa = data['baa10ym'].ffill()
        
        # 2. 计算边际变化冲量 (5个交易日捕捉一周内的预期巨变)
        dgs2_5d = dgs2.diff(5)
        t10y2y_5d = t10y2y.diff(5)
        
        # 3. 日度动量，确保信号触发当天仍处于顺势动能中 (非逆势飞刀)
        dgs2_1d = dgs2.diff(1)
        
        # 4. 信用利差防衰退飞刀保护 (3个交易日的波动情况)
        baa_3d = baa.diff(3)
        
        # -------------------------------------------------------------
        # 多头逻辑 (Bull Steepening + 金发姑娘预期): 
        # 1. 2年期利率 5 日急跌超 12bps (市场抢跑鸽派降息)
        # 2. 当日 2 年期利率继续下跌 (动量未衰竭)
        # 3. 收益率曲线变陡超 4bps (典型的流动性宽松陡峭化)
        # 4. 信用利差未在 3 日内暴涨超过 5bps (防死于信用违约/衰退危机)
        # -------------------------------------------------------------
        cond_buy = (
            (dgs2_5d <= -0.12) & 
            (dgs2_1d < 0.0) & 
            (t10y2y_5d >= 0.04) & 
            (baa_3d <= 0.05)
        )
        
        # -------------------------------------------------------------
        # 空头逻辑 (Bear Flattening / 鹰派冲击): 
        # 1. 2年期利率 5 日急升超 12bps (通胀重燃，加息预期反扑)
        # 2. 当日 2 年期利率继续上升
        # 3. 收益率曲线变平超 4bps (典型的紧缩杀估值期)
        # 4. 信用利差没有显著收窄 (市场未受到极度正面的基本面对冲)
        # -------------------------------------------------------------
        cond_sell = (
            (dgs2_5d >= 0.12) & 
            (dgs2_1d > 0.0) & 
            (t10y2y_5d <= -0.04) & 
            (baa_3d >= -0.05)
        )
        
        signal.loc[cond_buy] = 1.0
        signal.loc[cond_sell] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"