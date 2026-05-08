import numpy as np
import pandas as pd

class GlobalSafeHavenSqueezeFactor:
    """全球避险资产挤兑极值与衰竭因子 (panic_mean_reversion/nonlinear)

    逻辑: 当美股面临宏观层面的系统性抛售时，必然伴随着全球美元流动性干涸(美元指数飙升，Cash is King)
          以及避险资产本身的波动率爆炸(黄金波动率飙升，所有资产被无差别抛售)。
          这是完全独立于VIX与信用利差的"汇率+贵金属"避险维度。
          - 钝刀割肉期: 轻微且正在恶化的流动性收紧(美元与黄金波动率双升)会持续对美股抽血，此时应顺势看空(-1.0)防接飞刀。
          - 抄底时刻: 当避险挤兑达到极值(巨震洗盘)且动量开始回落(衰竭)时，意味着流动性危机见顶，此时才触发强看多(+1.0)。
    数据: dtwexbgs (广义美元指数), gvzcls (黄金波动率指数)
    输出: -1.0(流动性收紧恶化中，看空), 0.0(常态), +1.0(避险极度拥挤且开始衰竭，看多)
    触发条件: 综合压力 Z-Score > 2.0 且 3日变动 < 0 (看多)；0.8 < Z-Score <= 2.0 且 3日变动 > 0.15 (看空)。预期 Trigger Rate: 8% ~ 12%
    """

    def __init__(self):
        self.name = 'global_safehaven_squeeze_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 提取所需数据维度
        req_cols = ['dtwexbgs', 'gvzcls']
        missing_cols = [col for col in req_cols if col not in data.columns]
        
        # 若核心维度全军覆没则返回0
        if len(missing_cols) == len(req_cols):
            return signal
            
        # 安全获取数据并前向填充处理节假日
        usd = data['dtwexbgs'].ffill() if 'dtwexbgs' in data.columns else pd.Series(0.0, index=data.index)
        gvz = data['gvzcls'].ffill() if 'gvzcls' in data.columns else pd.Series(0.0, index=data.index)
        
        # 计算 60个交易日(约一季度)的短期 Z-Score，专门捕捉脉冲式的流动性紧张
        if 'dtwexbgs' in data.columns:
            usd_z = (usd - usd.rolling(60).mean()) / usd.rolling(60).std()
        else:
            usd_z = pd.Series(0.0, index=data.index)
            
        if 'gvzcls' in data.columns:
            gvz_z = (gvz - gvz.rolling(60).mean()) / gvz.rolling(60).std()
        else:
            gvz_z = pd.Series(0.0, index=data.index)
            
        usd_z = usd_z.fillna(0)
        gvz_z = gvz_z.fillna(0)
        
        # 构建全球避险挤兑综合压力指数 (Global Safe-Haven Squeeze Index)
        # 逻辑：如果 GVZ (黄金波动率) 缺失，代表在早期样本，此时将美元指标权重*2以对齐阈值量级
        valid_gvz = gvz.notna() & (gvz > 0)
        global_stress = pd.Series(0.0, index=data.index)
        global_stress[valid_gvz] = usd_z[valid_gvz] + gvz_z[valid_gvz]
        global_stress[~valid_gvz] = usd_z[~valid_gvz] * 2.0
        
        # 计算边际衰竭动量 (过去3天差值，避免单日噪音)
        stress_momentum = global_stress.diff(3)
        
        # 【衰竭抄底铁律】强看多条件 (+1.0): 
        # 1. 极端避险挤兑状态 (综合压力 > 2.0，约占正态分布求和后的前端极值)
        # 2. 且压力已经见顶回落 (边际变化 < 0.0，流动性危机度过危险期)
        buy_cond = (global_stress > 2.0) & (stress_momentum < 0.0)
        
        # 【防接飞刀铁律】强看空条件 (-1.0): 
        # 1. 处于逐渐发酵的轻/中度恐慌期 (0.8 < 综合压力 <= 2.0)
        # 2. 且流动性仍在持续收紧恶化中 (边际变化 > 0.15，动量向上，未见底)
        sell_cond = (global_stress > 0.8) & (global_stress <= 2.0) & (stress_momentum > 0.15)
        
        # 合成最终脉冲信号
        signal[sell_cond] = -1.0
        signal[buy_cond] = 1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"