import numpy as np
import pandas as pd

class CrossAssetVolRegimeFactor:
    """波动率极值与通胀状态交叉反转 (volatility/nonlinear)

    逻辑: 跨资产波动率(VIX+信用利差+利率曲线)极端飙升代表宏观恐慌。但恐慌衰竭时, 资金流向取决于恐慌的性质:
          1. 若伴随通胀预期上升(滞胀恐慌), 衰竭时债市见底反弹 (做多TLT)。
          2. 若伴随通胀预期下降(衰退恐慌), 衰竭时避险资金撤出, Risk-On重启, 债市下跌 (做空TLT)。
          以此彻底解决纯波动率衰竭因子预测方向模糊、条件IC为负的致命缺陷。
    数据: vixcls, bamlc0a4cbbb, t10y2y, t10yie
    触发: 跨资产波动率综合Z-Score > 1.0 且开始回落 (diff < 0), 结合20日通胀预期动量判断方向。
    输出: 脉冲信号, +1.0 看多美债, -1.0 看空美债。
    """

    def __init__(self):
        self.name = 'cross_asset_vol_regime_reversal'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 1. 零值休眠铁律: 初始化全0脉冲序列
        signal = pd.Series(0.0, index=data.index)
        
        # 计算各维度波动率Z-Score (使用252日滚动以符合宏观周期)
        z_scores = {}
        
        # 维度A: 股票市场恐慌
        if 'vixcls' in data.columns:
            vix = data['vixcls'].ffill()
            vix_z = (vix - vix.rolling(252, min_periods=60).mean()) / vix.rolling(252, min_periods=60).std()
            z_scores['vix'] = vix_z
            
        # 维度B: 信用市场恐慌 (BBB企业债利差水位代表信用紧缩压力)
        if 'bamlc0a4cbbb' in data.columns:
            cred = data['bamlc0a4cbbb'].ffill()
            cred_z = (cred - cred.rolling(252, min_periods=60).mean()) / cred.rolling(252, min_periods=60).std()
            z_scores['cred'] = cred_z
            
        # 维度C: 利率市场波动 (利用边际变化铁律: 期限利差日间波动的20日标准差代表宏观重定价剧烈度)
        if 't10y2y' in data.columns:
            spread = data['t10y2y'].ffill()
            curve_vol = spread.diff().rolling(20, min_periods=5).std()
            curve_z = (curve_vol - curve_vol.rolling(252, min_periods=60).mean()) / curve_vol.rolling(252, min_periods=60).std()
            z_scores['curve'] = curve_z
            
        # 缺失列异常处理
        if not z_scores:
            signal.name = self.name
            return signal
            
        # 构建跨资产恐慌综合指数
        z_df = pd.DataFrame(z_scores)
        total_vol = z_df.mean(axis=1)
        # 对综合指数再做一次Z-Score
        total_vol_z = (total_vol - total_vol.rolling(252, min_periods=60).mean()) / total_vol.rolling(252, min_periods=60).std()
        
        # 2. 二阶导数铁律: 绝对禁止直接追高，必须等待衰竭
        # 阈值 1.0 (约前15%极端位) 以确保触发率在 5-15% 的合理目标区间内
        is_extreme = total_vol_z > 1.0
        # 衰竭判定: 动量跌破0 且 跌破近期均值
        is_exhausting = (total_vol_z.diff() < 0) & (total_vol_z < total_vol_z.rolling(3).mean())
        cond_trigger = is_extreme & is_exhausting
        
        # 机制状态划分: 运用 10年期盈亏平衡通胀 区分冲击底色
        if 't10yie' in data.columns:
            inf_be = data['t10yie'].ffill()
            inf_mom = inf_be.diff(20)
        elif 'dcoilwtico' in data.columns:
            oil = data['dcoilwtico'].ffill()
            inf_mom = oil.diff(20)
        else:
            inf_mom = pd.Series(0.0, index=data.index)
            
        # 清洗 NaN 造成的布尔传播异常
        cond_trigger = cond_trigger.fillna(False)
        inf_mom = inf_mom.fillna(0.0)
        
        # 3. 赋值脉冲信号
        # 通胀恐慌衰竭 -> 紧缩见顶(如2022下半年) -> 美债报复性反弹 (+1.0)
        signal[cond_trigger & (inf_mom > 0)] = 1.0
        # 衰退恐慌衰竭 -> 避险盘瓦解(如2020年3月底) -> 资金回流股市，美债遭抛售 (-1.0)
        signal[cond_trigger & (inf_mom < 0)] = -1.0
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"