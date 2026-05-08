import numpy as np
import pandas as pd

class UnstructuredOptionsStressFactor:
    """非结构化政策压力与期权波动率共振因子 (unstructured/options)

    逻辑: 结合新闻文本测算的经济政策不确定性(EPU)与期权隐含波动率(VIX)构建宏观压力指数。
          绝对不逆势接飞刀！当宏观压力极度高企且开始出现衰竭(二阶回落)，
          同时2年期美债收益率暴跌且收益率曲线急剧变陡(Bull Steepening)时，此时确立美联储Dovish Pivot，脉冲看多美债(TLT)。
          反之，当宏观极度自满且压力开始抬头，伴随2年期上行和曲线平坦化(Bear Flattening)时，定价紧缩，脉冲看空。
    数据: usepuindxd (EPU), vixcls (VIX), dgs2 (短端利率), t10y2y (期限利差)
    触发: 压力Z-Score极端 + 二阶拐点 + 曲线形态验证
    输出: [-1.0, 1.0] 狙击手脉冲信号
    """

    def __init__(self):
        self.name = 'unstructured_options_stress_factor'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['usepuindxd', 'vixcls', 'dgs2', 't10y2y']
        for col in required_cols:
            if col not in data.columns:
                return signal
                
        # 1. 基础数据前向填充，处理缺失
        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()
        dgs2 = data['dgs2'].ffill()
        t10y2y = data['t10y2y'].ffill()
        
        # 2. 构建非结构化与期权联合宏观压力指数 (Stress Index)
        # EPU 基于文本抓取，噪音较大，使用 5日均值平滑
        epu_smooth = epu.rolling(window=5, min_periods=1).mean()
        
        # 分别计算 252 日滚动 Z-Score
        epu_mean = epu_smooth.rolling(window=252, min_periods=60).mean()
        epu_std = epu_smooth.rolling(window=252, min_periods=60).std()
        epu_z = (epu_smooth - epu_mean) / (epu_std + 1e-8)
        
        vix_mean = vix.rolling(window=252, min_periods=60).mean()
        vix_std = vix.rolling(window=252, min_periods=60).std()
        vix_z = (vix - vix_mean) / (vix_std + 1e-8)
        
        # 合并 非结构化文本不确定性 与 期权市场恐慌度 -> 宏观压力综合 Z-Score
        stress_z = (epu_z + vix_z) / 2.0
        
        # 3. 计算边际变化与动量衰竭 (严格遵守边际变化与二阶导数铁律)
        stress_ma5 = stress_z.rolling(window=5, min_periods=1).mean()
        dgs2_ma5 = dgs2.rolling(window=5, min_periods=1).mean()
        t10y2y_ma5 = t10y2y.rolling(window=5, min_periods=1).mean()
        
        # 4. 触发逻辑
        
        # 看多 TLT (Bull Steepening 驱动)：
        # 条件A: 宏观压力高企 (Z > 0.8，约对应前15%分位)
        # 条件B: 压力开始二阶衰竭 (Anti-Catch-Falling-Knife，小于5日均值)
        # 条件C: 2年期短端利率边际急剧下行 (定价降息预期)
        # 条件D: 期限利差边际变陡 (经典的牛市变陡形态，确立宽松周期)
        long_cond = (stress_z > 0.8) & \
                    (stress_z < stress_ma5) & \
                    (dgs2 < dgs2_ma5) & \
                    (t10y2y > t10y2y_ma5)
                    
        # 看空 TLT (Bear Flattening 驱动)：
        # 条件A: 宏观压力极度自满 (Z < -0.5，约对应后25%分位)
        # 条件B: 压力开始边际反弹抬头
        # 条件C: 2年期短端利率边际急剧上行 (定价加息或通胀失控)
        # 条件D: 期限利差边际平坦化 (经典的熊市平坦形态，确立紧缩周期)
        short_cond = (stress_z < -0.5) & \
                     (stress_z > stress_ma5) & \
                     (dgs2 > dgs2_ma5) & \
                     (t10y2y < t10y2y_ma5)
                     
        # 5. 狙击手脉冲赋值 (常态下保持0值休眠)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0
        
        signal.name = self.name
        return signal