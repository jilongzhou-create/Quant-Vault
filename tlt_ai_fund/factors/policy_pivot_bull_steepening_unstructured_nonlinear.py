import numpy as np
import pandas as pd

class UnstructuredPanicPivotNonlinearFactor:
    """非线性政策恐慌与衰竭因子 (unstructured/nonlinear)

    逻辑: 结合非结构化政策不确定性(EPU)与金融压力(FSI), 当双重恐慌指数极度飙升后开始衰竭, 且短端利率(dgs2)下行确认美联储降息预期时, 形成戴维斯双击看多美债(脉冲)。反之在极度自满且利率上行时看空。
    数据: usepuindxd (政策不确定性), stlfsi4 (金融压力), dgs2 (两年期美债)
    触发: 综合Z-Score极值 + 二阶导数转向(衰竭) + dgs2均线突破确认。满足后信号维持3天以达到目标Trigger Rate。
    输出: 脉冲型 [-1.0, 1.0], 正值看多美债(TLT)。
    """

    def __init__(self):
        self.name = 'unstructured_panic_pivot_nonlinear'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        required_cols = ['usepuindxd', 'stlfsi4', 'dgs2']
        if not all(col in data.columns for col in required_cols):
            return signal

        # 处理数据缺失 (金融数据可能因为发布频率存在少量缺失，使用前向填充)
        df = data[required_cols].ffill().bfill()

        # 1. 宏观状态 Z-scores (使用252个交易日约1年窗口，捕捉宏观大周期极值)
        epu = df['usepuindxd']
        fsi = df['stlfsi4']
        
        epu_z = (epu - epu.rolling(252).mean()) / (epu.rolling(252).std() + 1e-5)
        fsi_z = (fsi - fsi.rolling(252).mean()) / (fsi.rolling(252).std() + 1e-5)

        # 2. 复合恐慌指数 (非线性特征交叉：政策不确定性与金融压力的共振)
        panic_idx = epu_z + fsi_z

        # 3. 极值衰竭条件 (严格遵守 Anti-Catch-Falling-Knife 铁律)
        # 极度恐慌且开始见顶回落 (当前值低于5日均线, 且2日动量为负)
        panic_exhaustion_high = (panic_idx < panic_idx.rolling(5).mean()) & (panic_idx.diff(2) < 0)
        
        # 极度自满且开始复苏发散 (当前值高于5日均线, 且2日动量为正)
        panic_exhaustion_low = (panic_idx > panic_idx.rolling(5).mean()) & (panic_idx.diff(2) > 0)

        # 4. 短端利率确认条件 (严格遵守 边际变化 铁律: dgs2预期突变)
        # dgs2 下破 5日均线 (短端利率剧烈下行 = 美联储降息预期骤升 = 利多美债)
        dgs2_falling = df['dgs2'] < df['dgs2'].rolling(5).mean()
        # dgs2 上破 5日均线 (短端利率突破上行 = 美联储加息/紧缩预期 = 利空美债)
        dgs2_rising = df['dgs2'] > df['dgs2'].rolling(5).mean()

        # 5. 核心脉冲触发逻辑
        # 金融数据通常右偏，恐慌飙升峰值极高，而自满谷底较浅，故阈值非对称以保证触发率平衡
        long_cond = (panic_idx > 2.0) & panic_exhaustion_high & dgs2_falling
        short_cond = (panic_idx < -1.8) & panic_exhaustion_low & dgs2_rising

        # 6. 信号延展 (Pulse Extension) 维持3天，确保Trigger Rate处于 5%-15% 的黄金狙击区间
        long_pulse = long_cond.fillna(False).rolling(3).max() == 1
        short_pulse = short_cond.fillna(False).rolling(3).max() == 1

        # 7. 零值休眠信号赋值 (满足触发率铁律)
        signal[long_pulse] = 1.0
        # 排除多空信号在极端异构下同时发生的情境 (如有，归零)
        signal[short_pulse] = np.where(signal[short_pulse] == 1.0, 0.0, -1.0)
        
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"