import numpy as np
import pandas as pd

class UnstructuredPolicyVolOptionsFactor:
    """宏观不确定性与期权波动率背离因子 (unstructured/options)

    逻辑: 利用 EPU (基于NLP新闻的经济政策不确定性) 与 VIX (期权市场隐含波动率) 的微观背离构建预期差。
          当 EPU 相对 VIX 发生极端向上突变时，表明宏观政策、财政或通胀风险剧增，但期权市场尚未完全恐慌定价，
          此时长端债券期限溢价(Term Premium)被动上升，导致长债遭到抛售，看空美债(TLT)。
          反之，当 VIX 相对 EPU 极端飙升时，表明发生纯粹的微观流动性恐慌与去杠杆，宏观基本面预期未变，
          驱动资金大规模涌入避险资产(Flight to Safety)，看多美债(TLT)。
    数据: usepuindxd, vixcls
    输出: 脉冲型信号 [-1.0, 1.0]，极端背离时产生方向性脉冲，非触发日严格为 0.0
    """

    def __init__(self):
        self.name = 'unstructured_policy_vol_options'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 初始化全零信号
        signal = pd.Series(0.0, index=data.index, name=self.name)
        
        # 字段检查
        if 'usepuindxd' not in data.columns or 'vixcls' not in data.columns:
            return signal

        epu = data['usepuindxd'].ffill()
        vix = data['vixcls'].ffill()

        # 确保数据为正且有效，避免被除数为 0 或计算出负的异常背离
        valid_idx = (epu > 0) & (vix > 0) & epu.notna() & vix.notna()
        if not valid_idx.any():
            return signal

        # 计算比率背离序列
        policy_div = pd.Series(np.nan, index=data.index)
        panic_div = pd.Series(np.nan, index=data.index)

        # policy_div: 政策不确定性/市场恐慌 的单位风险
        policy_div.loc[valid_idx] = epu.loc[valid_idx] / vix.loc[valid_idx]
        # panic_div: 市场恐慌/政策不确定性 的单位溢价
        panic_div.loc[valid_idx] = vix.loc[valid_idx] / epu.loc[valid_idx]

        # 采用126个交易日(约半年)滚动窗口计算基准动态 Z-Score，反映最新宏观周期背景下的异常偏离
        window = 126
        min_periods = 63

        policy_mean = policy_div.rolling(window=window, min_periods=min_periods).mean()
        policy_std = policy_div.rolling(window=window, min_periods=min_periods).std()
        policy_z = (policy_div - policy_mean) / (policy_std + 1e-6)

        panic_mean = panic_div.rolling(window=window, min_periods=min_periods).mean()
        panic_std = panic_div.rolling(window=window, min_periods=min_periods).std()
        panic_z = (panic_div - panic_mean) / (panic_std + 1e-6)

        # 触发核心因子逻辑
        # 条件1：宏观政策不确定性主导的脉冲 -> 期限溢价飙升 -> 看空美债 (-1.0)
        trigger_short = policy_z > 2.5
        
        # 条件2：微观流动性恐慌主导的脉冲 -> 极致避险情绪 -> 看多美债 (+1.0)
        trigger_long = panic_z > 2.5

        signal.loc[trigger_short] = -1.0
        signal.loc[trigger_long] = 1.0

        # 处理可能出现的冲突日(极小概率两端同日超过阈值) -> 取消信号保持中立
        conflict = trigger_short & trigger_long
        signal.loc[conflict] = 0.0

        # 收尾清洗，确保非触发日严格为 0.0
        signal = signal.fillna(0.0).clip(-1.0, 1.0)

        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"