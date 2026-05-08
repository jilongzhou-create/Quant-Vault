import numpy as np
import pandas as pd

class UnstructuredFomcPivotConfirmationFactor:
    """FOMC情绪与收益率共振脉冲因子 (unstructured/unstructured)

    逻辑: 捕捉美联储声明情绪(fomc_sentiment)发生极端边际反转的瞬间，并强制等待2年期美债收益率(dgs2)在顺势方向上确立衰竭趋势才介入。纯净的狙击手级脉冲，买在政策转向且市场确认不杀回马枪的第一波。
    数据: fomc_sentiment, dgs2
    触发: fomc_sentiment.diff() 突破 2倍标准差 且在其后10天消化期内，dgs2 具备水位安全垫并呈现二阶导数上的反转 (日度与三日动量同向确认)。
    输出: +1.0 看多美债 (鸽派突变且短端利率回落不接飞刀)，-1.0 看空美债 (鹰派突变且短端利率反弹筑底)，常态输出 0.0。
    """

    def __init__(self):
        self.name = 'unstructured_fomc_pivot_confirmation'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0.0, index=data.index)
        
        # 数据完整性检查
        if 'fomc_sentiment' not in data.columns or 'dgs2' not in data.columns:
            return signal

        # 前向填充以处理非交易日或缺失数据
        fomc = data['fomc_sentiment'].ffill()
        dgs2 = data['dgs2'].ffill()

        # -------------------------------------------------------------
        # 1. 边际变化铁律：严禁使用绝对值，只提取阶梯状文本数据的瞬间极值跳跃
        # -------------------------------------------------------------
        fomc_diff = fomc.diff(1)
        
        # 252个交易日滚动分布，衡量本次事件在此前一年维度里的罕见程度
        roll_std = fomc_diff.rolling(window=252, min_periods=21).std()
        roll_std = roll_std.replace(0.0, np.nan).ffill().fillna(0.01)
        
        z_fomc_diff = fomc_diff / roll_std

        # 定义单日的极端跳跃事件 (Sniper事件锚点)
        # 鸽派跳跃：情绪急剧向1.0移动，且前一日偏鹰/中性 (存在巨大预期差)
        dovish_jump = ((z_fomc_diff > 2.0) & (fomc.shift(1) <= 0.0)).astype(float)
        # 鹰派跳跃：情绪急剧向-1.0移动，且前一日偏鸽/中性
        hawkish_jump = ((z_fomc_diff < -2.0) & (fomc.shift(1) >= 0.0)).astype(float)

        # 构造事件爆发后的 10日 市场消化余波窗口
        in_dovish_window = dovish_jump.rolling(window=10, min_periods=1).max() == 1.0
        in_hawkish_window = hawkish_jump.rolling(window=10, min_periods=1).max() == 1.0

        # -------------------------------------------------------------
        # 2. 二阶导数铁律：收益率水位安全垫与回落衰竭确认 (坚决不接飞刀)
        # -------------------------------------------------------------
        # 长期相对水位：用来保证 "不在历史极低点做多TLT，不在历史极高点做空TLT"
        dgs2_mean = dgs2.rolling(window=252, min_periods=21).mean()
        dgs2_std = dgs2.rolling(window=252, min_periods=21).std()
        dgs2_std = dgs2_std.replace(0.0, np.nan).ffill().fillna(0.01)
        z_dgs2 = (dgs2 - dgs2_mean) / dgs2_std
        
        # 短端利率动量：用来保证利率顺势而非逆势杀预期
        dgs2_diff1 = dgs2.diff(1)
        dgs2_diff3 = dgs2.diff(3)

        # -------------------------------------------------------------
        # 3. 极值+衰竭 脉冲触发逻辑
        # -------------------------------------------------------------
        # 做多 TLT：处于鸽派突变余波期，且短端利率未被极度透支 (Z > -0.5 留有安全垫)，
        # 且短端利率确立了连续下行的势头 (动量 < 0)
        long_cond = (
            in_dovish_window & 
            (z_dgs2 > -0.5) & 
            (dgs2_diff1 < 0.0) & 
            (dgs2_diff3 < 0.0)
        )

        # 做空 TLT：处于鹰派突变余波期，且短端利率未触碰天花板透支 (Z < 0.5 留有向上弹性)，
        # 且短端利率确立了回升筑底的势头 (动量 > 0)
        short_cond = (
            in_hawkish_window & 
            (z_dgs2 < 0.5) & 
            (dgs2_diff1 > 0.0) & 
            (dgs2_diff3 > 0.0)
        )

        # 生成脉冲信号 (默认0.0休眠状态)
        signal[long_cond] = 1.0
        signal[short_cond] = -1.0

        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{self.__class__.__name__}()"