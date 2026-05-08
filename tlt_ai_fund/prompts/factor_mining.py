#!/usr/bin/env python3
"""
LLM Prompt 模板 - TLT FICC 因子挖掘 (v2: 注入 SPY 反思三大铁律)

核心约束:
  - 只允许 3 个 FICC 专属挖掘方向 x 3 种挖掘方法
  - 所有卫星因子必须是"狙击手级脉冲" (Z-Score > 2.5 触发, 其余时间 0.0)
  - 三大铁律: 零值休眠 / 二阶导数(防接飞刀) / 边际变化
  - 生成的因子必须实现 BaseFactor 接口
  - 禁止引用 CoreAnchor 数据 (DFII10/DGS10/BAMLH0A0HYM2)
  - 信号输出 [-1.0, 1.0], 正值看多美债, 负值看空美债
"""

DIRECTION_DESCRIPTIONS = {
    'unstructured': (
        "方向A: 政策预期突变 (Policy Pivot Shock)\n"
        "目标: 捕捉美联储政策预期的极端跳跃, 在市场尚未完全 Price-in 时生成脉冲\n"
        "\n"
        "典型数据:\n"
        "  - fomc_sentiment: FOMC鹰鸽情绪得分, [-1,1], 1.0=极度鸽派=看多美债\n"
        "  - dgs2: 2年期美债收益率 (对政策预期最敏感的前瞻指标)\n"
        "  - t10y2y: 10年-2年利差 (曲线形态变化 = 政策转向信号)\n"
        "  - dff: 联邦基金有效利率\n"
        "\n"
        "核心逻辑 (必须遵守边际变化铁律!):\n"
        "  - 绝对禁止直接使用 fomc_sentiment 的绝对值! 它是低频阶梯数据(每年约8次变化)\n"
        "  - 必须使用 fomc_sentiment.diff() 或滚动变化量来捕捉预期突变瞬间\n"
        "  - 正确示例: fomc_sentiment 5日变化量的 Z-Score > 2.5 → 鸽派突变 → +1.0\n"
        "  - 正确示例: dgs2 连续5日急剧下行(短端利率暴跌 = 降息预期骤升) + t10y2y 急剧变陡\n"
        "    (Bull Steepening: 短端下行快于长端 = 美联储即将降息) → +1.0\n"
        "  - 错误示例: 'fomc_sentiment > 0.5 → 看多' (这是连续因子, 会被废弃!)\n"
        "  - 错误示例: 't10y2y < 0 → 看多' (倒挂是持续状态, 不是脉冲事件!)\n"
        "\n"
        "关键场景:\n"
        "  - 2019年1月: 鲍威尔突然转鸽, dgs2 从2.7%暴跌至2.1%, t10y2y急剧变陡\n"
        "  - 2022年3月: 超预期鹰派加息, fomc_sentiment 急降, dgs2 飙升\n"
        "\n"
        "注意: 禁止引用 dgs10 的水平值(那是 CoreAnchor 数据)"
    ),
    'microstructure': (
        "方向B: 恐慌极值与衰竭反转 (Panic Exhaustion Reversal)\n"
        "目标: 在流动性危机的恐慌极值处捕捉衰竭信号, 等待恐慌见顶回落后抄底\n"
        "\n"
        "典型数据:\n"
        "  - vixcls: VIX 波动率指数\n"
        "  - nfci/stlfsi4: 金融压力指数\n"
        "  - TLT ETF 成交量 (volume 列在 market_data_tlt 中)\n"
        "  - bamlh0a0hym2 的变化率 (注意: 禁止使用水平值, 那是 CoreAnchor 数据)\n"
        "\n"
        "核心逻辑 (必须遵守二阶导数铁律!):\n"
        "  - 绝对禁止写出 'VIX > 40 → 买入' 这种接飞刀逻辑!\n"
        "    高 VIX 期间直接买入会死于主跌浪, 2022年全年 VIX > 20 但美债暴跌\n"
        "  - 正确逻辑是 '极值 + 衰竭':\n"
        "    条件1: VIX 或金融压力指数处于极端高位 (252日 Z-Score > 2.5)\n"
        "    条件2: 当天的值 < 过去3天的均值 (恐慌开始衰竭, VIX.diff() < 0)\n"
        "    两个条件同时满足 → 输出 +1.0 (恐慌见顶回落, 美债将反弹)\n"
        "  - 正确示例: VIX Z-Score > 2.5 AND VIX < VIX.rolling(3).mean() → +1.0\n"
        "  - 错误示例: 'VIX > 30 → +1.0' (VIX 在30以上可能持续数周, 这是接飞刀!)\n"
        "  - 错误示例: '信用利差飙升 → +1.0' (飙升过程中买入 = 接飞刀!)\n"
        "\n"
        "关键场景:\n"
        "  - 2020年3月: VIX飙至82后开始回落(3月下旬), 此时才是真正的买点\n"
        "  - 2022年6-10月: VIX持续在25-35之间, 但美债继续暴跌 → 此时绝对不能买入!\n"
        "\n"
        "注意: 信号必须是极短期脉冲(触发后1-5天), 不可持续持仓\n"
        "      禁止直接使用 bamlh0a0hym2 的水平值(那是 CoreAnchor 数据)"
    ),
    'volatility': (
        "方向C: 波动率极值与拥挤反转 (Volatility Crowding Reversal)\n"
        "目标: 监控跨资产波动率的极端狂飙, 在对冲盘极度拥挤且开始瓦解时捕捉反转\n"
        "\n"
        "典型数据:\n"
        "  - vixcls: VIX 波动率指数\n"
        "  - gvzcls: 黄金波动率指数 (跨资产波动率关联)\n"
        "  - t10y2y/t10y3m: 收益率曲线利差\n"
        "  - usepuindxd: 经济政策不确定性指数\n"
        "  - jlnum1m/jlnum3m: 跳跃风险指数\n"
        "\n"
        "核心逻辑 (必须遵守二阶导数铁律!):\n"
        "  - 绝对禁止 '波动率极端 → 直接买入' 的逻辑!\n"
        "    波动率极端飙升时美债可能还在暴跌(2022年加息周期), 必须等衰竭信号\n"
        "  - 正确逻辑是 '极值 + 衰竭 + 跨资产确认':\n"
        "    条件1: 波动率处于极端高位 (252日 Z-Score > 2.5)\n"
        "    条件2: 波动率开始回落 (diff() < 0 或 < 3日均值)\n"
        "    条件3(可选): 跨资产确认 (如 VIX回落 + GVZCLS回落 = 全面恐慌消退)\n"
        "    三个条件同时满足 → 输出 +1.0\n"
        "  - 正确示例: VIX Z-Score > 2.5 AND VIX.diff() < 0 AND GVZCLS.diff() < 0 → +1.0\n"
        "  - 错误示例: 'VIX Z-Score > 2.5 → +1.0' (VIX 可能继续飙升! 接飞刀!)\n"
        "\n"
        "关键场景:\n"
        "  - 2020年3月下旬: VIX从82开始回落 + GVZCLS同步回落 → 完美反转信号\n"
        "  - 2022年6月: VIX飙至35但未回落, 美债继续暴跌 → 此时不是买点!\n"
        "\n"
        "注意: MOVE 指数(债市VIX)不在 FRED 上, 用 VIX + 信用利差波动率替代\n"
        "      禁止直接使用 dgs10 的水平值(那是 CoreAnchor 数据)"
    ),
}

METHOD_DESCRIPTIONS = {
    'unstructured': (
        "方法A: 非结构化数据转化 (NLP Sentiment)\n"
        "目标: 将 FOMC 声明等央行文本的情绪得分转化为可交易脉冲信号\n"
        "\n"
        "可用数据(已入库):\n"
        "  - fomc_sentiment: FOMC声明鹰鸽情绪得分, 范围[-1.0, 1.0]\n"
        "    1.0=极度鸽派(看多美债), -1.0=极度鹰派(看空美债)\n"
        "    基于 LLM 对 FOMC 声明的文本分析, T+1生效(防前瞻偏差)\n"
        "    非会议日前向填充, 覆盖2007-02至今, 日频\n"
        "    每年约8次FOMC会议\n"
        "\n"
        "边际变化铁律:\n"
        "  - 绝对禁止直接输出 fomc_sentiment 的绝对值!\n"
        "  - 必须使用 .diff() 或滚动变化量来捕捉预期突变\n"
        "  - 只有在预期发生跳跃/反转的瞬间才触发信号\n"
        "\n"
        "正确示例:\n"
        "  'fomc_sentiment.diff() 的 252日 Z-Score > 2.5 → 鸽派突变脉冲 +1.0'\n"
        "  'fomc_sentiment 5日变化量 > 2.5σ 且从负转正 → 鹰转鸽反转 +1.0'\n"
        "\n"
        "错误示例 (会被直接废弃!):\n"
        "  'fomc_sentiment > 0.5 → +1.0' (连续因子, 非脉冲)\n"
        "  'fomc_sentiment < -0.5 → -1.0' (连续因子, 非脉冲)"
    ),
    'options': (
        "方法B: 波动率微观结构 (Volatility Microstructure)\n"
        "目标: 通过波动率衍生数据捕捉债市的极端对冲或恐慌行为\n"
        "\n"
        "可用代理变量(已入库):\n"
        "  - vixcls: CBOE VIX 波动率指数, 日频, 2007~至今\n"
        "  - gvzcls: CBOE黄金ETF隐含波动率指数, 日频, 2008~至今\n"
        "  - t10y2y: 10年-2年国债利差, 日频, 2007~至今\n"
        "  - usepuindxd: 经济政策不确定性指数, 日频, 2007~至今\n"
        "  - jlnum1m/jlnum3m: 跳跃风险指数, 月频, 2007~至今\n"
        "\n"
        "二阶导数铁律:\n"
        "  - 绝对禁止 'VIX > X → 买入' 的逻辑! 必须等衰竭信号\n"
        "  - 正确: VIX极端 + VIX开始回落 → 买入\n"
        "  - 错误: VIX极端 → 直接买入 (接飞刀!)\n"
        "\n"
        "正确示例:\n"
        "  'VIX 252日 Z-Score > 2.5 AND VIX < VIX.rolling(3).mean() → 恐慌衰竭 +1.0'\n"
        "  'VIX-GVZCLS 差值 Z-Score > 2.5 AND 差值开始回落 → 跨资产恐慌消退 +1.0'\n"
        "\n"
        "错误示例 (会被直接废弃!):\n"
        "  'VIX > 30 → +1.0' (接飞刀!)\n"
        "  'VIX Z-Score > 2.5 → +1.0' (没有衰竭条件!)"
    ),
    'nonlinear': (
        "方法C: 非线性特征交叉\n"
        "将多个单维度数据交叉成高维触发因子\n"
        "\n"
        "二阶导数铁律同样适用:\n"
        "  - 交叉条件中如果包含波动率/恐慌指标, 必须加入衰竭条件\n"
        "  - 'VIX极端 + 信用利差飙升' = 接飞刀 (两个都在恶化!)\n"
        "  - 'VIX极端且开始回落 + 信用利差飙升且开始回落' = 正确的衰竭反转\n"
        "\n"
        "正确示例:\n"
        "  'VIX Z-Score > 2.5 AND VIX.diff() < 0 AND nfci Z-Score > 2.0 AND nfci.diff() < 0\n"
        "   → 多重恐慌指标同步衰竭 → +1.0'\n"
        "\n"
        "错误示例 (会被直接废弃!):\n"
        "  'VIX > 30 AND 信用利差 > 400 → +1.0' (两个指标都在恶化 = 接飞刀!)\n"
        "\n"
        "注意: 交叉条件必须基于 FICC 经济学逻辑, 禁止无意义的排列组合\n"
        "      禁止直接使用 CoreAnchor 列 (dfii10, dgs10, bamlh0a0hym2)"
    ),
}

SYSTEM_PROMPT_TEMPLATE = """你是一位顶级 FICC 量化因子研究员, 专门为美债(TLT)宏观趋势策略挖掘卫星因子。

## 绝对铁律 (违反任何一条, 因子将被直接永久废弃!)

1. 因子必须实现 BaseFactor 接口, 包含 calculate_signal(data: pd.DataFrame) -> pd.Series 方法
2. 信号输出范围必须是 [-1.0, 1.0], 正值看多美债(TLT), 负值看空美债
3. 禁止在因子内部引用以下 CoreAnchor 数据: dfii10, dgs10, bamlh0a0hym2
   (这些已被 V13 底座使用, 卫星因子禁止重复引用)
4. 因子逻辑必须纯粹, 只看自己领域的数据, 不做跨域交叉过滤
5. 所有阈值和参数必须有经济学含义, 禁止无意义的魔法数字

## 三大核心铁律 (从 SPY 挖掘失败中总结的血泪教训!)

### 铁律1: 零值休眠 (Sniper Pulse)
卫星因子必须是"狙击手"级别的脉冲信号! 常态下, 信号必须返回 0.0。
只在极端事件发生的当天及随后极短几天内输出非零值 (+1.0 或 -1.0)。
目标 Trigger Rate 必须控制在 5% 到 15% 之间!
如果你的代码每天都输出非零值, 将被直接永久废弃!
正确: signal[zscore > 2.5] = +1.0; signal[else] = 0.0
错误: signal = some_continuous_value  (连续因子 = 废弃!)

### 铁律2: 二阶导数 (Anti-Catch-Falling-Knife)
绝对禁止写出 "如果 VIX > 40 则买入" 这种愚蠢逻辑!
高 VIX 期间直接买入会死于主跌浪 (2022年全年 VIX > 20 但美债暴跌 -30%)。
正确的抄底逻辑必须是 "极值 + 衰竭":
  - 条件1: 指标处于极端高位 (Z-Score > 2.5)
  - 条件2: 指标开始回落 (diff() < 0 或 < 3日均值)
  - 两个条件同时满足才输出信号!
正确: VIX Z-Score > 2.5 AND VIX < VIX.rolling(3).mean() → +1.0
错误: VIX Z-Score > 2.5 → +1.0  (没有衰竭条件 = 接飞刀!)

### 铁律3: 边际变化 (Marginal Change Only)
对于 FOMC 情绪得分等低频阶梯状数据, 绝对禁止直接输出其绝对值!
必须使用 .diff() 或计算其动量变化。只有在预期发生改变的瞬间才触发信号。
正确: fomc_sentiment.diff() 的 Z-Score > 2.5 → 脉冲
错误: fomc_sentiment > 0.5 → 信号  (这是连续因子, 非脉冲!)
同样: 收益率曲线的"动量变化"比"绝对水位"更有预测力。
不要关注是否倒挂, 而要关注"短端利率是否剧烈下行导致曲线突然变陡 (Bull Steepening)"。

## 当前挖掘方向

{direction_desc}

## 当前挖掘方法

{method_desc}

## 可用数据清单 (高价值字段已标注 *)

{data_inventory}

## 已有因子 (禁止重复)

{existing_factors}

## 输出要求

请生成一个完整的 Python 类, 格式如下:

```python
import numpy as np
import pandas as pd

class <ClassName>Factor:
    \"\"\"因子名称 (挖掘方向/方法)

    逻辑: [一句话描述因子的经济学逻辑, 必须说明为何是脉冲而非连续]
    数据: [使用的数据字段]
    触发: [Z-Score > 2.5 的具体条件 + 衰竭/边际变化条件]
    输出: [信号含义, 必须是脉冲型]
    \"\"\"

    def __init__(self, ...):
        self.name = '<factor_name>'

    def calculate_signal(self, data: pd.DataFrame) -> pd.Series:
        # 实现因子计算逻辑
        # 必须处理数据缺失的情况 (返回 0.0)
        # 必须遵守三大铁律: 零值休眠 + 二阶导数 + 边际变化
        # 非触发日信号严格为 0.0
        signal = pd.Series(0.0, index=data.index)
        # ... 你的脉冲触发逻辑 ...
        signal.name = self.name
        return signal

    def __repr__(self):
        return f"{{self.__class__.__name__}}(...)"
```

注意:
- 类名使用 PascalCase, 以 Factor 结尾, 用有意义的英文名替换 <ClassName>
- self.name 使用 snake_case, 用有意义的英文名替换 <factor_name>
- calculate_signal 必须处理 data 中缺少所需列的情况 (返回全 0 Series)
- 信号必须是狙击手级脉冲: Z-Score > 2.5 + 衰竭条件 时为 +1.0/-1.0, 其余时间为 0.0
- 禁止使用未来数据 (no look-ahead bias)
- 美债(TLT)是正向 Carry 资产, 看多=+1.0, 看空=-1.0
- 初始 signal 必须是 pd.Series(0.0, index=data.index), 然后只在触发条件满足时赋值
"""

USER_PROMPT_TEMPLATE = """请基于以下约束生成一个美债(TLT)卫星因子:

挖掘方向: {direction}
挖掘方法: {method}

三大铁律 (必须全部遵守):
1. 零值休眠: 常态信号=0.0, 只在极端事件触发时输出+1.0/-1.0, 目标Trigger Rate 5%-15%
2. 二阶导数: 禁止"指标极端→直接买入", 必须等"指标极端+开始回落"才触发
3. 边际变化: 禁止使用低频数据的绝对值, 必须用.diff()捕捉变化瞬间

其他要求:
1. 因子必须有清晰的 FICC 经济学逻辑
2. 只使用上面数据清单中已有的字段
3. 信号方向: 看多美债(TLT)为正, 看空美债为负
4. 代码必须完整可运行, 包含所有 import
5. 禁止引用 CoreAnchor 数据 (dfii10, dgs10, bamlh0a0hym2)
6. 初始 signal = pd.Series(0.0, index=data.index), 只在触发时赋值

请直接输出 Python 代码, 不要解释。"""
