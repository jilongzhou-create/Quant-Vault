#!/usr/bin/env python3
"""
Macro Backtest Engine - TLT 敞口回测引擎 (v13 + Satellite)

专为"叙事驱动/目标敞口范式"设计的回测引擎。
支持连续敞口滑行 + 现金管理收益 (DTB3 无风险利率) + 卫星因子脉冲叠加。

核心铁律:
  1. shift(1): 昨天的目标敞口 × 今天的实际收益率，杜绝 Look-ahead bias
  2. ffill: 目标敞口通过 ffill() 优雅维持，避免每天买卖
  3. 交易成本: 可配置单边手续费 (default: 0.02%)
  4. 现金收益: 空仓时 (1.0 - position) × DTB3/252 按日计入收益
  5. 卫星合成: sat_intervention = clip(Sat_Total, -0.3, 0.3)
              final_exposure = clip(V13_Core + sat_intervention, 0.0, 1.0)
"""

import numpy as np
import pandas as pd


def synthesize_final_exposure(core_exposure: pd.Series,
                              sat_signals: dict = None,
                              sat_cap: float = 0.3) -> pd.Series:
    """
    V13 底座 + 卫星因子脉冲合成公式 (Long-Only 铁律)

    Args:
        core_exposure: V13 Core Anchor 敞口 [0.0, 1.0]
        sat_signals: dict of {factor_name: signal_series}, 每个信号 [-1.0, 1.0]
        sat_cap: 卫星因子最大干预权限 (default 0.3)

    Returns:
        final_target_exposure: 最终目标敞口 [0.0, 1.0]
    """
    if sat_signals is None or len(sat_signals) == 0:
        return core_exposure.copy()

    sat_total = pd.Series(0.0, index=core_exposure.index)
    for factor_name, signal in sat_signals.items():
        aligned = signal.reindex(core_exposure.index).fillna(0.0)
        sat_total = sat_total + aligned

    sat_intervention = np.clip(sat_total, -sat_cap, sat_cap)

    final_target_exposure = np.clip(core_exposure + sat_intervention, 0.0, 1.0)

    return final_target_exposure


class TltBacktestEngine:
    """
    TLT 连续敞口回测引擎 (含现金管理收益)

    输入:
      - df: DataFrame with columns [timestamp(index), market_price, target_exposure]
      - cost_rate: 单边交易成本率 (default 0.0002 = 0.02%)

    输出:
      - result: dict with performance metrics
      - df_result: DataFrame with daily PnL details
    """

    def __init__(self, cost_rate=0.0002, risk_free_rate=0.0):
        self.cost_rate = cost_rate
        self.risk_free_rate = risk_free_rate

    def run(self, df, rf_series=None):
        """
        运行回测

        Args:
            df: DataFrame, index=DatetimeIndex, 必须包含:
                - market_price: 每日收盘价 (adjClose)
                - target_exposure: 目标敞口 [0.0, 1.0] (纯多头)
            rf_series: 可选, pd.Series, 每日年化无风险利率(%),
                       如 DTB3 日频序列。缺失则 fallback 到 self.risk_free_rate

        Returns:
            (result_dict, df_detail)
        """
        df = df.copy()
        df = df.sort_index()

        df['market_return'] = df['market_price'].pct_change()

        df['position'] = df['target_exposure'].shift(1).ffill()
        df = df.dropna(subset=['position', 'market_return'])

        df['position_change'] = df['position'].diff().abs()
        df.loc[df.index[0], 'position_change'] = abs(df['position'].iloc[0])

        df['trade_cost'] = df['position_change'] * self.cost_rate

        if rf_series is not None:
            rf_aligned = rf_series.reindex(df.index).ffill().fillna(0.0)
            daily_rf = rf_aligned / 100.0 / 252
            daily_rf = daily_rf.clip(lower=0.0)
        else:
            daily_rf = self.risk_free_rate / 252

        cash_portion = 1.0 - df['position']
        df['strategy_return'] = (
            df['position'] * df['market_return']
            + cash_portion * daily_rf
            - df['trade_cost']
        )

        df['cum_market_return'] = (1 + df['market_return']).cumprod()
        df['cum_strategy_return'] = (1 + df['strategy_return']).cumprod()
        df['nav'] = df['cum_strategy_return']

        result = self._compute_metrics(df)

        return result, df

    def _compute_metrics(self, df):
        """计算绩效指标"""
        strategy_returns = df['strategy_return'].dropna()
        market_returns = df['market_return'].dropna()

        n_days = len(strategy_returns)
        if n_days < 2:
            return {}

        total_return = df['cum_strategy_return'].iloc[-1] - 1
        market_total_return = df['cum_market_return'].iloc[-1] - 1

        years = n_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        market_annualized = (1 + market_total_return) ** (1 / years) - 1 if years > 0 else 0

        annualized_vol = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return / annualized_vol) if annualized_vol > 0 else 0

        risk_free_rate = 0.0
        excess_returns = strategy_returns - risk_free_rate / 252
        sharpe_ratio_rf = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

        cum_max = df['cum_strategy_return'].cummax()
        drawdown = (df['cum_strategy_return'] - cum_max) / cum_max
        max_drawdown = drawdown.min()

        rolling_max = df['cum_strategy_return'].cummax()
        dd_duration = (df['cum_strategy_return'] < rolling_max).astype(int)
        dd_groups = (dd_duration.diff() != 0).cumsum()
        max_dd_duration = 0
        if dd_duration.sum() > 0:
            max_dd_duration = dd_duration.groupby(dd_groups).sum().max()

        win_rate = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0

        winning = strategy_returns[strategy_returns > 0]
        losing = strategy_returns[strategy_returns < 0]
        avg_win = winning.mean() if len(winning) > 0 else 0
        avg_loss = abs(losing.mean()) if len(losing) > 0 else 1e-9
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        total_turnover = df['position_change'].sum()

        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        skewness = strategy_returns.skew()
        kurtosis = strategy_returns.kurtosis()

        result = {
            'total_return': total_return,
            'market_total_return': market_total_return,
            'annualized_return': annualized_return,
            'market_annualized': market_annualized,
            'annualized_vol': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sharpe_ratio_rf': sharpe_ratio_rf,
            'max_drawdown': max_drawdown,
            'max_dd_duration': int(max_dd_duration),
            'calmar_ratio': calmar_ratio,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'total_turnover': total_turnover,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'n_days': n_days,
            'n_years': years,
            'start_date': str(df.index[0].date()),
            'end_date': str(df.index[-1].date()),
        }

        return result

    @staticmethod
    def print_report(result, title="TLT Backtest Report"):
        """打印格式化的绩效报告"""
        print("\n" + "=" * 78)
        print(f"  {title}")
        print("=" * 78)
        print(f"  Period:          {result['start_date']} ~ {result['end_date']}")
        print(f"  Duration:        {result['n_days']} days ({result['n_years']:.2f} years)")
        print("-" * 78)
        print(f"  Total Return:    {result['total_return']:>10.2%}    (Benchmark: {result['market_total_return']:>10.2%})")
        print(f"  Annual Return:   {result['annualized_return']:>10.2%}    (Benchmark: {result['market_annualized']:>10.2%})")
        print(f"  Annual Vol:      {result['annualized_vol']:>10.2%}")
        print("-" * 78)
        print(f"  Sharpe Ratio:    {result['sharpe_ratio']:>10.4f}")
        print(f"  Sharpe (rf=0):   {result['sharpe_ratio_rf']:>10.4f}")
        print(f"  Calmar Ratio:    {result['calmar_ratio']:>10.4f}")
        print("-" * 78)
        print(f"  Max Drawdown:    {result['max_drawdown']:>10.2%}")
        print(f"  Max DD Duration: {result['max_dd_duration']:>10d} days")
        print("-" * 78)
        print(f"  Win Rate:        {result['win_rate']:>10.2%}")
        print(f"  Profit/Loss:     {result['profit_loss_ratio']:>10.4f}")
        print(f"  Total Turnover:  {result['total_turnover']:>10.2f}")
        print("-" * 78)
        print(f"  Skewness:        {result['skewness']:>10.4f}")
        print(f"  Kurtosis:        {result['kurtosis']:>10.4f}")
        print("=" * 78)
