#!/usr/bin/env python3
"""
GoldMacroFramework - 全自动宏观趋势框架 (底座 + 卫星因子版)

数据流:
  Raw Data → Core Anchor → DynamicSynthesizer → Execution → Wide DataFrame
     │           │              │                    │
     │           ├ vote_tips    ├ credit_panic_sig   ├ trend_intact
     │           ├ vote_dxy     ├ credit_panic_ic    ├ trend_break
     │           ├ vote_walcl   ├ credit_panic_wt    ├ score_regime
     │           └ core_signal  ├ sge_premium_sig    └ target_exposure
     │                          ├ sge_premium_ic
     │                          ├ sge_premium_wt
     │                          └ total_score
"""

import os
import sys
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from isolated_macro.core.valuation_model import (
    load_market_data,
    load_fred_raw_series,
    load_macro_factor,
)
from isolated_macro.core.framework.core_anchor import CoreMacroAnchor
from isolated_macro.core.framework.synthesizer import (
    CreditPanicFactor,
    SgePremiumFactor,
    DynamicSynthesizer,
)
from isolated_macro.core.framework.execution import AdaptiveExecution


class GoldMacroFramework:
    """
    全自动宏观趋势框架 (底座 + 卫星因子自动合成版)

    组装流程:
      1. 加载原始数据 (价格 + 宏观因子)
      2. Core Anchor 计算 SMA 投票底座信号
      3. DynamicSynthesizer 注册内置卫星因子, 计算 Rolling IC, 动态赋权
      4. Execution 自适应执行, 输出 target_exposure (core + 内置卫星)
      5. [新增] 自动加载生产池AI卫星因子 → MLSynthesizer 合成 final_exposure
      6. 返回包含所有中间过程的宽表

    卫星因子自动生效逻辑:
      - 从 gold_ai_factor_registry 加载 status='accepted' 的因子
      - 通过 MLSynthesizer 合成 sat_composite
      - final_exposure = clip(target_exposure + sat_composite, 0.0, 1.0)
      - 若无生产池因子, final_exposure = target_exposure (退化为纯底座)
    """

    def __init__(self, sma_window: int = 60,
                 ic_window: int = 60, ic_threshold: float = 0.05,
                 max_weight: float = 0.5,
                 sma_standard: int = 50, sma_slow: int = 200,
                 enable_satellite: bool = True):
        self.core_anchor = CoreMacroAnchor(sma_window=sma_window)

        self.synthesizer = DynamicSynthesizer(
            ic_window=ic_window,
            ic_threshold=ic_threshold,
            max_weight_per_factor=max_weight,
        )
        self.synthesizer.register(CreditPanicFactor())
        self.synthesizer.register(SgePremiumFactor())

        self.execution = AdaptiveExecution(
            sma_standard=sma_standard,
            sma_slow=sma_slow,
        )
        self.enable_satellite = enable_satellite

        print("[GoldMacroFramework] Initialized (Core + Satellites)")
        print(f"  Core: {self.core_anchor}")
        print(f"  Synthesizer: {self.synthesizer}")
        print(f"  Execution: {self.execution}")
        print(f"  AI Satellite: {'AUTO-ENABLED' if enable_satellite else 'DISABLED'}")

    def _load_data(self, start_date=None, end_date=None):
        df_gold = load_market_data('GCUSD', start_date, end_date)
        df_dfii10 = load_fred_raw_series('DFII10', start_date, end_date)
        df_dtwexbgs = load_fred_raw_series('DTWEXBGS', start_date, end_date)
        df_walcl = load_macro_factor('walcl', start_date, end_date)
        df_baa10ym = load_fred_raw_series('BAA10YM', start_date, end_date)
        df_sge = load_macro_factor('sge_premium', start_date, end_date)

        df = df_gold.copy()
        if not df_dfii10.empty:
            df = df.join(df_dfii10[['dfii10']], how='left')
        if not df_dtwexbgs.empty:
            df = df.join(df_dtwexbgs[['dtwexbgs']], how='left')
        if not df_walcl.empty:
            df = df.join(df_walcl[['walcl']], how='left')
        if not df_baa10ym.empty:
            df = df.join(df_baa10ym[['baa10ym']], how='left')
        if not df_sge.empty:
            df = df.join(df_sge[['sge_premium']], how='left')

        df = df.sort_index()
        df = df.ffill()
        df = df.dropna(subset=['market_price', 'dfii10', 'dtwexbgs'])

        return df

    def run(self, start_date=None, end_date=None) -> pd.DataFrame:
        """
        运行完整流水线, 返回宽表
        """
        print("\n[Step 1/4] Loading raw data...")
        df = self._load_data(start_date, end_date)
        if df.empty:
            print("[ERROR] No data loaded!")
            return pd.DataFrame()
        print(f"  Loaded {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"  Columns: {list(df.columns)}")

        print("[Step 2/4] Computing core anchor signals (SMA Voting)...")
        df = self.core_anchor.calculate(df)
        print(f"  core_signal: mean={df['core_signal'].mean():.4f}, "
              f"std={df['core_signal'].std():.4f}")
        vote_counts = {
            'all_bull': int((df['core_signal'].round(2) == 1.0).sum()),
            'two_bull': int((df['core_signal'].round(2) == 0.33).sum()),
            'two_bear': int((df['core_signal'].round(2) == -0.33).sum()),
            'all_bear': int((df['core_signal'].round(2) == -1.0).sum()),
        }
        print(f"  Vote distribution: {vote_counts}")

        print("[Step 3/4] Dynamic synthesis (Rolling IC weighting)...")
        df = self.synthesizer.calculate(df, df['core_signal'])
        print(f"  total_score: mean={df['total_score'].mean():.4f}, "
              f"std={df['total_score'].std():.4f}")
        for sat in self.synthesizer.satellites:
            sig_col = f'{sat.name}_signal'
            wt_col = f'{sat.name}_weight'
            ic_col = f'{sat.name}_ic'
            if sig_col in df.columns:
                sig = df[sig_col]
                print(f"  {sat.name}: trigger_rate={sig.mean():.2%}, "
                      f"avg_weight={df[wt_col].mean():.4f}, "
                      f"avg_ic={df[ic_col].mean():.4f}")

        print("[Step 4/5] Adaptive execution...")
        df = self.execution.calculate(df, df['total_score'])

        n = len(df)
        exp = df['target_exposure']
        regime_vc = df['score_regime'].value_counts()
        print(f"\n[Framework Result] {n} rows")
        print(f"  Score regimes: {dict(regime_vc)}")
        print(f"  Exposure: mean={exp.mean():.4f}, "
              f"full={int((exp >= 0.99).sum())}d, "
              f"partial={int(((exp > 0) & (exp < 0.99)).sum())}d, "
              f"zero={int((exp == 0.0).sum())}d")

        if self.enable_satellite:
            df = self._apply_satellite_factors(df)
        else:
            df['final_exposure'] = df['target_exposure']
            df['sat_composite'] = 0.0
            df['n_active_factors'] = 0

        return df

    def _apply_satellite_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        from gold_ai_fund.db.schema import get_accepted_factors
        from gold_ai_fund.agents.factor_miner import load_factor_instance
        from gold_ai_fund.agents.ml_synthesizer import MLSynthesizer

        accepted = get_accepted_factors()

        if not accepted:
            print("\n[Step 5/5] No accepted satellite factors in production pool.")
            print("  -> final_exposure = target_exposure (pure core)")
            df['final_exposure'] = df['target_exposure']
            df['sat_composite'] = 0.0
            df['n_active_factors'] = 0
            return df

        print(f"\n[Step 5/5] Loading {len(accepted)} AI satellite factors from production pool...")

        from gold_ai_fund.agents.gatekeeper import Gatekeeper
        df_enriched = Gatekeeper._enrich_with_data_lake(df, str(df.index[0].date()), str(df.index[-1].date()))

        factor_signals = {}
        factor_directions = {}
        for af in accepted:
            try:
                instance = load_factor_instance(af['factor_id'], af['source_file'])
                sig = instance.calculate_signal(df_enriched)
                factor_signals[af['factor_id']] = sig
                factor_directions[af['factor_id']] = af['mining_direction']
            except Exception as e:
                print(f"  [WARN] Failed to load factor {af['factor_id']}: {e}")

        if not factor_signals:
            print("  [WARN] No usable factor signals, falling back to pure core")
            df['final_exposure'] = df['target_exposure']
            df['sat_composite'] = 0.0
            df['n_active_factors'] = 0
            return df

        print(f"  Loaded {len(factor_signals)} factor signals: {list(factor_signals.keys())}")

        synthesizer = MLSynthesizer(model_type='zscore_pulse')

        result_df = synthesizer.synthesize(
            df_enriched, df['core_signal'], factor_signals,
            factor_directions=factor_directions,
        )

        if not result_df.empty and 'total_score' in result_df.columns:
            df['sat_composite'] = result_df['sat_composite'].reindex(df.index).fillna(0.0)
            df['final_exposure'] = (df['target_exposure'] + df['sat_composite']).clip(0.0, 1.0)
            df['n_active_factors'] = result_df['n_active_factors'].reindex(df.index).fillna(0).astype(int)
        else:
            df['sat_composite'] = 0.0
            df['final_exposure'] = df['target_exposure']
            df['n_active_factors'] = 0

        final_exp = df['final_exposure']
        print(f"  Satellite composite: mean={df['sat_composite'].mean():.4f}, "
              f"std={df['sat_composite'].std():.4f}")
        print(f"  Final Exposure: mean={final_exp.mean():.4f}, "
              f"full={int((final_exp >= 0.99).sum())}d, "
              f"partial={int(((final_exp > 0) & (final_exp < 0.99)).sum())}d, "
              f"zero={int((final_exp == 0.0).sum())}d")

        return df
