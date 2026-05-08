#!/usr/bin/env python3
"""
TLT 宏观框架 - Yield Shock Z-Score + Value (v13)

数据流:
  Raw Data → ffill → Core Anchor (Yield Shock Veto) → Bear Trap Only → target_exposure
     │                  │                                            │
     ├ DGS10            ├ is_tightening_shock (Z>1.5σ→紧缩0.0)        ├ bear_trap (强制清仓)
     ├ BAMLH0A0HYM2     ├ is_panic (恐慌→1.0)                         └ 无追涨代码
     ├ DFII10           ├ normal_carry (TIPS Z-Score)
     └ DTB3             └ tlt_core_signal

卫星因子自动生效逻辑:
  - 从 tlt_ai_factor_registry 加载 status='accepted' 的因子
  - 通过 MLSynthesizer 合成 sat_composite
  - final_exposure = clip(target_exposure + sat_composite, 0.0, 1.0)
  - 若无生产池因子, final_exposure = target_exposure (退化为纯底座)
"""

import os
import sys
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tlt_ai_fund.core.framework.tlt_core_anchor import TltCoreAnchor
from tlt_ai_fund.core.framework.tlt_execution import TltExecution


class TltMacroFramework:
    """
    TLT Yield Shock Z-Score + Value 框架 (v13)

    卫星因子自动生效逻辑:
      - 从 tlt_ai_factor_registry 加载 status='accepted' 的因子
      - 通过 MLSynthesizer 合成 sat_composite
      - final_exposure = clip(target_exposure + sat_composite, 0.0, 1.0)
      - 若无生产池因子, final_exposure = target_exposure (退化为纯底座)
    """

    def __init__(self, sma_standard: int = 50, sma_slow: int = 200,
                 enable_satellite: bool = True):
        self.core_anchor = TltCoreAnchor()
        self.execution = TltExecution(sma_standard=sma_standard, sma_slow=sma_slow)
        self.enable_satellite = enable_satellite

        print("[TltMacroFramework] Initialized (Yield Shock Z-Score + Value v13)")
        print(f"  Core: {self.core_anchor}")
        print(f"  Execution: {self.execution}")
        print(f"  AI Satellite: {'AUTO-ENABLED' if enable_satellite else 'DISABLED'}")

    def _load_data(self, start_date=None, end_date=None):
        from isolated_macro.core.valuation_model import load_fred_raw_series

        df_tlt = load_market_data_tlt('TLT', start_date, end_date)
        if df_tlt.empty:
            raise ValueError("TLT 行情数据为空，请先运行 sync_tlt_data.py")

        df_baml = load_fred_raw_series('BAMLH0A0HYM2', start_date, end_date)
        df_dfii10 = load_fred_raw_series('DFII10', start_date, end_date)
        df_dgs10 = load_fred_raw_series('DGS10', start_date, end_date)
        df_dtb3 = load_fred_raw_series('DTB3', start_date, end_date)

        if df_dgs10.empty:
            raise ValueError("DGS10 数据为空，请检查 FRED 数据抓取")

        df = df_tlt.copy()

        if not df_baml.empty:
            df_baml_r = df_baml.rename(columns={'bamlh0a0hym2': 'BAMLH0A0HYM2'})
            df = df.join(df_baml_r[['BAMLH0A0HYM2']], how='left')
        if not df_dfii10.empty:
            df_dfii10_r = df_dfii10.rename(columns={'dfii10': 'DFII10'})
            df = df.join(df_dfii10_r[['DFII10']], how='left')
        if not df_dgs10.empty:
            df_dgs10_r = df_dgs10.rename(columns={'dgs10': 'DGS10'})
            df = df.join(df_dgs10_r[['DGS10']], how='left')
        if not df_dtb3.empty:
            df = df.join(df_dtb3[['dtb3']], how='left')

        df = df.sort_index()

        fred_cols = ['BAMLH0A0HYM2', 'DFII10', 'DGS10', 'dtb3']
        for col in fred_cols:
            if col in df.columns:
                df[col] = df[col].ffill()

        df = df.dropna(subset=['market_price'])

        return df

    def run(self, start_date=None, end_date=None) -> pd.DataFrame:
        print("\n[Step 1/3] Loading raw data...")
        df = self._load_data(start_date, end_date)
        if df.empty:
            print("[ERROR] No data loaded!")
            return pd.DataFrame()
        print(f"  Loaded {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")

        print("[Step 2/3] Computing Yield Shock Z-Score + Value signals...")
        df = self.core_anchor.calculate(df)
        n = len(df)
        sig = df['tlt_core_signal']
        print(f"  tlt_core_signal: mean={sig.mean():.4f}, std={sig.std():.4f}")

        panic_days = int(df['is_panic'].sum())
        shock_days = int(df['is_tightening_shock'].sum())
        normal_days = n - panic_days - shock_days
        print(f"  Regime: Panic={panic_days}d, TighteningShock={shock_days}d, Normal={normal_days}d")

        if 'tips_zscore' in df.columns:
            valid_z = df['tips_zscore'][df['tips_zscore'] != 0]
            if len(valid_z) > 0:
                print(f"  TIPS Z-Score (valid): mean={valid_z.mean():.4f}, std={valid_z.std():.4f}")
        print(f"  normal_carry: mean={df['normal_carry'].mean():.4f}")

        print("[Step 3/4] Applying Bear Trap Only...")
        df = self.execution.calculate(df)

        exp = df['target_exposure']
        regime_vc = df['trend_regime'].value_counts()
        print(f"\n[Framework Result] {n} rows")
        print(f"  Trend regimes: {dict(regime_vc)}")
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
        from tlt_ai_fund.db.schema import get_accepted_factors
        from tlt_ai_fund.agents.factor_miner import load_factor_instance
        from tlt_ai_fund.agents.ml_synthesizer import MLSynthesizer

        accepted = get_accepted_factors()

        if not accepted:
            print("\n[Step 4/4] No accepted satellite factors in production pool.")
            print("  -> final_exposure = target_exposure (pure core)")
            df['final_exposure'] = df['target_exposure']
            df['sat_composite'] = 0.0
            df['n_active_factors'] = 0
            return df

        print(f"\n[Step 4/4] Loading {len(accepted)} AI satellite factors from production pool...")

        from tlt_ai_fund.agents.gatekeeper import Gatekeeper
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

        synthesizer = MLSynthesizer()

        core_col = 'target_exposure' if 'target_exposure' in df.columns else 'tlt_core_signal'
        if core_col not in df.columns:
            core_col = 'core_signal'

        result_df = synthesizer.synthesize(
            df_enriched, df[core_col], factor_signals,
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


def load_market_data_tlt(symbol, start_date=None, end_date=None):
    import sqlite3
    from config import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    query = "SELECT timestamp, adj_close, close FROM market_data_tlt WHERE symbol = ?"
    params = [symbol]
    if start_date:
        query += " AND timestamp >= ?"
        params.append(start_date)
    if end_date:
        query += " AND timestamp <= ?"
        params.append(end_date)
    query += " ORDER BY timestamp"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()
    df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
    df.rename(columns={'adj_close': 'market_price'}, inplace=True)
    df['adj_close'] = df['market_price']
    return df
