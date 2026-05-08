#!/usr/bin/env python3
"""
SPY 宏观框架 - 全自动宏观趋势框架 (底座 + 趋势霸权版)

数据流:
  Raw Data → Core Anchor → SPY Trend Sovereignty → target_exposure
     │                  │
     ├ vote_growth      ├ sma_50 / sma_200 (trend filter)
     ├ vote_employment  ├ upward_exemption (防踏空铁律)
     ├ vote_liquidity   ├ downward_circuit (主跌浪规避)
     └ spy_core_signal  └ 线性映射 + 豁免覆写

数据复权铁律:
  ⚠️ SPY 的所有价格计算必须且只能使用 adjClose (复权收盘价)
  以应对美股长期的分红除息
"""

import os
import sys
import inspect
import numpy as np
import pandas as pd
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from spy_ai_fund.core.framework.spy_core_anchor import SpyCoreAnchor
from spy_ai_fund.core.framework.spy_execution import SpyExecution


class SpyMacroFramework:
    """
    SPY 全自动宏观趋势框架 (底座 + 趋势霸权 + 卫星因子自动合成版)

    组装流程:
      1. 加载原始数据 (SPY adjClose 价格 + FRED 宏观因子)
      2. Core Anchor 计算 SMA 投票底座信号
      3. SpyExecution 应用 SPY 专属双向趋势霸权，输出 target_exposure (core only)
      4. [新增] 自动加载生产池卫星因子 → MLSynthesizer 合成 final_exposure
      5. 返回包含所有中间过程的宽表

    卫星因子自动生效逻辑:
      - 从 spy_ai_factor_registry 加载 status='accepted' 的因子
      - 通过 MLSynthesizer 合成 sat_composite
      - 应用 SPY 非对称干预风控 (主升浪禁止做空 + RSI<30 抄底共振)
      - final_exposure = clip(core_exposure + sat_composite, 0.0, 1.0)
      - 若无生产池因子, final_exposure = target_exposure (退化为纯底座)
    """

    def __init__(self, sma_window: int = 60,
                 sma_standard: int = 50, sma_slow: int = 200,
                 rsi_window: int = 14, rsi_oversold: float = 30.0,
                 enable_satellite: bool = True):
        self.core_anchor = SpyCoreAnchor(sma_window=sma_window)
        self.execution = SpyExecution(
            sma_standard=sma_standard,
            sma_slow=sma_slow,
            rsi_window=rsi_window,
            rsi_oversold=rsi_oversold,
        )
        self.enable_satellite = enable_satellite

        print("[SpyMacroFramework] Initialized (Core + LongBull/MeanReversion Execution)")
        print(f"  Core: {self.core_anchor}")
        print(f"  Execution: {self.execution}")
        print(f"  Satellite: {'AUTO-ENABLED' if enable_satellite else 'DISABLED'}")

    def _load_data(self, start_date=None, end_date=None):
        from database.db_manager import get_raw_data_by_source

        symbol_spy = 'SPY'
        df_spy = load_market_data_spy(symbol_spy, start_date, end_date)

        if df_spy.empty:
            raise ValueError("SPY 行情数据为空，请先运行 sync_spy_data.py")

        fred_series = {
            'INDPRO': 'indpro',
            'ICSA': 'icsa',
            'WALCL': 'walcl',
            'WTREGEN': 'wtregen',
            'RRPONTSYD': 'rrpontsyd',
            'DTB3': 'dtb3',
        }

        df = df_spy.copy()

        for series_id, col_name in fred_series.items():
            df_fred = load_fred_series(series_id, start_date, end_date)
            if df_fred.empty:
                print(f"  [WARN] {series_id} factor_data 为空，尝试从 raw_data 解析...")
                df_fred = load_fred_from_raw(series_id, col_name, start_date)

            if not df_fred.empty:
                all_days = pd.date_range(
                    df_fred.index.min(), df_fred.index.max(), freq='D'
                )
                df_fred = df_fred.reindex(all_days).ffill()
                df_fred = df_fred.reindex(df.index).ffill()
                df[col_name] = df_fred[col_name]
            else:
                print(f"  [ERROR] {series_id} 数据完全不可用!")

        df = df.sort_index()
        df = df.dropna(subset=['market_price'])

        return df

    def _scan_factor_dependencies(self, accepted_factors: list) -> set:
        import ast
        required = set()
        for af in accepted_factors:
            try:
                from spy_ai_fund.agents.factor_miner import load_factor_instance
                instance = load_factor_instance(af['factor_id'], af['source_file'])
                source = inspect.getsource(instance.calculate_signal)
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Subscript):
                        if isinstance(node.value, ast.Name) and node.value.id == 'data':
                            if isinstance(node.slice, ast.Constant):
                                required.add(node.slice.value.lower())
                            elif isinstance(node.slice, ast.Index):
                                if isinstance(node.slice.value, ast.Constant):
                                    required.add(node.slice.value.lower())
            except Exception:
                pass
        core_cols = {'market_price', 'spy_core_signal', 'target_exposure', 'rsi_14',
                     'sma_50', 'sma_200', 'vote_growth', 'vote_employment', 'vote_liquidity',
                     'indpro', 'icsa', 'walcl', 'wtregen', 'rrpontsyd', 'dtb3', 'net_liquidity',
                     'upward_exemption', 'downward_circuit', 'trend_regime'}
        required = required - core_cols
        print(f"  Satellite dependencies: {sorted(required)}")
        return required

    def _enrich_targeted(self, df: pd.DataFrame, required_fields: set) -> pd.DataFrame:
        if not required_fields:
            return df

        existing_cols = set(c.lower() for c in df.columns)
        missing = required_fields - existing_cols
        if not missing:
            return df

        start_date = str(df.index[0].date())
        end_date = str(df.index[-1].date())

        for field in missing:
            series_id = field.upper()
            col_name = field.lower()

            df_fred = load_fred_series(series_id, start_date, end_date)
            if df_fred.empty:
                df_fred = load_fred_from_raw(series_id, col_name, start_date)

            if not df_fred.empty:
                if series_id in df_fred.columns:
                    df_fred = df_fred.rename(columns={series_id: col_name})
                if col_name in df_fred.columns:
                    all_days = pd.date_range(df_fred.index.min(), df_fred.index.max(), freq='D')
                    df_fred = df_fred.reindex(all_days).ffill()
                    df_fred = df_fred.reindex(df.index).ffill()
                    df[col_name] = df_fred[col_name]
                    print(f"  + {col_name}: loaded")
            else:
                print(f"  - {col_name}: not available")

        return df

    def run(self, start_date=None, end_date=None) -> pd.DataFrame:
        """
        运行完整流水线, 返回宽表
        """
        print("\n[Step 1/3] Loading raw data...")
        df = self._load_data(start_date, end_date)
        if df.empty:
            print("[ERROR] No data loaded!")
            return pd.DataFrame()
        print(f"  Loaded {len(df)} rows, {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"  Columns: {list(df.columns)}")

        macro_cols = ['indpro', 'icsa', 'walcl', 'wtregen', 'rrpontsyd']
        for col in macro_cols:
            if col in df.columns:
                non_null = df[col].notna().sum()
                print(f"  {col}: {non_null} non-null values")
            else:
                print(f"  [WARN] {col} 列缺失!")

        print("[Step 2/3] Computing core anchor signals (SMA Voting)...")
        df = self.core_anchor.calculate(df)
        print(f"  spy_core_signal: mean={df['spy_core_signal'].mean():.4f}, "
              f"std={df['spy_core_signal'].std():.4f}")
        vote_counts = {
            'all_bull': int((df['spy_core_signal'].round(2) == 1.0).sum()),
            'two_bull': int((df['spy_core_signal'].round(2) == 0.33).sum()),
            'two_bear': int((df['spy_core_signal'].round(2) == -0.33).sum()),
            'all_bear': int((df['spy_core_signal'].round(2) == -1.0).sum()),
        }
        print(f"  Vote distribution: {vote_counts}")

        print("[Step 3/4] Applying SPY Trend Sovereignty...")
        df = self.execution.calculate(df, df['spy_core_signal'])

        n = len(df)
        exp = df['target_exposure']
        regime_vc = df['trend_regime'].value_counts()
        print(f"\n[Framework Result] {n} rows")
        print(f"  Trend regimes: {dict(regime_vc)}")
        print(f"  Core Exposure: mean={exp.mean():.4f}, "
              f"full={int((exp >= 0.99).sum())}d, "
              f"partial={int(((exp > 0) & (exp < 0.99)).sum())}d, "
              f"zero={int((exp == 0.0).sum())}d")

        # Step 4: Satellite factor auto-synthesis
        if self.enable_satellite:
            df = self._apply_satellite_factors(df)
        else:
            df['final_exposure'] = df['target_exposure']
            df['sat_composite'] = 0.0
            df['n_active_factors'] = 0

        return df

    def _apply_satellite_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        自动加载生产池卫星因子并合成 final_exposure

        流程:
          1. 从 spy_ai_factor_registry 加载 accepted 因子
          2. 丰富数据湖 (Gatekeeper._enrich_with_data_lake)
          3. MLSynthesizer.synthesize() 合成 sat_composite
          4. final_exposure = clip(core_exposure + sat_composite, 0.0, 1.0)
          5. 若无生产池因子, final_exposure = target_exposure (退化为纯底座)
        """
        from spy_ai_fund.db.schema import get_accepted_factors
        from spy_ai_fund.agents.factor_miner import load_factor_instance
        from spy_ai_fund.agents.ml_synthesizer import MLSynthesizer

        accepted = get_accepted_factors()

        if not accepted:
            print("\n[Step 4/4] No accepted satellite factors in production pool.")
            print("  -> final_exposure = target_exposure (pure core)")
            df['final_exposure'] = df['target_exposure']
            df['sat_composite'] = 0.0
            df['n_active_factors'] = 0
            return df

        print(f"\n[Step 4/4] Loading {len(accepted)} satellite factors from production pool...")

        required_fields = self._scan_factor_dependencies(accepted)
        df_enriched = self._enrich_targeted(df, required_fields)

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

        core_exposure = df['target_exposure']
        rsi_14 = df['rsi_14'] if 'rsi_14' in df.columns else None

        result_df = synthesizer.synthesize(
            df_enriched, df['spy_core_signal'], factor_signals,
            factor_directions=factor_directions,
            core_exposure=core_exposure,
            rsi_14=rsi_14,
        )

        if not result_df.empty and 'final_exposure' in result_df.columns:
            df['sat_composite'] = result_df['sat_composite'].reindex(df.index).fillna(0.0)
            df['final_exposure'] = result_df['final_exposure'].reindex(df.index)
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


def load_market_data_spy(symbol, start_date=None, end_date=None):
    """
    加载 SPY 日线行情 (adjClose 作为 market_price)

    优先级:
      1. market_data_spy 表 (含 adj_close，由 sync_spy_data.py 从 FMP 抓取)
      2. market_data_us_stock 表 (仅含 close，作为临时备用)

    ⚠️ 数据复权铁律: SPY 必须使用 adjClose 计算收益率和均线
    """
    import sqlite3
    from config import DB_PATH

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data_spy'")
    spy_table_exists = cursor.fetchone() is not None

    if spy_table_exists:
        query = "SELECT timestamp, adj_close FROM market_data_spy WHERE symbol = ?"
        params = [symbol]
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        query += " ORDER BY timestamp"

        df = pd.read_sql_query(query, conn, params=params)

        if not df.empty:
            conn.close()
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.normalize()
            df = df.drop_duplicates(subset=['timestamp']).set_index('timestamp').sort_index()
            df.rename(columns={'adj_close': 'market_price'}, inplace=True)
            print(f"  [SPY] 从 market_data_spy 加载 {len(df)} 行 (adjClose)")
            return df

    print("  [SPY] market_data_spy 无数据，尝试从 market_data_us_stock 加载 (close 作为临时替代)...")
    print("  [SPY] [WARNING] 强烈建议运行 python spy_ai_fund/sync_spy_data.py 获取 adjClose 数据!")

    query = "SELECT timestamp, close FROM market_data_us_stock WHERE symbol = ?"
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
    df.rename(columns={'close': 'market_price'}, inplace=True)
    print(f"  [SPY] 从 market_data_us_stock 加载 {len(df)} 行 (close, 非复权)")
    return df


def load_fred_series(series_id, start_date=None, end_date=None):
    """从 factor_data 表加载 FRED 序列"""
    import sqlite3
    from config import DB_PATH

    col_map = {
        'INDPRO': 'indpro',
        'ICSA': 'icsa',
        'WALCL': 'walcl',
        'WTREGEN': 'wtregen',
        'RRPONTSYD': 'rrpontsyd',
        'DTB3': 'dtb3',
    }

    col_name = col_map.get(series_id, series_id.lower())

    conn = sqlite3.connect(DB_PATH)
    query = "SELECT timestamp, factor_value FROM factor_data WHERE symbol = 'MACRO' AND factor_name = ?"
    params = [series_id]
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
    df.rename(columns={'factor_value': col_name}, inplace=True)
    return df


def load_fred_from_raw(series_id, col_name, start_date=None):
    """从 raw_data 表解析 FRED 序列 (备用通道)"""
    from database.db_manager import get_raw_data_by_source

    source_id = f'fred_{series_id}'
    records = get_raw_data_by_source(
        source_id,
        pd.Timestamp(start_date) if start_date else None
    )

    if not records:
        return pd.DataFrame()

    rows = []
    for rec in records:
        try:
            raw = rec.get('raw_content', rec) if isinstance(rec, dict) else {}
            if isinstance(raw, str):
                raw = json.loads(raw)
            val_str = raw.get('value', '.')
            if val_str and val_str != '.':
                rows.append({
                    'timestamp': pd.to_datetime(raw.get('date', rec.get('event_timestamp'))),
                    col_name: float(val_str)
                })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index('timestamp').sort_index()
    df = df[~df.index.duplicated(keep='last')]
    return df
