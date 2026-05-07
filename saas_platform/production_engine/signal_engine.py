"""
云端信号与净值引擎 (Cloud Signal & NAV Engine)

职责：每日定时任务的指挥中心
  1. 从 Supabase 读取行情 + 因子数据，按日期对齐拼装宽表 DataFrame
  2. 从 Supabase 读取活跃策略的 python_code，安全沙盒执行，获取 target_position
  3. 异常隔离：单策略报错不崩溃，记录日志跳过
  4. 结果回写：更新 target_position 到 saas_strategies，计算模拟净值 UPSERT 到 saas_equity_curves

严格约束：
  - 数据拼装：从 Supabase 读取，内存中按 timestamp 对齐
  - 安全沙盒：exec + 受控 globals/locals
  - 异常隔离：try-except 包裹每个策略
  - 结果回写：is_backtest=False 标记实盘净值
"""

import os
import sys
import logging
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from saas_platform.saas_config import get_config_summary

from saas_platform.database.supabase_client import (
    get_client,
    get_market_data,
    get_factor_data,
    get_live_strategies,
    update_strategy_position,
    update_daily_nav,
    get_equity_curve,
    upsert_equity_curve,
)

logger = logging.getLogger('saas_platform.production_engine.signal_engine')

SLIPPAGE_BPS = 5
COMMISSION_BPS = 10
INITIAL_NAV = 1000.0
MAX_LOOKBACK_DAYS = 400


class CloudSignalEngine:
    """
    云端信号与净值计算引擎

    核心流程：
      1. 拉取指定标的的行情宽表（OHLCV + 因子）
      2. 遍历所有活跃策略，沙盒执行 python_code
      3. 计算模拟净值曲线（含滑点/手续费）
      4. 回写 Supabase
    """

    def __init__(self):
        self._data_cache: dict[str, pd.DataFrame] = {}

    def _fetch_wide_table(self, symbol: str, days: int = MAX_LOOKBACK_DAYS) -> pd.DataFrame:
        """
        从 Supabase 拉取行情 + 因子数据，在内存中按 timestamp 对齐拼装成宽表

        实现思路：
          - 行情数据：从 saas_market_data 取最近 days 天的 OHLCV + 技术指标
          - 因子数据：分别取 symbol 本身和 MACRO 的因子，pivot 成宽表
          - 对齐方式：pd.merge_asof 向后对齐（direction='backward'），确保因子值不偷看未来
        """
        cache_key = f"{symbol}_{days}"
        if cache_key in self._data_cache:
            return self._data_cache[cache_key]

        logger.info(f"[宽表] 开始拼装 {symbol} 的行情+因子宽表 (days={days})...")

        market_records = get_market_data(symbol, limit=days * 2, order='timestamp.asc')
        if not market_records:
            logger.warning(f"[宽表] {symbol} 无行情数据（Supabase saas_market_data 中 symbol={symbol} 无记录）")
            return pd.DataFrame()

        df_market = pd.DataFrame(market_records)
        df_market['timestamp'] = pd.to_datetime(df_market['timestamp'])
        df_market = df_market.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"[宽表] {symbol} 行情原始记录: {len(df_market)} 条, 时间范围: {df_market['timestamp'].min()} ~ {df_market['timestamp'].max()}")

        if days < 99999:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            df_market = df_market[df_market['timestamp'] >= cutoff]

        if df_market.empty:
            logger.warning(f"[宽表] {symbol} 行情数据在 {days} 天内为空")
            return pd.DataFrame()

        # 2. 拉取因子数据（标的自身 + MACRO）
        all_factor_records = []

        for factor_symbol in [symbol, 'MACRO']:
            factor_limit = days * 20 if days < 99999 else 99999
            factor_records = get_factor_data(
                factor_symbol,
                limit=factor_limit,
                order='timestamp.asc',
            )
            if factor_records:
                logger.info(f"[宽表] {factor_symbol} 因子记录: {len(factor_records)} 条")
                all_factor_records.extend(factor_records)

        if all_factor_records:
            df_factor = pd.DataFrame(all_factor_records)
            df_factor['timestamp'] = pd.to_datetime(df_factor['timestamp'])
            df_factor['factor_value'] = pd.to_numeric(df_factor['factor_value'], errors='coerce')

            # 去重：同一 timestamp + factor_name 保留最新
            df_factor = df_factor.drop_duplicates(subset=['timestamp', 'factor_name'], keep='last')

            # pivot 成宽表
            pivot_df = df_factor.pivot_table(
                index='timestamp',
                columns='factor_name',
                values='factor_value',
                aggfunc='last',
            )
            pivot_df = pivot_df.sort_index().ffill()
            pivot_df = pivot_df.reset_index()

            # merge_asof 向后对齐：行情时间点取最近的因子值（不偷看未来）
            df_market = pd.merge_asof(
                df_market.sort_values('timestamp'),
                pivot_df.sort_values('timestamp'),
                on='timestamp',
                direction='backward',
            )

        # 前向填充 + 零填充
        df_market = df_market.ffill().fillna(0)
        df_market = df_market.set_index('timestamp')

        logger.info(f"[宽表] {symbol} 宽表拼装完成: {df_market.shape}")
        self._data_cache[cache_key] = df_market
        return df_market

    def _execute_strategy_code(self, python_code: str, df: pd.DataFrame, params_json: dict = None) -> Optional[float]:
        """
        安全沙盒执行策略代码，返回 target_position

        实现思路：
          - python_code 可能是纯 Python 代码（单策略）或 JSON 数组（组合策略）
          - 组合策略格式：[{"dir_id": ..., "name": ..., "code": ..., "params": {...}, "sharpe": ...}, ...]
          - 对每个子策略分别执行 generate_signals，获取各子策略的 target_position
          - 根据 weight_mode（equal/sharpe）加权汇总为最终仓位
          - 任何异常均被捕获，返回 None
        """
        import json as _json

        try:
            sub_strategies = _json.loads(python_code)
            if not isinstance(sub_strategies, list):
                sub_strategies = None
        except (ValueError, TypeError):
            sub_strategies = None

        if sub_strategies:
            return self._execute_portfolio_strategy(sub_strategies, df, params_json)
        else:
            return self._execute_single_strategy(python_code, df)

    def _execute_single_strategy(self, code: str, df: pd.DataFrame, params: dict = None) -> Optional[float]:
        """
        执行单个策略代码，返回 target_position
        """
        try:
            safe_globals = {
                '__builtins__': __builtins__,
                'pd': pd,
                'np': np,
            }
            safe_locals = {'df': df.copy()}

            exec(code, safe_globals, safe_locals)

            func = safe_locals.get('generate_signals')
            if func is None:
                logger.error("[沙盒] 策略代码中未找到 generate_signals 函数")
                return None

            if params:
                result = func(safe_locals['df'], params)
            else:
                result = func(safe_locals['df'])

            if result is None:
                logger.warning("[沙盒] generate_signals 返回 None")
                return None

            if isinstance(result, pd.Series):
                last_pos = result.iloc[-1]
                return float(last_pos)

            if isinstance(result, pd.DataFrame):
                if result.empty:
                    logger.warning("[沙盒] generate_signals 返回空 DataFrame")
                    return None
                if 'target_position' not in result.columns:
                    logger.error("[沙盒] generate_signals 返回的 df 中无 target_position 列")
                    return None
                last_pos = result['target_position'].iloc[-1]
                return float(last_pos)

            logger.error(f"[沙盒] generate_signals 返回了不支持的类型: {type(result)}")
            return None

        except Exception as e:
            logger.error(f"[沙盒] 单策略执行异常: {e}\n{traceback.format_exc()}")
            return None

    def _execute_portfolio_strategy(self, sub_strategies: list, df: pd.DataFrame, params_json: dict = None) -> Optional[float]:
        """
        执行组合策略：分别执行每个子策略，按权重汇总 target_position

        实现思路：
          - 遍历组合中的每个子策略
          - 对每个子策略调用 _execute_single_strategy
          - 根据 weight_mode 决定权重：
            - equal: 等权平均
            - sharpe: 按夏普率加权（需要 sharpe 字段）
          - 汇总所有子策略的 target_position 为最终仓位
          - 单个子策略报错不影响其他子策略
        """
        weight_mode = 'equal'
        if params_json and isinstance(params_json, dict):
            weight_mode = params_json.get('weight_mode', 'equal')

        positions = []
        sharpes = []

        for i, sub in enumerate(sub_strategies):
            sub_name = sub.get('name', f'sub_{i}')
            sub_code = sub.get('code', '')
            sub_params = sub.get('params', {})
            sub_sharpe = sub.get('sharpe') or 0

            if not sub_code:
                logger.warning(f"[组合] 子策略 {sub_name} 无代码，跳过")
                continue

            logger.info(f"[组合] 执行子策略 {i+1}/{len(sub_strategies)}: {sub_name}")

            pos = self._execute_single_strategy(sub_code, df, sub_params)

            if pos is not None:
                pos = max(-1.0, min(1.0, pos))
                positions.append(pos)
                sharpes.append(max(sub_sharpe, 0))
                logger.info(f"[组合] {sub_name} -> target_position = {pos:.4f}")
            else:
                logger.warning(f"[组合] {sub_name} 执行失败，跳过")

        if not positions:
            logger.error("[组合] 所有子策略均执行失败")
            return None

        if weight_mode == 'sharpe' and sum(sharpes) > 0:
            total_sharpe = sum(sharpes)
            weights = [s / total_sharpe for s in sharpes]
            final_position = sum(p * w for p, w in zip(positions, weights))
            logger.info(f"[组合] 夏普加权: weights={[f'{w:.3f}' for w in weights]}, final={final_position:.4f}")
        else:
            final_position = sum(positions) / len(positions)
            logger.info(f"[组合] 等权平均: {len(positions)} 个子策略, final={final_position:.4f}")

        return max(-1.0, min(1.0, final_position))

    def _calc_simulated_nav(
        self,
        strategy_id: str,
        symbol: str,
        prev_position: float,
        new_position: float,
        today_close: float,
        today_date: str,
    ) -> Optional[float]:
        """
        计算模拟净值（含滑点和手续费）

        实现思路：
          1. 从 saas_equity_curves 取昨日实盘净值（is_backtest=False）
          2. 若无历史净值，以 INITIAL_NAV 初始化
          3. 计算仓位变化 → 交易成本（滑点 + 手续费）
          4. 根据旧仓位和今日收益率更新净值
          5. UPSERT 到 saas_equity_curves（is_backtest=False）
        """
        try:
            equity_records = get_equity_curve(strategy_id, is_backtest=False, limit=5)

            prev_nav = INITIAL_NAV

            if equity_records:
                latest = equity_records[0]
                prev_nav = float(latest.get('nav_value', INITIAL_NAV))

            market_records = get_market_data(symbol, limit=2, order='timestamp.desc')
            if len(market_records) >= 2:
                yesterday_close = float(market_records[1]['close'])
            else:
                yesterday_close = today_close

            if yesterday_close == 0:
                logger.warning(f"[净值] {symbol} 昨日收盘价为 0，跳过净值计算")
                return None

            daily_return = (today_close - yesterday_close) / yesterday_close

            position_change = abs(new_position - prev_position)
            trade_cost_rate = (SLIPPAGE_BPS + COMMISSION_BPS) / 10000.0
            trade_cost = position_change * trade_cost_rate

            strategy_return = prev_position * daily_return - trade_cost
            new_nav = prev_nav * (1 + strategy_return)

            update_daily_nav(
                strategy_id=strategy_id,
                date=today_date,
                nav_value=round(new_nav, 4),
                is_backtest=False,
            )

            logger.info(
                f"[净值] {strategy_id[:8]}... | "
                f"仓位: {prev_position:.2f}→{new_position:.2f} | "
                f"日收益: {daily_return:.4f} | "
                f"成本: {trade_cost:.6f} | "
                f"NAV: {prev_nav:.2f}→{new_nav:.2f}"
            )

            return new_nav

        except Exception as e:
            logger.error(f"[净值] 计算失败: {e}\n{traceback.format_exc()}")
            return None

    def run_signal_calculation(self, target_symbol: str = None) -> dict:
        """
        执行信号计算主流程

        流程：
          1. 获取所有活跃策略（LIVE 状态）
          2. 按标的分组，拉取宽表
          3. 逐策略沙盒执行，获取 target_position
          4. 计算模拟净值
          5. 回写 Supabase

        Args:
            target_symbol: 可选，指定只计算某个标的的策略

        Returns:
            dict: 各策略的计算结果摘要
        """
        logger.info("=" * 60)
        logger.info("🧠 云端信号与净值引擎启动")
        logger.info("=" * 60)

        config_summary = get_config_summary()
        logger.info(f"配置状态: {config_summary}")

        # 1. 获取活跃策略
        strategies = get_live_strategies(target_symbol=target_symbol)
        if not strategies:
            logger.warning("无活跃策略，引擎退出")
            return {'total': 0, 'success': 0, 'failed': 0, 'details': {}}

        logger.info(f"发现 {len(strategies)} 个活跃策略")

        results = {
            'total': len(strategies),
            'success': 0,
            'failed': 0,
            'details': {},
        }

        # 2. 按标的分组
        symbol_groups: dict[str, list[dict]] = {}
        for s in strategies:
            sym = s.get('target_symbol', 'BTC_USDT')
            if sym not in symbol_groups:
                symbol_groups[sym] = []
            symbol_groups[sym].append(s)

        # 3. 逐标的处理
        for symbol, group in symbol_groups.items():
            logger.info(f"\n{'─' * 40}")
            logger.info(f"📊 处理标的: {symbol} ({len(group)} 个策略)")

            # 拉取宽表（同标的复用）
            df = self._fetch_wide_table(symbol)
            if df.empty:
                logger.error(f"[{symbol}] 宽表为空，跳过该标的的所有策略")
                for s in group:
                    results['failed'] += 1
                    results['details'][s['id']] = {'status': 'error', 'reason': '宽表为空'}
                continue

            today_date = datetime.now(timezone.utc).strftime('%Y-%m-%d')
            today_close = float(df['close'].iloc[-1]) if 'close' in df.columns else 0

            # 4. 逐策略执行
            for strategy in group:
                sid = strategy['id']
                sname = strategy.get('name', 'unknown')
                logger.info(f"  🔧 执行策略: {sname} ({sid[:8]}...)")

                try:
                    # 获取完整策略信息（含 python_code）
                    db = get_client()
                    full_strategy = db.select(
                        'saas_strategies',
                        columns='id,name,python_code,params_json,target_symbol,target_asset,timeframe,current_target_position',
                        filters={'id': f'eq.{sid}'},
                    )

                    if not full_strategy:
                        logger.error(f"  策略 {sid} 查询失败")
                        results['failed'] += 1
                        results['details'][sid] = {'status': 'error', 'reason': '查询失败'}
                        continue

                    s_info = full_strategy[0]
                    python_code = s_info.get('python_code', '')
                    params_json = s_info.get('params_json')

                    if not python_code:
                        logger.error(f"  策略 {sname} 无 python_code")
                        results['failed'] += 1
                        results['details'][sid] = {'status': 'error', 'reason': '无代码'}
                        continue

                    if isinstance(params_json, str):
                        import json as _json
                        try:
                            params_json = _json.loads(params_json)
                        except (ValueError, TypeError):
                            params_json = None

                    target_position = self._execute_strategy_code(python_code, df, params_json)

                    if target_position is None:
                        logger.warning(f"  策略 {sname} 执行返回 None，跳过")
                        results['failed'] += 1
                        results['details'][sid] = {'status': 'error', 'reason': '执行异常'}
                        continue

                    target_position = max(-1.0, min(1.0, target_position))

                    prev_position = float(s_info.get('current_target_position', 0))

                    update_strategy_position(sid, target_position)
                    logger.info(f"  ✅ {sname}: target_position = {target_position:.4f}")

                    nav = self._calc_simulated_nav(
                        strategy_id=sid,
                        symbol=symbol,
                        prev_position=prev_position,
                        new_position=target_position,
                        today_close=today_close,
                        today_date=today_date,
                    )

                    results['success'] += 1
                    results['details'][sid] = {
                        'status': 'success',
                        'name': sname,
                        'target_position': target_position,
                        'nav': nav,
                    }

                except Exception as e:
                    logger.error(f"  ❌ 策略 {sname} 处理异常: {e}")
                    results['failed'] += 1
                    results['details'][sid] = {'status': 'error', 'reason': str(e)}

        logger.info("\n" + "=" * 60)
        logger.info(
            f"🏁 引擎执行完成: 成功 {results['success']}/{results['total']}, "
            f"失败 {results['failed']}/{results['total']}"
        )
        logger.info("=" * 60)

        return results

    def backfill_historical_nav(self, target_symbol: str = None) -> dict:
        """
        回填历史实盘净值：从回测结束日期到今天，逐日计算信号和净值

        核心流程：
          1. 获取活跃策略
          2. 拉取完整宽表（不截断，使用所有可用行情数据）
          3. 对每个策略，在完整宽表上执行 generate_signals，获取每日仓位
          4. 从回测结束日期开始，逐日计算实盘净值
          5. 批量写入 Supabase

        注意：调用方应在调用前清除旧实盘净值数据
        """
        logger.info("=" * 60)
        logger.info("🔄 历史实盘净值回填启动")
        logger.info("=" * 60)

        strategies = get_live_strategies(target_symbol=target_symbol)
        if not strategies:
            logger.warning("无活跃策略，回填退出")
            return {'total': 0, 'backfilled': 0}

        results = {'total': len(strategies), 'backfilled': 0}

        symbol_groups = {}
        for s in strategies:
            sym = s.get('target_symbol', 'BTC_USDT')
            symbol_groups.setdefault(sym, []).append(s)

        for symbol, group in symbol_groups.items():
            logger.info(f"\n📊 处理标的: {symbol} ({len(group)} 个策略)")

            df = self._fetch_wide_table(symbol, days=99999)
            if df.empty:
                logger.error(f"[{symbol}] 宽表为空，跳过")
                continue

            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')

            logger.info(f"[{symbol}] 宽表数据范围: {df.index.min().strftime('%Y-%m-%d')} ~ {df.index.max().strftime('%Y-%m-%d')} ({len(df)} 天)")

            for strategy in group:
                sid = strategy['id']
                sname = strategy.get('name', 'unknown')

                try:
                    db = get_client()
                    if not db:
                        continue
                    full_strategy = db.select(
                        'saas_strategies',
                        columns='id,name,python_code,params_json,target_symbol,backtest_end_date,backtest_start_date,current_target_position',
                        filters={'id': f'eq.{sid}'},
                    )
                    if not full_strategy:
                        continue

                    s_info = full_strategy[0]
                    python_code = s_info.get('python_code', '')
                    params_json = s_info.get('params_json')

                    if isinstance(params_json, str):
                        import json as _json
                        try:
                            params_json = _json.loads(params_json)
                        except (ValueError, TypeError):
                            params_json = None

                    if not python_code:
                        logger.error(f"  {sname} 无 python_code，跳过")
                        continue

                    bt_equity = get_equity_curve(sid, is_backtest=True, limit=1)
                    if bt_equity:
                        last_bt_date = pd.Timestamp(bt_equity[0].get('date'))
                        if last_bt_date.tz is None:
                            last_bt_date = last_bt_date.tz_localize('UTC')
                        start_date = last_bt_date + pd.Timedelta(days=1)
                        logger.info(f"  {sname}: 回测净值最后日期 {last_bt_date.strftime('%Y-%m-%d')}，从 {start_date.strftime('%Y-%m-%d')} 开始回填")
                    else:
                        bt_end = s_info.get('backtest_end_date')
                        if bt_end:
                            start_date = pd.Timestamp(bt_end) + pd.Timedelta(days=1)
                            if start_date.tz is None:
                                start_date = start_date.tz_localize('UTC')
                        else:
                            start_date = df.index.min()
                        logger.info(f"  {sname}: 无回测净值，从 {start_date.strftime('%Y-%m-%d')} 开始回填")

                    data_start = df.index.min()
                    if start_date < data_start:
                        logger.info(f"  {sname}: 回填起始 {start_date.strftime('%Y-%m-%d')} 早于行情数据起始 {data_start.strftime('%Y-%m-%d')}，从行情数据起始开始")
                        start_date = data_start

                    df_subset = df[df.index >= start_date]
                    if df_subset.empty:
                        logger.info(f"  {sname}: 无需回填（start_date={start_date.strftime('%Y-%m-%d')} 在数据范围外）")
                        continue

                    logger.info(f"  🔧 {sname}: 回填范围 {df_subset.index.min().strftime('%Y-%m-%d')} ~ {df_subset.index.max().strftime('%Y-%m-%d')} ({len(df_subset)} 天)")

                    target_positions = self._execute_strategy_for_series(python_code, df, params_json)
                    if target_positions is None:
                        logger.error(f"  {sname}: 策略执行失败，跳过")
                        continue

                    logger.info(f"  {sname}: 策略执行成功，生成 {len(target_positions)} 个仓位信号")

                    prev_nav = INITIAL_NAV
                    if bt_equity:
                        prev_nav = float(bt_equity[0].get('nav_value', INITIAL_NAV))
                        logger.info(f"  {sname}: 从回测末尾净值 {prev_nav:.2f} 开始")

                    prev_position = 0.0
                    nav_records = []
                    sorted_dates = sorted(df_subset.index)
                    skipped = 0

                    for i, date in enumerate(sorted_dates):
                        date_str = date.strftime('%Y-%m-%d')

                        if date in target_positions.index:
                            new_position = float(target_positions.loc[date])
                        else:
                            new_position = prev_position
                            skipped += 1

                        new_position = max(-1.0, min(1.0, new_position))

                        close = float(df_subset.loc[date, 'close']) if 'close' in df_subset.columns else 0

                        if i > 0:
                            prev_date = sorted_dates[i - 1]
                            prev_close = float(df_subset.loc[prev_date, 'close']) if 'close' in df_subset.columns else close
                        else:
                            prev_close = close

                        if prev_close > 0:
                            daily_return = (close - prev_close) / prev_close
                        else:
                            daily_return = 0

                        position_change = abs(new_position - prev_position)
                        trade_cost_rate = (SLIPPAGE_BPS + COMMISSION_BPS) / 10000.0
                        trade_cost = position_change * trade_cost_rate

                        strategy_return = prev_position * daily_return - trade_cost
                        new_nav = prev_nav * (1 + strategy_return)

                        nav_records.append({
                            'strategy_id': sid,
                            'date': date_str,
                            'nav_value': round(new_nav, 4),
                            'is_backtest': False,
                        })

                        prev_nav = new_nav
                        prev_position = new_position

                    if skipped > 0:
                        logger.info(f"  {sname}: {skipped}/{len(sorted_dates)} 天无信号，沿用前日仓位")

                    if nav_records:
                        from saas_platform.database.supabase_client import bulk_upsert_equity_curves
                        count = bulk_upsert_equity_curves(nav_records)
                        logger.info(f"  ✅ {sname}: 回填完成，写入 {count} 条净值记录 (NAV: {INITIAL_NAV:.2f} → {prev_nav:.2f})")

                        last_pos = prev_position
                        update_strategy_position(sid, last_pos)
                        logger.info(f"  📌 {sname}: 更新 current_target_position = {last_pos:.4f}")

                        results['backfilled'] += 1
                    else:
                        logger.warning(f"  {sname}: 未生成任何净值记录")

                except Exception as e:
                    logger.error(f"  ❌ {sname} 回填失败: {e}\n{traceback.format_exc()}")

        logger.info(f"\n🔄 回填完成: {results['backfilled']}/{results['total']} 个策略")
        return results

    def _execute_strategy_for_series(self, python_code: str, df: pd.DataFrame, params_json: dict = None) -> Optional[pd.Series]:
        """
        执行策略代码，返回完整的 target_position 时间序列
        """
        import json as _json

        try:
            sub_strategies = _json.loads(python_code)
            if not isinstance(sub_strategies, list):
                sub_strategies = None
        except (ValueError, TypeError):
            sub_strategies = None

        if sub_strategies:
            return self._execute_portfolio_for_series(sub_strategies, df, params_json)
        else:
            return self._execute_single_for_series(python_code, df)

    def _execute_single_for_series(self, code: str, df: pd.DataFrame, params: dict = None) -> Optional[pd.Series]:
        try:
            safe_globals = {'__builtins__': __builtins__, 'pd': pd, 'np': np}
            safe_locals = {'df': df.copy()}
            exec(code, safe_globals, safe_locals)

            func = safe_locals.get('generate_signals')
            if func is None:
                return None

            if params:
                result = func(safe_locals['df'], params)
            else:
                result = func(safe_locals['df'])

            if result is None:
                return None

            if isinstance(result, pd.DataFrame) and 'target_position' in result.columns:
                series = result['target_position']
                if series.index.tz is None and df.index.tz is not None:
                    series.index = series.index.tz_localize(df.index.tz)
                return series

            if isinstance(result, pd.Series):
                if result.index.tz is None and df.index.tz is not None:
                    result.index = result.index.tz_localize(df.index.tz)
                return result

            return None
        except Exception as e:
            logger.error(f"[沙盒] 执行异常: {e}")
            return None

    def _execute_portfolio_for_series(self, sub_strategies: list, df: pd.DataFrame, params_json: dict = None) -> Optional[pd.Series]:
        weight_mode = 'equal'
        if params_json and isinstance(params_json, dict):
            weight_mode = params_json.get('weight_mode', 'equal')

        all_positions = []
        all_sharpes = []

        for i, sub in enumerate(sub_strategies):
            sub_code = sub.get('code', '')
            sub_params = sub.get('params', {})
            sub_sharpe = sub.get('sharpe') or 0

            if not sub_code:
                continue

            pos_series = self._execute_single_for_series(sub_code, df, sub_params)
            if pos_series is not None:
                all_positions.append(pos_series.clip(-1, 1))
                all_sharpes.append(max(sub_sharpe, 0))

        if not all_positions:
            return None

        if weight_mode == 'sharpe' and sum(all_sharpes) > 0:
            total_sharpe = sum(all_sharpes)
            weights = [s / total_sharpe for s in all_sharpes]
            combined = sum(p * w for p, w in zip(all_positions, weights))
        else:
            combined = sum(all_positions) / len(all_positions)

        return combined.clip(-1, 1)

    def clear_cache(self):
        self._data_cache.clear()


# ============================================================
# 便捷入口
# ============================================================

def run_daily_signal(target_symbol: str = None) -> dict:
    """
    每日定时任务入口：计算所有活跃策略的信号与净值

    Args:
        target_symbol: 可选，指定只计算某个标的
    """
    engine = CloudSignalEngine()
    return engine.run_signal_calculation(target_symbol=target_symbol)
