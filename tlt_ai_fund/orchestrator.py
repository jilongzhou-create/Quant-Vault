#!/usr/bin/env python3
"""
TLT AI Fund - Orchestrator (总调度器)

流水线:
  DataEngineer -> FactorMiner -> Gatekeeper(IS) -> OOS Autopsy -> MLSynthesizer
  数据拓荒者     因子挖掘器      IS 铁面裁判      OOS 验尸       ML 合成器

铁律:
  Phase 3 (IS 审核): 因子逻辑构建和阈值选择只能基于 IS 数据 (2007-2019)
  Phase 3.5 (OOS 验尸): IS 通过后代码 FROZEN, OOS 失败直接标记 DEAD_OOS 永久废弃
  Phase 4 (ML 合成): 仅用通过 IS + OOS 双重验证的因子
"""

import sys
import os

if __name__ == '__main__':
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

import time
import logging
from uuid import uuid4

from tlt_ai_fund.config import (
    MINING_DIRECTIONS, MINING_METHODS, IS_START, IS_END,
    DISABLED_METHODS, AUTO_MINE_INTERVAL_SEC, AUTO_MINE_MAX_CYCLES,
)
from tlt_ai_fund.db.schema import (
    init_ai_fund_tables, log_agent, get_accepted_factors,
    get_factors_needing_oos_autopsy, seed_data_requirements,
    check_data_feasibility, update_data_coverage_status,
    update_factor_status,
)
from tlt_ai_fund.agents.data_engineer import DataEngineer
from tlt_ai_fund.agents.factor_miner import FactorMiner, load_factor_instance
from tlt_ai_fund.agents.gatekeeper import Gatekeeper
from tlt_ai_fund.agents.ml_synthesizer import MLSynthesizer

MAX_REFLEXION_RETRIES = 2

logger = logging.getLogger(__name__)


class Orchestrator:

    def __init__(self, model_type: str = 'zscore_pulse'):
        init_ai_fund_tables()
        seed_data_requirements()

        self.data_engineer = DataEngineer()
        self.factor_miner = FactorMiner(self.data_engineer)
        self.gatekeeper = Gatekeeper()
        self.ml_synthesizer = MLSynthesizer(model_type=model_type)

    def run_cycle(self, directions: list[str] = None,
                  methods: list[str] = None,
                  skip_mining: bool = False,
                  skip_oos: bool = False) -> dict:
        run_id = uuid4().hex[:8]
        logger.info(f"{'='*70}")
        logger.info(f"  TLT AI Fund - Cycle {run_id}")
        logger.info(f"{'='*70}")

        # Phase 1: Data scan + feasibility check
        t0 = time.time()
        data_inv = self.data_engineer.scan_available_data()
        available_fields = self.data_engineer.get_available_field_names()
        t1 = time.time()
        log_agent(run_id, 'data_engineer', 'scan',
                  output_summary=f'{len(data_inv)} series',
                  duration_sec=t1 - t0)
        logger.info(f"[Phase 1] Data lake: {len(data_inv)} series ({t1-t0:.1f}s)")

        # Phase 1.5: Data feasibility pre-check
        dirs = directions or MINING_DIRECTIONS
        meths = methods or MINING_METHODS

        feasible_combos = []
        skipped_combos = []

        for direction in dirs:
            for method in meths:
                if method in DISABLED_METHODS:
                    skipped_combos.append(f"{direction}/{method} (method disabled)")
                    logger.info(f"[Phase 1.5] X {direction}/{method}: method disabled, skip")
                    continue

                feasibility = check_data_feasibility(direction, method, available_fields)

                if not feasibility['feasible']:
                    skipped_combos.append(
                        f"{direction}/{method} (missing: {', '.join(feasibility['missing_required'])})"
                    )
                    logger.info(
                        f"[Phase 1.5] X {direction}/{method}: "
                        f"missing core data {feasibility['missing_required']}, skip"
                    )
                    for f in feasibility['missing_required']:
                        update_data_coverage_status(direction, method, f, 'missing_unfetchable')
                    continue

                if feasibility['auto_fetchable']:
                    logger.info(
                        f"[Phase 1.5] {direction}/{method}: "
                        f"auto-fetching {len(feasibility['auto_fetchable'])} missing data..."
                    )
                    fetch_result = self.data_engineer.auto_fetch_missing(feasibility['auto_fetchable'])

                    if fetch_result['fetched']:
                        available_fields.update(fetch_result['fetched'])
                        logger.info(f"[Phase 1.5] Fetched: {fetch_result['fetched']}")
                        for f in fetch_result['fetched']:
                            update_data_coverage_status(direction, method, f, 'fetched')

                    if fetch_result['failed']:
                        still_missing = [item['field'] for item in fetch_result['failed']]
                        all_required_still_missing = all(
                            f in still_missing for f in feasibility['missing_required']
                        )
                        if all_required_still_missing:
                            skipped_combos.append(
                                f"{direction}/{method} (fetch failed: {', '.join(still_missing)})"
                            )
                            logger.info(
                                f"[Phase 1.5] X {direction}/{method}: "
                                f"core data fetch failed {still_missing}, skip"
                            )
                            for f in still_missing:
                                update_data_coverage_status(direction, method, f, 'fetch_failed')
                            continue
                        else:
                            for item in fetch_result['failed']:
                                update_data_coverage_status(
                                    direction, method, item['field'], 'fetch_failed'
                                )

                if feasibility['missing_optional']:
                    logger.info(
                        f"[Phase 1.5] {direction}/{method}: "
                        f"optional data missing (ok): {feasibility['missing_optional']}"
                    )
                    for f in feasibility['missing_optional']:
                        update_data_coverage_status(direction, method, f, 'optional_missing')

                feasible_combos.append((direction, method))
                logger.info(f"[Phase 1.5] OK {direction}/{method}: feasible")

        logger.info(
            f"[Phase 1.5] Pre-check done: {len(feasible_combos)} feasible, "
            f"{len(skipped_combos)} skipped"
        )

        # Phase 2+3: Factor mining + IS audit (Actor-Critic Reflexion Loop)
        new_factor_ids = []
        is_accepted_ids = []
        is_dead_ids = []
        reflexion_stats = {'attempts': 0, 'reflexions': 0, 'reflexion_successes': 0}

        if not skip_mining:
            for direction, method in feasible_combos:
                t0 = time.time()
                try:
                    ids = self.factor_miner.mine(direction, method, data_inv)
                    new_factor_ids.extend(ids)
                except Exception as e:
                    logger.error(f"[Phase 2] Mining failed {direction}/{method}: {e}")
                    ids = []
                t1 = time.time()
                log_agent(run_id, 'factor_miner', 'mine',
                          input_summary=f'{direction}/{method}',
                          output_summary=f'{len(ids)} factors',
                          duration_sec=t1 - t0)
                logger.info(f"[Phase 2] {direction}/{method}: generated {len(ids)} factors ({t1-t0:.1f}s)")

                for factor_id in ids:
                    reflexion_stats['attempts'] += 1
                    loop_result = self._reflexion_loop(
                        run_id, factor_id, direction, method, data_inv
                    )
                    reflexion_stats['reflexions'] += loop_result['reflexion_count']
                    if loop_result['reflexion_success']:
                        reflexion_stats['reflexion_successes'] += 1
                    if loop_result['accepted']:
                        is_accepted_ids.append(factor_id)
                    else:
                        is_dead_ids.append(factor_id)
        else:
            logger.info("[Phase 2] Mining skipped")

        # Audit legacy draft factors
        legacy_factors = self._get_legacy_draft_factors(new_factor_ids)
        for item in legacy_factors:
            t0 = time.time()
            try:
                result = self.gatekeeper.audit(item['factor_id'], item['source_file'])
            except Exception as e:
                logger.error(f"[Phase 3] Audit exception {item['factor_id']}: {e}")
                result = {'verdict': 'DEAD', 'reject_reason': str(e)}
            t1 = time.time()

            log_agent(run_id, 'gatekeeper', 'is_audit',
                      input_summary=item['factor_id'],
                      output_summary=result['verdict'],
                      duration_sec=t1 - t0)

            if result['verdict'] == 'ACCEPTED':
                is_accepted_ids.append(item['factor_id'])
                logger.info(
                    f"[Phase 3] OK {item['factor_id']}: IS ACCEPTED "
                    f"(CondIC={result.get('conditional_ic', 0):+.4f}, "
                    f"HitRate={result.get('hit_rate', 0):.1%})"
                )
            else:
                is_dead_ids.append(item['factor_id'])
                logger.info(
                    f"[Phase 3] X {item['factor_id']}: IS DEAD "
                    f"({result.get('reject_reason', 'unknown')})"
                )

        logger.info(
            f"[Phase 2+3] Mining+Audit done: {len(is_accepted_ids)} ACCEPTED, "
            f"{len(is_dead_ids)} DEAD "
            f"(reflexion retries: {reflexion_stats['reflexions']}, "
            f"reflexion successes: {reflexion_stats['reflexion_successes']})"
        )

        # Phase 3.5: OOS Autopsy
        oos_survived_ids = []
        oos_dead_ids = []

        if not skip_oos:
            factors_for_autopsy = get_factors_needing_oos_autopsy()

            if factors_for_autopsy:
                logger.info(f"[Phase 3.5] OOS autopsy: {len(factors_for_autopsy)} factors pending")

                for item in factors_for_autopsy:
                    t0 = time.time()
                    try:
                        result = self.gatekeeper.oos_autopsy(item['factor_id'], item['source_file'])
                    except Exception as e:
                        logger.error(f"[Phase 3.5] Autopsy exception {item['factor_id']}: {e}")
                        result = {'verdict': 'DEAD_OOS', 'reject_reason': str(e)}
                    t1 = time.time()

                    log_agent(run_id, 'gatekeeper', 'oos_autopsy',
                              input_summary=item['factor_id'],
                              output_summary=result['verdict'],
                              duration_sec=t1 - t0)

                    if result['verdict'] == 'ACCEPTED':
                        oos_survived_ids.append(item['factor_id'])
                        logger.info(
                            f"[Phase 3.5] OK {item['factor_id']}: OOS SURVIVED "
                            f"(CondIC={result.get('conditional_ic', 0):+.4f}, "
                            f"HitRate={result.get('hit_rate', 0):.1%})"
                        )
                    else:
                        oos_dead_ids.append(item['factor_id'])
                        logger.info(
                            f"[Phase 3.5] X {item['factor_id']}: OOS DEAD "
                            f"({result.get('reject_reason', 'unknown')})"
                        )

                logger.info(
                    f"[Phase 3.5] OOS autopsy done: {len(oos_survived_ids)} SURVIVED, "
                    f"{len(oos_dead_ids)} DEAD_OOS"
                )
            else:
                logger.info("[Phase 3.5] No factors need OOS autopsy")
        else:
            logger.info("[Phase 3.5] OOS autopsy skipped (debug mode)")

        # Phase 4: ML Synthesis
        all_accepted = get_accepted_factors()
        if all_accepted:
            t0 = time.time()
            try:
                self._run_synthesis(all_accepted)
                t1 = time.time()
                log_agent(run_id, 'ml_synthesizer', 'synthesize',
                          output_summary=f'{len(all_accepted)} factors',
                          duration_sec=t1 - t0)
                logger.info(f"[Phase 4] ML synthesis done: {len(all_accepted)} factors ({t1-t0:.1f}s)")
            except Exception as e:
                logger.error(f"[Phase 4] ML synthesis failed: {e}")
                t1 = time.time()
                log_agent(run_id, 'ml_synthesizer', 'synthesize',
                          output_summary=f'error: {e}',
                          duration_sec=t1 - t0,
                          status='error')
        else:
            logger.info("[Phase 4] No ACCEPTED factors, skip synthesis")

        # Phase 5: Report
        report = self._generate_report(
            run_id, new_factor_ids,
            is_accepted_ids, is_dead_ids,
            oos_survived_ids, oos_dead_ids,
            all_accepted,
            feasible_combos, skipped_combos,
        )
        logger.info(f"\n{report['summary']}")
        return report

    def _reflexion_loop(self, run_id: str, factor_id: str,
                        direction: str, method: str,
                        data_inv: list) -> dict:
        source_file = f"{factor_id}.py"
        reflexion_count = 0
        reflexion_success = False

        for attempt in range(1 + MAX_REFLEXION_RETRIES):
            t0 = time.time()
            try:
                result = self.gatekeeper.audit(factor_id, source_file)
            except Exception as e:
                logger.error(f"[Reflexion] Audit exception {factor_id} (attempt {attempt}): {e}")
                result = {'verdict': 'DEAD', 'reject_reason': str(e)}
            t1 = time.time()

            log_agent(run_id, 'gatekeeper', 'is_audit',
                      input_summary=f"{factor_id} (attempt {attempt})",
                      output_summary=result['verdict'],
                      duration_sec=t1 - t0)

            if result['verdict'] == 'ACCEPTED':
                logger.info(
                    f"[Reflexion] OK {factor_id}: IS ACCEPTED "
                    f"(attempt {attempt}, "
                    f"CondIC={result.get('conditional_ic', 0):+.4f}, "
                    f"HitRate={result.get('hit_rate', 0):.1%})"
                )
                if attempt > 0:
                    reflexion_success = True
                return {
                    'accepted': True,
                    'reflexion_count': reflexion_count,
                    'reflexion_success': reflexion_success,
                }

            reject_reason = result.get('reject_reason', 'unknown')
            logger.info(
                f"[Reflexion] X {factor_id}: IS DEAD "
                f"(attempt {attempt}, reason: {reject_reason})"
            )

            if attempt >= MAX_REFLEXION_RETRIES:
                logger.info(f"[Reflexion] {factor_id}: exhausted {1 + MAX_REFLEXION_RETRIES} attempts, giving up")
                return {
                    'accepted': False,
                    'reflexion_count': reflexion_count,
                    'reflexion_success': False,
                }

            reflexion_prompt = self._build_reflexion_prompt(factor_id, result, direction, method)
            reflexion_count += 1
            logger.info(f"[Reflexion] {factor_id}: starting attempt {attempt + 1} reflexion rewrite...")

            try:
                new_ids = self.factor_miner.mine(
                    direction, method, data_inv,
                    reflexion_prompt=reflexion_prompt,
                    overwrite_factor_id=factor_id,
                )
                if not new_ids:
                    logger.warning(f"[Reflexion] {factor_id}: reflexion returned empty code, giving up")
                    return {
                        'accepted': False,
                        'reflexion_count': reflexion_count,
                        'reflexion_success': False,
                    }
            except Exception as e:
                logger.error(f"[Reflexion] Reflexion rewrite failed {factor_id}: {e}")
                return {
                    'accepted': False,
                    'reflexion_count': reflexion_count,
                    'reflexion_success': False,
                }

        return {
            'accepted': False,
            'reflexion_count': reflexion_count,
            'reflexion_success': False,
        }

    def _build_reflexion_prompt(self, factor_id: str, audit_result: dict,
                                direction: str, method: str) -> str:
        reject_reason = audit_result.get('reject_reason', 'unknown')
        cond_ic = audit_result.get('conditional_ic', 0)
        hit_rate = audit_result.get('hit_rate', 0)
        trigger_rate = audit_result.get('trigger_rate', 0)
        pk_triggered = audit_result.get('pk_triggered', False)
        pk_won = audit_result.get('pk_won', None)
        max_overlap_a = audit_result.get('max_overlap_a', 0)
        max_overlap_b = audit_result.get('max_overlap_b', 0)

        diagnosis_lines = [f"- Reject reason: {reject_reason}"]
        diagnosis_lines.append(f"- Trigger Rate: {trigger_rate:.1%}")
        diagnosis_lines.append(f"- Conditional IC: {cond_ic:+.4f}")
        diagnosis_lines.append(f"- Hit Rate: {hit_rate:.1%}")

        if pk_triggered:
            diagnosis_lines.append(
                f"- Orthogonality PK: triggered "
                f"(Overlap_A={max_overlap_a:.2f}, Overlap_B={max_overlap_b:.2f})"
            )
            if pk_won is False:
                diagnosis_lines.append("- PK result: LOST (CondIC weaker than overlapping opponent)")

        fix_instructions = []
        if 'code' in reject_reason.lower() or 'error' in reject_reason.lower() or 'failed' in reject_reason.lower():
            fix_instructions.append(
                "1. Code execution error: Fix the Bug directly. Handle NaN values (.fillna(0.0)), "
                "ensure all column names exist in DataFrame (use data.get('col', pd.Series) to prevent KeyError)."
            )
        if trigger_rate > 0.20:
            fix_instructions.append(
                "2. Trigger rate too high (>20%): Your conditions are too loose or you're not using diff(). "
                "Please completely change the calculation logic (e.g., use ratio change, first derivative, "
                "Z-Score threshold trigger). The signal must NOT be non-zero on most trading days!"
            )
        if trigger_rate < 0.001:
            fix_instructions.append(
                "2. Trigger rate near 0%: Your conditions are too strict, almost no trading days trigger. "
                "Please relax the threshold or switch to a continuous signal approach (e.g., tanh compression)."
            )
        if cond_ic <= 0.015 and trigger_rate > 0:
            fix_instructions.append(
                "3. Conditional IC extremely low: Signal direction prediction is insufficient. "
                "Please re-examine the economic logic to ensure the signal direction has a causal "
                "relationship with TLT price movement."
            )
        if hit_rate < 0.53 and trigger_rate > 0:
            fix_instructions.append(
                "4. Hit Rate too low (<53%): Signal direction prediction accuracy is insufficient. "
                "Please check if the signal direction is correct (bullish TLT should be positive), "
                "or adjust trigger conditions."
            )
        if pk_triggered and pk_won is False:
            fix_instructions.append(
                "5. Orthogonality PK lost: Your logic overlaps heavily with an existing factor. "
                "Please ABANDON the current approach and find a completely new economic dimension. "
                "Do NOT just tweak parameters (e.g., changing >2.0 to >2.05), you must restructure at the logic level!"
            )
        if 'marginal' in reject_reason.lower() or 'toxic' in reject_reason.lower():
            fix_instructions.append(
                "6. Marginal contribution insufficient (toxic factor): Your factor passed individual metrics, "
                "but adding it to the portfolio didn't improve Sharpe by 0.005+. "
                "This means your signal creates internal friction or redundancy with existing factors. "
                "Please completely change the factor's trigger logic or signal direction."
            )

        if not fix_instructions:
            fix_instructions.append(
                "1. Please re-examine the factor's economic logic to ensure the signal has a clear causal explanation."
            )

        anti_phacking = (
            "WARNING: Anti-overfitting: Absolutely do NOT try to pass by fine-tuning constant parameters "
            "(e.g., changing >2.0 to >2.05, or window=20 to window=21). "
            "You MUST restructure at the financial economics logic level!"
        )

        prompt = (
            f"Your previous factor code ran successfully, but was rejected by the system judge during IS testing.\n\n"
            f"[Failure Diagnosis Report]:\n"
            + "\n".join(diagnosis_lines) + "\n\n"
            f"[Mandatory Fix Instructions]:\n"
            + "\n".join(fix_instructions) + "\n\n"
            f"{anti_phacking}\n\n"
            f"Please output the modified complete Python code."
        )

        return prompt

    def _get_legacy_draft_factors(self, exclude_ids: list[str]) -> list[dict]:
        from tlt_ai_fund.db.schema import get_legacy_draft_factors
        return get_legacy_draft_factors(exclude_ids)

    def _run_synthesis(self, accepted_factors: list[dict]):
        from tlt_ai_fund.core.tlt_macro_framework import TltMacroFramework

        fw = TltMacroFramework()
        df = fw.run(start_date=IS_START, end_date=IS_END)
        df = Gatekeeper._enrich_with_data_lake(df, IS_START, IS_END)

        core_col = 'tlt_core_signal' if 'tlt_core_signal' in df.columns else 'core_signal'
        core_signal = df[core_col]

        factor_signals = {}
        for af in accepted_factors:
            try:
                instance = load_factor_instance(af['factor_id'], af['source_file'])
                sig = instance.calculate_signal(df)
                factor_signals[af['factor_id']] = sig
            except Exception as e:
                logger.warning(f"Failed to load factor {af['factor_id']}: {e}")

        if not factor_signals:
            logger.warning("No usable factor signals, skip synthesis")
            return

        result_df = self.ml_synthesizer.synthesize(df, core_signal, factor_signals)

        if not result_df.empty:
            logger.info(
                f"[MLSynthesizer] Synthesis complete: "
                f"sat_composite mean={result_df['sat_composite'].mean():.4f}, "
                f"total_score mean={result_df['total_score'].mean():.4f}"
            )

    def _generate_report(self, run_id: str, new_factor_ids: list,
                         is_accepted_ids: list, is_dead_ids: list,
                         oos_survived_ids: list, oos_dead_ids: list,
                         all_accepted: list,
                         feasible_combos: list = None,
                         skipped_combos: list = None) -> dict:
        lines = [
            f"{'='*60}",
            f"  TLT AI Fund - Cycle Report [{run_id}]",
            f"{'='*60}",
            f"",
            f"  Feasible combos: {len(feasible_combos or [])}",
            f"  Skipped combos: {len(skipped_combos or [])}",
            f"  New factors mined: {len(new_factor_ids)}",
            f"  IS accepted: {len(is_accepted_ids)}",
            f"  IS dead: {len(is_dead_ids)}",
            f"  OOS survived: {len(oos_survived_ids)}",
            f"  OOS dead: {len(oos_dead_ids)}",
            f"  Production pool: {len(all_accepted)}",
            f"",
        ]

        if skipped_combos:
            lines.append("  Skipped combos:")
            for s in skipped_combos:
                lines.append(f"    - {s}")
            lines.append("")

        if is_accepted_ids:
            lines.append("  IS ACCEPTED:")
            for fid in is_accepted_ids:
                lines.append(f"    - {fid}")

        if is_dead_ids:
            lines.append("  IS DEAD:")
            for fid in is_dead_ids:
                lines.append(f"    - {fid}")

        if oos_survived_ids:
            lines.append("  OOS SURVIVED:")
            for fid in oos_survived_ids:
                lines.append(f"    - {fid}")

        if oos_dead_ids:
            lines.append("  OOS DEAD (permanently discarded):")
            for fid in oos_dead_ids:
                lines.append(f"    - {fid}")

        if all_accepted:
            lines.append("")
            lines.append("  Production Factor Pool (IS+OOS passed):")
            for af in all_accepted:
                lines.append(
                    f"    - {af['factor_id']} "
                    f"({af['mining_direction']}/{af['mining_method']})"
                )

        lines.append(f"{'='*60}")

        return {
            'run_id': run_id,
            'feasible_combos': len(feasible_combos or []),
            'skipped_combos': len(skipped_combos or []),
            'new_factors': len(new_factor_ids),
            'is_accepted': len(is_accepted_ids),
            'is_dead': len(is_dead_ids),
            'oos_survived': len(oos_survived_ids),
            'oos_dead': len(oos_dead_ids),
            'production_pool_size': len(all_accepted),
            'is_accepted_ids': is_accepted_ids,
            'is_dead_ids': is_dead_ids,
            'oos_survived_ids': oos_survived_ids,
            'oos_dead_ids': oos_dead_ids,
            'summary': '\n'.join(lines),
        }


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    parser = argparse.ArgumentParser(description='TLT AI Fund Orchestrator')
    parser.add_argument('--mode', choices=['once', 'loop'], default='loop',
                        help='Run mode: once=single, loop=continuous auto-mining (default: loop)')
    parser.add_argument('--interval', type=int, default=None,
                        help=f'Loop interval in seconds (default: {AUTO_MINE_INTERVAL_SEC})')
    parser.add_argument('--max-cycles', type=int, default=None,
                        help=f'Max loop cycles, 0=infinite (default: {AUTO_MINE_MAX_CYCLES})')
    parser.add_argument('--skip-mining', action='store_true',
                        help='Skip mining phase, only audit existing draft factors')
    parser.add_argument('--skip-oos', action='store_true',
                        help='Skip OOS autopsy (debug only)')
    parser.add_argument('--direction', type=str, default=None,
                        help='Mine only specified direction (unstructured/microstructure/volatility)')
    parser.add_argument('--method', type=str, default=None,
                        help='Mine only specified method (unstructured/options/nonlinear)')
    args = parser.parse_args()

    orch = Orchestrator(model_type='zscore_pulse')

    directions = [args.direction] if args.direction else None
    methods = [args.method] if args.method else None
    interval = args.interval or AUTO_MINE_INTERVAL_SEC
    max_cycles = args.max_cycles if args.max_cycles is not None else AUTO_MINE_MAX_CYCLES

    if args.mode == 'once':
        logger.info("Single run mode")
        orch.run_cycle(
            directions=directions,
            methods=methods,
            skip_mining=args.skip_mining,
            skip_oos=args.skip_oos,
        )
    else:
        logger.info(f"Continuous auto-mining mode | interval: {interval}s | max cycles: {max_cycles or 'infinite'}")
        cycle_count = 0
        while True:
            cycle_count += 1
            logger.info(f"{'#'*70}")
            logger.info(f"  Auto-Mine Cycle #{cycle_count}")
            logger.info(f"{'#'*70}")

            try:
                orch.run_cycle(
                    directions=directions,
                    methods=methods,
                    skip_mining=args.skip_mining,
                    skip_oos=args.skip_oos,
                )
            except Exception as e:
                logger.error(f"Cycle #{cycle_count} exception: {e}", exc_info=True)

            if max_cycles > 0 and cycle_count >= max_cycles:
                logger.info(f"Reached max cycles {max_cycles}, exiting")
                break

            logger.info(f"Next cycle in {interval}s... (Ctrl+C to exit)")
            try:
                time.sleep(interval)
            except KeyboardInterrupt:
                logger.info("User interrupted, exiting")
                break
