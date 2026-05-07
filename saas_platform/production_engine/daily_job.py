#!/usr/bin/env python3
"""
云端实盘生产大脑 - 每日定时任务入口

执行流程：
  Step 1: 增量拉取最新行情 + 因子数据 → Supabase
  Step 2: 热加载策略代码，计算信号 + 模拟净值 → Supabase

部署方式：
  - Linux cron: 0 8 * * * cd /path/to/project && python -m saas_platform.production_engine.daily_job
  - Windows Task Scheduler: 每日 08:00 执行
  - 云函数 / GitHub Actions: 定时触发
"""

import os
import sys
import logging
import time
from datetime import datetime, timezone

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from saas_platform.saas_config import get_config_summary

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger('daily_job')


def run():
    start_time = time.time()
    now = datetime.now(timezone.utc)

    logger.info("=" * 70)
    logger.info(f"🚀 云端实盘生产大脑启动 - {now.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    logger.info("=" * 70)

    config_summary = get_config_summary()
    logger.info(f"📋 配置状态: {config_summary}")

    # ── Step 1: 增量拉取数据 ──
    logger.info("\n📡 Step 1: 增量拉取行情与因子数据...")
    try:
        from saas_platform.production_engine.data_fetcher import run_daily_sync
        sync_result = run_daily_sync()
        logger.info(f"✅ 数据同步完成: 行情 {sync_result.get('total_market', 0)} 条, "
                     f"因子 {sync_result.get('total_factors', 0)} 条")
    except Exception as e:
        logger.error(f"❌ 数据同步失败: {e}")
        sync_result = {'error': str(e)}

    # ── Step 2: 计算信号与净值 ──
    logger.info("\n🧠 Step 2: 计算策略信号与模拟净值...")
    try:
        from saas_platform.production_engine.signal_engine import run_daily_signal
        signal_result = run_daily_signal()
        logger.info(f"✅ 信号计算完成: 成功 {signal_result.get('success', 0)}/{signal_result.get('total', 0)}")
    except Exception as e:
        logger.error(f"❌ 信号计算失败: {e}")
        signal_result = {'error': str(e)}

    # ── Step 3: 跟单执行路由 ──
    logger.info("\n🔀 Step 3: 云端跟单执行路由...")
    try:
        from saas_platform.production_engine.copy_trading_router import run_copy_trading
        copy_result = run_copy_trading()
        logger.info(f"✅ 跟单路由完成: 策略 {copy_result.get('total_strategies', 0)} 个, "
                     f"用户 {copy_result.get('total_users', 0)} 人, "
                     f"下单 {copy_result.get('total_orders', 0)} 笔, "
                     f"异常 {copy_result.get('errors', 0)} 个")
    except Exception as e:
        logger.error(f"❌ 跟单路由失败: {e}")
        copy_result = {'error': str(e)}

    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info(f"🏁 每日任务完成 - 耗时 {elapsed:.1f}s")
    logger.info("=" * 70)

    return {
        'sync': sync_result,
        'signal': signal_result,
        'copy_trading': copy_result,
        'elapsed_seconds': round(elapsed, 1),
    }


if __name__ == '__main__':
    result = run()
    print(f"\n执行结果: {result}")
