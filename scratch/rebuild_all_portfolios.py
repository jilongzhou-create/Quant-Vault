#!/usr/bin/env python3
"""
批量重跑所有组合回测，用各自的 weight_mode 更新 metrics
weight_mode 为空的组合默认使用 sharpe 模式
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sqlite3
from database.db_manager import DB_PATH
from trading_engine.portfolio_optimizer import run_portfolio_optimization
from logger_setup import get_logger

logger = get_logger(__name__)


def main():
    print("=" * 80)
    print("🔄 批量重跑所有组合回测".center(76))
    print("=" * 80)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT portfolio_id, name, weight_mode, status
        FROM portfolios
        ORDER BY portfolio_id
    ''')

    portfolios = [dict(row) for row in cursor.fetchall()]
    conn.close()

    if not portfolios:
        print("\n❌ 没有找到任何组合！")
        return

    print(f"\n📋 共找到 {len(portfolios)} 个组合:")
    for p in portfolios:
        mode = p['weight_mode'] or 'sharpe (默认)'
        print(f"  ID={p['portfolio_id']:<4} {p['name'][:40]:<42} mode={mode}")

    success = 0
    fail = 0

    for i, p in enumerate(portfolios, 1):
        pid = p['portfolio_id']
        name = p['name']
        mode = p['weight_mode'] or 'sharpe'

        print(f"\n{'─' * 70}")
        print(f"  [{i}/{len(portfolios)}] 组合: {name} (ID={pid}) | 模式: {mode}")
        print(f"{'─' * 70}")

        try:
            metrics = run_portfolio_optimization(pid, mode)

            if metrics:
                print(f"  ✅ 回测成功:")
                print(f"     年化收益: {metrics.get('annualized_return', 0)*100:.2f}%")
                print(f"     夏普率:   {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"     最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
                success += 1
            else:
                print(f"  ❌ 回测返回空指标")
                fail += 1
        except Exception as e:
            logger.error(f"组合 {pid} 回测失败: {e}")
            print(f"  ❌ 回测失败: {e}")
            fail += 1

    print(f"\n{'=' * 80}")
    print(f"🏁 全部完成 | 成功: {success} | 失败: {fail}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
