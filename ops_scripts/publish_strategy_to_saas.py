#!/usr/bin/env python3
"""
本地投研系统 → 云端 SaaS 平台 发布桥梁

这是本地投研目录中唯一允许与 SaaS 数据库交互的脚本。
负责将本地跑出的优秀策略组合"一键上云"。

功能：
  1. 查看云端已有策略列表
  2. 发布新策略（自动按名称去重）
  3. 更新已有策略的净值数据
  4. 删除云端策略
"""

import sys
import os
import json
import sqlite3
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DB_PATH
from logger_setup import get_logger

logger = get_logger(__name__)

logging.getLogger('saas_platform').setLevel(logging.INFO)
logging.getLogger('saas_platform').addHandler(logging.StreamHandler())


def _get_portfolio_info(portfolio_id: int) -> dict:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT portfolio_id, name, description, status,
               target_asset, target_symbol, weight_mode,
               metric_annualized_return, metric_sharpe, metric_max_drawdown,
               created_at
        FROM portfolios
        WHERE portfolio_id = ?
    ''', (portfolio_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {}
    return dict(row)


def _get_sub_strategies(portfolio_id: int) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT
            pc.dir_id,
            sd.name AS strategy_name,
            sd.description,
            sd.target_asset,
            sd.target_symbol,
            sd.timeframe,
            sv.code_content,
            sv.params_json,
            sv.metric_sharpe,
            sv.metric_annualized_return,
            sv.metric_max_drawdown,
            sv.metric_win_rate,
            sv.metric_total_trades,
            sv.metric_start_date,
            sv.metric_end_date
        FROM portfolio_components pc
        JOIN strategy_directions sd ON pc.dir_id = sd.dir_id
        JOIN strategy_versions sv ON sd.best_version_id = sv.ver_id
        WHERE pc.portfolio_id = ?
          AND sv.run_status != 'OVERFITTED'
    ''', (portfolio_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _get_backtest_nav_records(portfolio_id: int) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT date, nav, daily_return, combined_signal, turnover, fee_paid
        FROM portfolio_daily_records
        WHERE portfolio_id = ? AND run_phase = 'BACKTEST'
        ORDER BY date ASC
    ''', (portfolio_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def list_cloud_strategies():
    from saas_platform.database.supabase_client import get_client

    db = get_client()
    if not db:
        print("\n❌ Supabase 未连接，请检查配置")
        return

    strategies = db.select(
        'saas_strategies',
        columns='id,name,status,target_symbol,backtest_sharpe,backtest_annualized_return,backtest_max_drawdown,created_at,updated_at',
        order='created_at.desc',
    )

    if not strategies:
        print("\n📋 云端暂无策略")
        return

    print("\n" + "=" * 110)
    print("📋 云端已有策略列表:")
    print("=" * 110)
    print(f"{'序号':<5} {'名称':<42} {'状态':<7} {'标的':<12} {'夏普':>8} {'年化':>9} {'更新时间':<20}")
    print("-" * 110)
    for i, s in enumerate(strategies, 1):
        name = s.get('name', 'N/A')[:40]
        status = s.get('status', '?')
        symbol = s.get('target_symbol', '?')
        sharpe = s.get('backtest_sharpe')
        ann_ret = s.get('backtest_annualized_return')
        updated = s.get('updated_at', '')[:19].replace('T', ' ')
        sharpe_str = f"{sharpe:.2f}" if sharpe else '—'
        ann_str = f"{ann_ret:.1%}" if ann_ret else '—'
        print(f"{i:<5} {name:<42} {status:<7} {symbol:<12} {sharpe_str:>8} {ann_str:>9} {updated:<20}")
    print("=" * 110)
    print(f"共 {len(strategies)} 个策略")


def delete_cloud_strategy():
    from saas_platform.database.supabase_client import get_client

    db = get_client()
    if not db:
        print("\n❌ Supabase 未连接")
        return

    strategies = db.select(
        'saas_strategies',
        columns='id,name,status,target_symbol',
        order='created_at.desc',
    )

    if not strategies:
        print("\n📋 云端暂无策略")
        return

    print("\n📋 选择要删除的策略:")
    for i, s in enumerate(strategies, 1):
        print(f"  {i}. [{s.get('status', '?')}] {s.get('name', 'N/A')} ({s.get('target_symbol', '?')})")

    choice = input("\n输入序号删除 (q 取消): ").strip()
    if choice.lower() == 'q':
        return

    try:
        idx = int(choice)
        if 1 <= idx <= len(strategies):
            target = strategies[idx - 1]
            confirm = input(f"⚠️ 确认删除 [{target['name']}]? 输入 YES 确认: ").strip()
            if confirm == 'YES':
                db.delete('saas_equity_curves', filters={'strategy_id': f"eq.{target['id']}"})
                db.delete('saas_daily_insights', filters={'strategy_id': f"eq.{target['id']}"})
                db.delete('saas_subscriptions', filters={'strategy_id': f"eq.{target['id']}"})
                db.delete('saas_strategies', filters={'id': f"eq.{target['id']}"})
                print(f"✅ 已删除: {target['name']}")
            else:
                print("已取消")
        else:
            print("❌ 无效选项")
    except ValueError:
        print("❌ 请输入数字")


def publish_portfolio_to_cloud(portfolio_id: int) -> dict:
    from saas_platform.database.supabase_client import (
        upsert_strategy,
        bulk_upsert_equity_curves,
        is_configured,
        get_client,
    )

    result = {
        'portfolio_id': portfolio_id,
        'strategy_pushed': False,
        'equity_pushed': False,
        'errors': [],
    }

    if not is_configured():
        msg = "SaaS Supabase 未配置，请设置 SAAS_SUPABASE_URL 和 SAAS_SUPABASE_KEY"
        logger.error(msg)
        result['errors'].append(msg)
        return result

    print("\n" + "=" * 80)
    print(f"🚀 发布组合 {portfolio_id} 到云端 SaaS 平台".center(80))
    print("=" * 80)

    print("\n📥 Step 1: 提取本地组合元数据...")
    portfolio = _get_portfolio_info(portfolio_id)
    if not portfolio:
        msg = f"组合 {portfolio_id} 不存在"
        logger.error(msg)
        result['errors'].append(msg)
        return result

    target_symbol = portfolio.get('target_symbol', 'BTC_USDT')
    target_asset = portfolio.get('target_asset', 'crypto')
    weight_mode = portfolio.get('weight_mode', 'equal')
    print(f"  组合名称: {portfolio['name']}")
    print(f"  标的: {target_symbol} ({target_asset})")
    print(f"  权重模式: {weight_mode}")
    print(f"  年化收益: {(portfolio.get('metric_annualized_return', 0) or 0) * 100:.2f}%")
    print(f"  夏普率: {portfolio.get('metric_sharpe', 0) or 0:.2f}")

    db = get_client()
    existing = db.select(
        'saas_strategies',
        columns='id,name,updated_at',
        filters={'name': f"eq.{portfolio['name']}"},
    ) if db else []

    if existing:
        cloud_id = existing[0]['id']
        print(f"  ⚠️ 云端已存在同名策略 (ID: {cloud_id})")
        print(f"     将更新策略代码和净值数据")
        result['cloud_strategy_id'] = cloud_id

    print("\n📥 Step 2: 提取子策略源码...")
    sub_strategies = _get_sub_strategies(portfolio_id)
    if not sub_strategies:
        msg = f"组合 {portfolio_id} 没有有效的子策略"
        logger.error(msg)
        result['errors'].append(msg)
        return result
    print(f"  找到 {len(sub_strategies)} 个子策略:")
    for i, s in enumerate(sub_strategies, 1):
        print(f"    {i}. [{s['dir_id'][:8]}...] {s['strategy_name']} (Sharpe: {s.get('metric_sharpe', 0) or 0:.2f})")

    print("\n☁️ Step 3: 推送策略到云端 saas_strategies...")

    combined_code_parts = []
    required_factors = set()
    for s in sub_strategies:
        combined_code_parts.append({
            'dir_id': s['dir_id'],
            'name': s['strategy_name'],
            'code': s['code_content'],
            'params': json.loads(s['params_json']) if s['params_json'] else {},
            'sharpe': s.get('metric_sharpe'),
        })

    strategy_data = {
        'name': portfolio['name'],
        'description': portfolio.get('description', '') or '',
        'target_asset': target_asset,
        'target_symbol': target_symbol,
        'python_code': json.dumps(combined_code_parts, ensure_ascii=False),
        'params_json': {'weight_mode': weight_mode},
        'required_factors': list(required_factors),
        'timeframe': sub_strategies[0].get('timeframe', '1d'),
        'current_target_position': 0,
        'status': 'LIVE',
        'backtest_sharpe': portfolio.get('metric_sharpe'),
        'backtest_annualized_return': portfolio.get('metric_annualized_return'),
        'backtest_max_drawdown': portfolio.get('metric_max_drawdown'),
        'backtest_start_date': sub_strategies[0].get('metric_start_date'),
        'backtest_end_date': sub_strategies[0].get('metric_end_date'),
    }

    try:
        cloud_strategy = upsert_strategy(strategy_data)
        if cloud_strategy:
            cloud_id = cloud_strategy.get('id', 'unknown')
            result['strategy_pushed'] = True
            result['cloud_strategy_id'] = cloud_id
            action = "更新" if existing else "新建"
            print(f"  ✅ 策略已{action} (云端 ID: {cloud_id})")
        else:
            msg = "upsert_strategy 返回空数据"
            logger.error(msg)
            result['errors'].append(msg)
    except Exception as e:
        msg = f"推送策略失败: {e}"
        logger.error(msg)
        result['errors'].append(msg)

    if not result['strategy_pushed']:
        print(f"  ❌ 策略推送失败，跳过后续步骤")
        return result

    cloud_strategy_id = result.get('cloud_strategy_id')

    print("\n☁️ Step 4: 推送历史回测净值到 saas_equity_curves...")
    nav_records = _get_backtest_nav_records(portfolio_id)
    if nav_records:
        equity_data = []
        for rec in nav_records:
            date_str = rec['date']
            if len(date_str) > 10:
                date_str = date_str[:10]
            equity_data.append({
                'strategy_id': cloud_strategy_id,
                'date': date_str,
                'nav_value': float(rec['nav']) if rec['nav'] is not None else 1.0,
                'is_backtest': True,
            })

        try:
            count = bulk_upsert_equity_curves(equity_data)
            result['equity_pushed'] = True
            result['equity_count'] = count
            print(f"  ✅ 净值曲线已推送 ({count} 条记录)")
        except Exception as e:
            msg = f"推送净值曲线失败: {e}"
            logger.error(msg)
            result['errors'].append(msg)
    else:
        print("  ⚠️ 没有回测净值记录，跳过")

    print("\n" + "=" * 80)
    if result['strategy_pushed'] and result['equity_pushed']:
        print("🎉 发布成功！本地组合已上云".center(80))
    else:
        print("⚠️ 发布部分失败，请检查错误日志".center(80))
    print("=" * 80)

    return result


def select_portfolio() -> int:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT portfolio_id, name, status, target_symbol,
               metric_annualized_return, metric_sharpe
        FROM portfolios
        WHERE status IN ('TESTED', 'PAPER', 'LIVE')
        ORDER BY metric_sharpe DESC NULLS LAST
    ''')
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print("\n❌ 没有可发布的组合（需要 TESTED/PAPER/LIVE 状态）")
        return None

    print("\n" + "=" * 90)
    print("📋 本地可发布的组合列表:")
    print("=" * 90)
    print(f"{'序号':<5} {'ID':<5} {'组合名称':<35} {'状态':<8} {'标的':<12} {'年化收益':>10} {'夏普':>8}")
    print("-" * 90)
    for i, r in enumerate(rows, 1):
        ret = (r['metric_annualized_return'] or 0) * 100
        sharpe = r['metric_sharpe'] or 0
        print(f"{i:<5} {r['portfolio_id']:<5} {r['name'][:33]:<35} {r['status']:<8} {r['target_symbol'] or 'BTC_USDT':<12} {ret:>9.2f}% {sharpe:>8.2f}")
    print("=" * 90)

    while True:
        choice = input("\n请输入序号选择要发布的组合 (q 退出): ").strip()
        if choice.lower() == 'q':
            return None
        try:
            idx = int(choice)
            if 1 <= idx <= len(rows):
                selected = rows[idx - 1]
                print(f"✅ 已选择: [{selected['portfolio_id']}] {selected['name']}")
                return selected['portfolio_id']
            else:
                print("❌ 无效选项")
        except ValueError:
            print("❌ 请输入数字")


def main():
    print("=" * 80)
    print("🚀 策略上云发布工具".center(80))
    print("本地投研系统 → 云端 SaaS 平台".center(80))
    print("=" * 80)

    while True:
        print("\n操作菜单:")
        print("  1. 📋 查看云端已有策略")
        print("  2. 🚀 发布新策略 / 更新已有策略")
        print("  3. 🗑️  删除云端策略")
        print("  q. 退出")

        choice = input("\n请选择操作: ").strip()

        if choice == '1':
            list_cloud_strategies()
        elif choice == '2':
            portfolio_id = select_portfolio()
            if portfolio_id is None:
                continue
            result = publish_portfolio_to_cloud(portfolio_id)
            if result.get('errors'):
                print("\n❌ 错误汇总:")
                for err in result['errors']:
                    print(f"  - {err}")
        elif choice == '3':
            delete_cloud_strategy()
        elif choice.lower() == 'q':
            print("👋 再见！")
            break
        else:
            print("❌ 无效选项")


if __name__ == "__main__":
    main()
