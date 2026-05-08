#!/usr/bin/env python3
"""
批量运行 auto_portfolio_agent 10 次，汇总打印最佳组合
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import sqlite3
from config import DATA_DIR
from database.db_manager import DB_PATH
from agents.auto_portfolio_agent import main as run_agent, AVAILABLE_SYMBOLS, SYMBOL_LABELS

ROUNDS = 10


def select_target_symbol():
    print("\n" + "=" * 60)
    print("请选择挖掘组合的标的:")
    print("=" * 60)
    for i, sym in enumerate(AVAILABLE_SYMBOLS, 1):
        label = SYMBOL_LABELS.get(sym, sym)
        print(f"  {i}. {label}")
    print("=" * 60)

    while True:
        choice = input("\n请输入选项 (1-{}): ".format(len(AVAILABLE_SYMBOLS))).strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(AVAILABLE_SYMBOLS):
                selected = AVAILABLE_SYMBOLS[idx - 1]
                print(f"✅ 已选择: {SYMBOL_LABELS.get(selected, selected)}")
                return selected
            else:
                print("❌ 无效选项")
        except ValueError:
            print("❌ 请输入数字")


def print_ranking(target_symbol: str = None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    symbol_filter = ''
    params = []
    if target_symbol:
        symbol_filter = ' AND target_symbol = ?'
        params.append(target_symbol)

    cursor.execute(f'''
        SELECT portfolio_id, name, weight_mode, target_symbol,
               metric_annualized_return, metric_sharpe, metric_max_drawdown
        FROM portfolios
        WHERE metric_annualized_return IS NOT NULL
        {symbol_filter}
        ORDER BY metric_annualized_return DESC
    ''', params)
    by_return = [dict(r) for r in cursor.fetchall()]

    cursor.execute(f'''
        SELECT portfolio_id, name, weight_mode, target_symbol,
               metric_annualized_return, metric_sharpe, metric_max_drawdown
        FROM portfolios
        WHERE metric_sharpe IS NOT NULL
        {symbol_filter}
        ORDER BY metric_sharpe DESC
    ''', params)
    by_sharpe = [dict(r) for r in cursor.fetchall()]

    conn.close()

    symbol_hint = f" [{target_symbol}]" if target_symbol else ""

    print("\n" + "=" * 90)
    print(f"🏆 年化收益率 TOP 5{symbol_hint}".center(90))
    print("=" * 90)
    print(f"{'排名':<5} {'ID':<5} {'组合名称':<40} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10}")
    print("-" * 90)
    for i, p in enumerate(by_return[:5], 1):
        ret = p['metric_annualized_return'] or 0
        sharpe = p['metric_sharpe'] or 0
        dd = p['metric_max_drawdown'] or 0
        print(f"{i:<5} {p['portfolio_id']:<5} {p['name'][:38]:<40} {ret*100:>9.2f}% {sharpe:>8.2f} {dd*100:>9.2f}%")

    print("\n" + "=" * 90)
    print(f"🏆 夏普率 TOP 5{symbol_hint}".center(90))
    print("=" * 90)
    print(f"{'排名':<5} {'ID':<5} {'组合名称':<40} {'年化收益':>10} {'夏普':>8} {'最大回撤':>10}")
    print("-" * 90)
    for i, p in enumerate(by_sharpe[:5], 1):
        ret = p['metric_annualized_return'] or 0
        sharpe = p['metric_sharpe'] or 0
        dd = p['metric_max_drawdown'] or 0
        print(f"{i:<5} {p['portfolio_id']:<5} {p['name'][:38]:<40} {ret*100:>9.2f}% {sharpe:>8.2f} {dd*100:>9.2f}%")

    print("=" * 90)


def main():
    print("=" * 90)
    print(f"🔄 批量运行 Auto Portfolio Agent × {ROUNDS} 轮".center(90))
    print("=" * 90)

    target_symbol = select_target_symbol()

    for i in range(1, ROUNDS + 1):
        print(f"\n{'█' * 90}")
        print(f"  第 {i}/{ROUNDS} 轮 [{SYMBOL_LABELS.get(target_symbol, target_symbol)}]".center(90))
        print(f"{'█' * 90}")
        try:
            run_agent(target_symbol=target_symbol)
        except Exception as e:
            print(f"\n❌ 第 {i} 轮执行失败: {e}")
            continue

    print_ranking(target_symbol=target_symbol)


if __name__ == "__main__":
    main()
