#!/usr/bin/env python3
"""
查询策略排行榜：每个策略方向的最佳版本，按总收益率排序
支持按资产类别筛选
"""

import sys
import os
import sqlite3
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import DB_PATH
from database.db_manager import ASSET_TABLE_MAP

ASSET_LABELS = {
    'crypto': '加密货币',
    'gold': '黄金',
    'oil': '原油',
    'us_stock': '美股',
}

ALL_ASSETS = list(ASSET_TABLE_MAP.keys())


def select_asset():
    print("请选择资产类别：")
    for i, asset in enumerate(ALL_ASSETS, 1):
        label = ASSET_LABELS.get(asset, asset)
        print(f"  [{i}] {label} ({asset})")
    print(f"  [{len(ALL_ASSETS)+1}] 全部资产")
    print()

    while True:
        try:
            choice = input("选择 [1-{}]: ".format(len(ALL_ASSETS)+1)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return None
        if not choice:
            return None
        try:
            idx = int(choice)
            if 1 <= idx <= len(ALL_ASSETS):
                return ALL_ASSETS[idx - 1]
            elif idx == len(ALL_ASSETS) + 1:
                return 'all'
        except ValueError:
            pass
        print("无效输入，请重新选择")


def main():
    print("=" * 120)
    print("📊 策略排行榜 - 按年化超额收益排序 Top 30")
    print("=" * 120)

    if not os.path.exists(DB_PATH):
        print(f"错误：数据库文件不存在: {DB_PATH}")
        return

    selected_asset = select_asset()
    if selected_asset is None:
        return

    if selected_asset == 'all':
        asset_filter = ""
        asset_label = "全部资产"
    else:
        asset_filter = f"AND sd.target_asset = '{selected_asset}'"
        asset_label = ASSET_LABELS.get(selected_asset, selected_asset)

    print(f"\n数据库文件: {DB_PATH}")
    print(f"筛选资产: {asset_label}")
    print()

    try:
        conn = sqlite3.connect(DB_PATH)

        sql = f'''
        SELECT 
            sd.dir_id,
            sd.name,
            sd.description,
            sd.timeframe,
            sd.source,
            sd.target_asset,
            sd.target_symbol,
            sv.metric_sharpe,
            sv.metric_annualized_return,
            sv.metric_excess_annual_return,
            sv.metric_max_drawdown,
            sv.metric_win_rate,
            sv.metric_profit_loss_ratio,
            sv.metric_total_trades,
            sv.metric_total_profit_loss,
            sv.metric_avg_profit_loss_per_trade,
            sv.metric_avg_hold_period,
            sv.metric_return,
            sv.metric_start_date,
            sv.metric_end_date
        FROM strategy_directions sd
        INNER JOIN strategy_versions sv ON sd.best_version_id = sv.ver_id
        WHERE sv.metric_total_trades >= 3
        {asset_filter}
        ORDER BY sv.metric_excess_annual_return DESC
        LIMIT 30
        '''

        df = pd.read_sql_query(sql, conn)
        conn.close()

        if df.empty:
            print(f"没有找到符合条件的策略（{asset_label}，交易次数 >= 3）")
            return

        print(f"共找到 {len(df)} 个策略方向（{asset_label}，交易次数 >= 3）")
        print()

        print("=" * 170)
        print(f"{'排名':<4} {'策略名称':<32} {'标的':<12} {'周期':<8} {'来源':<12} {'年化收益':<12} {'年化超额':<12} {'夏普率':<8} {'最大回撤':<10} {'胜率':<8} {'持仓周期':<10} {'交易次数':<8}")
        print("-" * 170)

        for idx, row in df.iterrows():
            rank = idx + 1
            name = row['name'][:29] + '...' if len(row['name']) > 32 else row['name']
            symbol = row.get('target_symbol', '') or row.get('target_asset', '') or 'N/A'
            timeframe = row['timeframe'] or 'N/A'
            source = row['source'] or 'N/A'
            annualized_return = row['metric_annualized_return']
            excess_return = row['metric_excess_annual_return']
            sharpe = row['metric_sharpe']
            max_drawdown = row['metric_max_drawdown']
            win_rate = row['metric_win_rate']
            avg_hold_period = row['metric_avg_hold_period']
            total_trades = int(row['metric_total_trades']) if pd.notna(row['metric_total_trades']) else 0

            annualized_str = f"{annualized_return*100:+.2f}%" if pd.notna(annualized_return) else "N/A"
            excess_str = f"{excess_return*100:+.2f}%" if pd.notna(excess_return) else "N/A"
            sharpe_str = f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A"
            max_drawdown_str = f"{max_drawdown*100:.2f}%" if pd.notna(max_drawdown) else "N/A"
            win_rate_str = f"{win_rate*100:.1f}%" if pd.notna(win_rate) else "N/A"
            hold_period_str = f"{avg_hold_period:.1f} K" if pd.notna(avg_hold_period) else "N/A"

            if pd.notna(annualized_return) and annualized_return > 0:
                annualized_str = f"\033[92m{annualized_str}\033[0m"
            elif pd.notna(annualized_return) and annualized_return < 0:
                annualized_str = f"\033[91m{annualized_str}\033[0m"

            if pd.notna(excess_return) and excess_return > 0:
                excess_str = f"\033[92m{excess_str}\033[0m"
            elif pd.notna(excess_return) and excess_return < 0:
                excess_str = f"\033[91m{excess_str}\033[0m"

            print(f"{rank:<4} {name:<32} {symbol:<12} {timeframe:<8} {source:<12} {annualized_str:<21} {excess_str:<21} {sharpe_str:<8} {max_drawdown_str:<10} {win_rate_str:<8} {hold_period_str:<10} {total_trades:<8}")

        print("=" * 170)
        print()

        print("📋 策略详细信息：")
        print("=" * 140)

        for idx, row in df.iterrows():
            symbol = row.get('target_symbol', '') or row.get('target_asset', '') or 'N/A'
            asset_name = ASSET_LABELS.get(row.get('target_asset', ''), row.get('target_asset', 'N/A'))
            print(f"\n【第 {idx+1} 名】 {row['name']} ({row['timeframe']}) [{symbol}]")
            print("-" * 140)
            print(f"  策略描述: {row['description']}")
            print(f"  资产类别: {asset_name} | 交易标的: {symbol}")
            print(f"  策略来源: {row['source'] or 'N/A'}")
            print(f"  时间范围: {row['metric_start_date']} 到 {row['metric_end_date']}")
            print()
            print(f"  📈 收益指标:")
            print(f"    总收益率: {row['metric_return']*100:+.2f}%")
            print(f"    年化收益率: {row['metric_annualized_return']*100:+.2f}%")
            print(f"    超额年化收益: {row['metric_excess_annual_return']*100:+.2f}%")
            print(f"    单笔平均盈亏: {row['metric_avg_profit_loss_per_trade']:+.2f}")
            print()
            print(f"  📊 风险指标:")
            print(f"    夏普率: {row['metric_sharpe']:.2f}")
            print(f"    最大回撤: {row['metric_max_drawdown']*100:.2f}%")
            print(f"    胜率: {row['metric_win_rate']*100:.1f}%")
            print(f"    盈亏比: {row['metric_profit_loss_ratio']:.2f}")
            print(f"    平均持仓周期: {row['metric_avg_hold_period']:.1f} 根K线" if pd.notna(row['metric_avg_hold_period']) else "    平均持仓周期: N/A")
            print(f"    总交易次数: {int(row['metric_total_trades'])}")

        print()
        print("=" * 140)

    except Exception as e:
        print(f"查询失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
