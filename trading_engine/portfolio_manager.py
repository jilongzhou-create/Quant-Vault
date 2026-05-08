
#!/usr/bin/env python3
"""
策略组合工具 - 交互式创建和管理策略组合
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:

    sys.path.insert(0, project_root)

import sqlite3
from database.db_manager import (
    DB_PATH,
    create_portfolio,
    add_portfolio_component,
    get_portfolios_by_status,
    get_portfolio_components,
    SYMBOL_ASSET_MAP
)
from logger_setup import get_logger

logger = get_logger(__name__)

AVAILABLE_SYMBOLS = ['BTC_USDT', 'SPY', 'QQQ', 'GCUSD', 'PAXG_USDT', 'BZUSD']
SYMBOL_LABELS = {
    'BTC_USDT': 'BTC_USDT (加密货币)',
    'SPY': 'SPY (美股大盘)',
    'QQQ': 'QQQ (美股科技)',
    'GCUSD': 'GCUSD (黄金期货)',
    'PAXG_USDT': 'PAXG_USDT (黄金代币)',
    'BZUSD': 'BZUSD (布伦特原油)',
}


def select_target_symbol():
    print("\n" + "=" * 60)
    print("请选择组合标的:")
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


def get_high_sharpe_strategies(target_symbol=None):
    """获取高年化收益率的策略列表（年化 >= 10%），按年化降序排列前80名"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        query = '''
        SELECT 
            sd.dir_id, 
            sd.name, 
            sd.description,
            sv.metric_sharpe, 
            sv.metric_annualized_return,
            sv.metric_total_trades,
            sv.metric_win_rate,
            sv.metric_avg_hold_period
        FROM strategy_directions sd
        INNER JOIN strategy_versions sv ON sd.best_version_id = sv.ver_id
        WHERE sd.best_version_id IS NOT NULL AND sd.best_version_id != ''
          AND sv.run_status != 'OVERFITTED'
          AND sv.metric_total_trades >= 3
          AND sv.metric_sharpe > 0
          AND sv.metric_annualized_return >= 0.10
        '''
        params = []
        
        if target_symbol:
            query += ' AND sd.target_symbol = ?'
            params.append(target_symbol)
        
        query += ' ORDER BY sv.metric_annualized_return DESC LIMIT 80'
        
        cursor.execute(query, params)
        
        rows = cursor.fetchall()
        conn.close()
        
        return rows
    except Exception as e:
        logger.error(f"获取策略列表失败: {e}")
        import traceback
        traceback.print_exc()
        return []


def print_strategies(strategies):
    """打印策略列表（包含策略描述）"""
    print("\n" + "=" * 140)
    print(f"{'序号':<6} {'策略ID':<36} {'名称':<20} {'夏普':<8} {'年化':<10} {'胜率':<8} {'持仓周期':<10} {'交易次数':<8}")
    print("-" * 140)
    
    for idx, (dir_id, name, desc, sharpe, annualized, trades, win_rate, avg_hold) in enumerate(strategies, 1):
        sharpe_str = f"{sharpe:.2f}" if sharpe is not None else "N/A"
        annualized_str = f"{annualized*100:.1f}%" if annualized is not None else "N/A"
        trades_str = str(int(trades)) if trades is not None else "N/A"
        win_rate_str = f"{win_rate*100:.1f}%" if win_rate is not None else "N/A"
        avg_hold_str = f"{avg_hold:.1f}K" if avg_hold is not None else "N/A"
        
        if len(name) > 20:
            name_display = name[:18] + "..."
        else:
            name_display = name
        
        print(f"{idx:<6} {dir_id:<36} {name_display:<20} {sharpe_str:<8} {annualized_str:<10} {win_rate_str:<8} {avg_hold_str:<10} {trades_str:<8}")
        if desc and desc.strip():
            print(f"{'':<6}  描述: {desc}")
        print("-" * 140)
    
    print("=" * 140)


def list_existing_portfolios():
    """列出现有的组合"""
    portfolios = get_portfolios_by_status(['DRAFT', 'TESTED', 'PAPER', 'LIVE'])
    
    if not portfolios:
        print("\n📭 暂无现有组合")
        return
    
    print("\n" + "=" * 80)
    print(f"{'组合ID':<10} {'名称':<25} {'状态':<15} {'创建时间':<20}")
    print("-" * 80)
    
    for p in portfolios:
        print(f"{p['portfolio_id']:<10} {p['name'][:23]:<25} {p['status']:<15} {p['created_at'][:16]:<20}")
    
    print("=" * 80)


def view_portfolio_detail(portfolio_id):
    """查看组合详情"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT portfolio_id, name, description, status, target_symbol, created_at
        FROM portfolios
        WHERE portfolio_id = ?
        ''', (portfolio_id,))
        
        portfolio = cursor.fetchone()
        if not portfolio:
            print("\n❌ 组合不存在")
            return
        
        components = get_portfolio_components(portfolio_id)
        
        print("\n" + "=" * 80)
        print(f"📊 组合详情: {portfolio[1]}")
        print("=" * 80)
        print(f"组合ID: {portfolio[0]}")
        print(f"描述: {portfolio[2] or '无'}")
        print(f"标的: {portfolio[4] or '未指定'}")
        print(f"状态: {portfolio[3]}")
        print(f"创建时间: {portfolio[5]}")
        print(f"\n📈 包含策略数量: {len(components)}")
        
        if components:
            print("\n策略列表:")
            for i, dir_id in enumerate(components, 1):
                cursor.execute('''
                SELECT name FROM strategy_directions WHERE dir_id = ?
                ''', (dir_id,))
                name = cursor.fetchone()
                print(f"  {i}. {name[0] if name else dir_id} ({dir_id})")
        
        print("=" * 80)
        conn.close()
    except Exception as e:
        logger.error(f"查看组合详情失败: {e}")
        print(f"❌ 查看组合详情失败: {e}")


def main():
    print("=" * 100)
    print("🎯 策略组合工具 - 多策略对冲基金管理")
    print("=" * 100)
    
    while True:
        print("\n请选择操作:")
        print("  1. 创建新组合")
        print("  2. 查看现有组合")
        print("  3. 退出")
        
        choice = input("\n请输入选项 (1-3): ").strip()
        
        if choice == '1':
            target_symbol = select_target_symbol()
            
            strategies = get_high_sharpe_strategies(target_symbol=target_symbol)
            
            if not strategies:
                print(f"\n❌ 没有找到 {target_symbol} 的可用策略（需要有 best_version_id 和正夏普率）")
                continue
            
            print(f"\n📊 显示 {target_symbol} 按年化收益率排名前 {len(strategies)} 名的策略：")
            print_strategies(strategies)
            
            portfolio_name = input("\n请输入组合名称 (如 Macro_Blend_01): ").strip()
            if not portfolio_name:
                print("⚠️  组合名称不能为空")
                continue
            
            portfolio_desc = input("请输入组合描述 (可选，直接回车跳过): ").strip()
            
            indices_str = input("\n请输入要入池的策略序号 (逗号分隔，如 1,3,5): ").strip()
            if not indices_str:
                print("⚠️  未输入任何策略序号")
                continue
            
            try:
                indices = []
                for x in indices_str.split(','):
                    x_stripped = x.strip()
                    if x_stripped:
                        indices.append(int(x_stripped))
                
                selected_dir_ids = []
                for i in indices:
                    if 1 <= i and i <= len(strategies):
                        selected_dir_ids.append(strategies[i-1][0])
                
                if not selected_dir_ids:
                    print("⚠️  没有有效的策略序号")
                    continue
                
                portfolio_id = create_portfolio(portfolio_name, portfolio_desc, target_symbol=target_symbol)
                print(f"\n✅ 组合创建成功！组合ID: {portfolio_id}，标的: {target_symbol}")
                
                for dir_id in selected_dir_ids:
                    add_portfolio_component(portfolio_id, dir_id)
                
                print(f"✅ 已成功添加 {len(selected_dir_ids)} 个策略到组合")
                print(f"✅ 组合状态已设置为 DRAFT")
                
            except Exception as e:
                print(f"❌ 操作失败: {e}")
                import traceback
                traceback.print_exc()
        
        elif choice == '2':
            list_existing_portfolios()
            
            portfolio_id_str = input("\n请输入要查看的组合ID (直接回车返回): ").strip()
            if portfolio_id_str:
                try:
                    portfolio_id = int(portfolio_id_str)
                    view_portfolio_detail(portfolio_id)
                except ValueError:
                    print("❌ 无效的组合ID")
        
        elif choice == '3':
            print("\n👋 再见！")
            break
        
        else:
            print("❌ 无效选项，请重新选择")


if __name__ == "__main__":
    main()

