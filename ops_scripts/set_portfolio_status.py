#!/usr/bin/env python3
"""
组合状态管理脚本 - 用于更改组合状态（TESTED → PAPER/LIVE）
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:

    sys.path.insert(0, project_root)

from database.db_manager import (
    get_portfolios_by_status,
    update_portfolio_status
)


def main():
    print("\n" + "=" * 80)
    print("组合状态管理")
    print("=" * 80)
    
    # 获取所有非 LIVE 状态的组合
    all_portfolios = []
    for status in ['DRAFT', 'TESTED', 'PAPER']:
        portfolios = get_portfolios_by_status([status])
        all_portfolios.extend(portfolios)
    
    if not all_portfolios:
        print("\n❌ 没有找到可用的组合")
        return
    
    # 排序
    all_portfolios.sort(key=lambda x: x['portfolio_id'])
    
    print(f"\n{'ID':<6} {'组合名称':<30} {'当前状态':<15} {'创建时间':<20}")
    print("-" * 80)
    
    for p in all_portfolios:
        print(f"{p['portfolio_id']:<6} {p['name'][:28]:<30} {p['status']:<15} {p['created_at'][:16]:<20}")
    
    print("=" * 80)
    
    # 选择组合
    while True:
        choice = input("\n请输入要修改的组合 ID: ").strip()
        try:
            portfolio_id = int(choice)
            selected = next((p for p in all_portfolios if p['portfolio_id'] == portfolio_id), None)
            if selected:
                break
            else:
                print("❌ 无效的组合 ID，请重新输入")
        except ValueError:
            print("❌ 请输入有效的数字 ID")
    
    # 选择新状态
    print("\n请选择新状态:")
    print("  1. TESTED (已回测)")
    print("  2. PAPER (模拟盘)")
    print("  3. LIVE (实盘 - 谨慎！)")
    
    while True:
        choice = input("\n请输入选项 (1-3): ").strip()
        if choice == '1':
            new_status = 'TESTED'
            break
        elif choice == '2':
            new_status = 'PAPER'
            break
        elif choice == '3':
            confirm = input("\n⚠️  WARNING: 你确定要进入实盘交易吗？请输入 'YES' 确认: ").strip()
            if confirm == 'YES':
                new_status = 'LIVE'
                break
            else:
                print("已取消")
                return
        else:
            print("❌ 无效选项，请重新输入")
    
    # 执行更新
    success = update_portfolio_status(portfolio_id, new_status)
    
    if success:
        print(f"\n✅ 组合 {portfolio_id} 状态已更新为 {new_status}")
    else:
        print(f"\n❌ 更新失败")


if __name__ == "__main__":
    main()
