
#!/usr/bin/env python3
"""
删除 portfolio_daily_records 表中 run_phase = BACKTEST 的记录的脚本
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

if project_root not in sys.path:

    sys.path.insert(0, project_root)

from database.db_manager import DB_PATH
from logger_setup import get_logger
import sqlite3

logger = get_logger(__name__)


def clear_portfolio_daily_records():
    """删除 portfolio_daily_records 表中 run_phase = BACKTEST 的记录"""
    try:
        print("=" * 80)
        print("⚠️  删除 portfolio_daily_records 表中 BACKTEST 记录 ⚠️".center(80))
        print("=" * 80)
        
        # 先查看当前有多少条记录
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 查看各 run_phase 的记录数
        cursor.execute("""
            SELECT run_phase, COUNT(*) 
            FROM portfolio_daily_records 
            GROUP BY run_phase
        """)
        phase_counts = cursor.fetchall()
        
        if not phase_counts:
            print("\n✅ portfolio_daily_records 表已经是空的了！")
            return
        
        print("\n📊 当前表中各阶段记录数:")
        backtest_count = 0
        for phase, count in phase_counts:
            print(f"  - {phase}: {count} 条")
            if phase == 'BACKTEST':
                backtest_count = count
        
        if backtest_count == 0:
            print("\n✅ 没有 BACKTEST 记录需要删除！")
            return
        
        # 确认
        confirm = input(f"\n⚠️  确认要删除 {backtest_count} 条 BACKTEST 记录吗？此操作不可撤销！(输入 'yes' 确认): ").strip().lower()
        
        if confirm != 'yes':
            print("\n❌ 操作已取消")
            return
        
        # 删除 BACKTEST 记录
        cursor.execute("DELETE FROM portfolio_daily_records WHERE run_phase = 'BACKTEST'")
        conn.commit()
        
        print(f"\n✅ 成功删除 {backtest_count} 条 BACKTEST 记录！")
        
        # 验证
        cursor.execute("SELECT run_phase, COUNT(*) FROM portfolio_daily_records GROUP BY run_phase")
        new_phase_counts = cursor.fetchall()
        
        print("\n📊 验证：表中现在各阶段记录数:")
        if new_phase_counts:
            for phase, count in new_phase_counts:
                print(f"  - {phase}: {count} 条")
        else:
            print("  - 表为空")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"删除记录失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    clear_portfolio_daily_records()

