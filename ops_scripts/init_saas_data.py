#!/usr/bin/env python3
"""
一键初始化 SaaS 云端数据

执行流程：
  Step 1: 拉取最新行情+因子数据到 Supabase
  Step 2: 回填历史实盘净值（从回测结束日期到今天）
  Step 3: 计算今日信号+更新净值
  Step 4: 验证数据完整性

用法：
  python -m ops_scripts.init_saas_data
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from saas_platform.saas_config import is_configured, get_config_summary


def main():
    print("=" * 70)
    print("🚀 SaaS 云端数据初始化".center(70))
    print("=" * 70)

    if not is_configured():
        print("\n❌ Supabase 未配置！请先设置 .env：")
        for k, v in get_config_summary().items():
            print(f"  {k}: {v}")
        return

    print("\n✅ Supabase 已连接")

    # ── Step 1: 拉取数据 ──
    print("\n" + "=" * 70)
    print("📥 Step 1: 拉取行情+因子数据到 Supabase".center(70))
    print("=" * 70)

    try:
        from saas_platform.production_engine.data_fetcher import run_daily_sync
        run_daily_sync()
        print("\n✅ Step 1 完成：数据已拉取到 Supabase")
    except Exception as e:
        print(f"\n❌ Step 1 失败: {e}")
        import traceback
        traceback.print_exc()

    # ── Step 2: 回填历史实盘净值 ──
    print("\n" + "=" * 70)
    print("📊 Step 2: 回填历史实盘净值（从回测结束日期到今天）".center(70))
    print("=" * 70)

    try:
        from saas_platform.production_engine.signal_engine import CloudSignalEngine
        from saas_platform.database.supabase_client import delete_equity_curves, get_public_strategies

        strategies = get_public_strategies()
        for s in strategies:
            sid = s['id']
            sname = s.get('name', 'unknown')
            deleted = delete_equity_curves(sid, is_backtest=False)
            print(f"  🗑️  清除 {sname} 旧实盘净值: {deleted} 条")

        engine = CloudSignalEngine()
        engine.clear_cache()
        result = engine.backfill_historical_nav()
        print(f"\n✅ Step 2 完成：回填了 {result.get('backfilled', 0)} 个策略的历史净值")
    except Exception as e:
        print(f"\n❌ Step 2 失败: {e}")
        import traceback
        traceback.print_exc()

    # ── Step 3: 计算今日信号 ──
    print("\n" + "=" * 70)
    print("🧠 Step 3: 计算今日策略信号".center(70))
    print("=" * 70)

    try:
        from saas_platform.production_engine.signal_engine import CloudSignalEngine
        engine = CloudSignalEngine()
        engine.run_signal_calculation()
        print("\n✅ Step 3 完成：今日信号已更新")
    except Exception as e:
        print(f"\n❌ Step 3 失败: {e}")
        import traceback
        traceback.print_exc()

    # ── Step 4: 验证 ──
    print("\n" + "=" * 70)
    print("🔍 Step 4: 验证数据完整性".center(70))
    print("=" * 70)

    try:
        from saas_platform.database.supabase_client import (
            get_public_strategies,
            get_strategy_equity_curve,
            get_client,
        )

        strategies = get_public_strategies()
        print(f"\n策略数量: {len(strategies)}")

        db = get_client()
        if db:
            md = db.select('saas_market_data', columns='symbol', limit=5000)
            fd = db.select('saas_factor_data', columns='symbol', limit=5000)
            from collections import Counter
            md_sym = Counter(r['symbol'] for r in md) if md else {}
            fd_sym = Counter(r['symbol'] for r in fd) if fd else {}
            print(f"行情数据: {dict(md_sym)}")
            print(f"因子数据: {len(fd)} 条 ({len(fd_sym)} 个标的)")

        for s in strategies:
            sid = s['id']
            name = s.get('name', 'N/A')
            bt = get_strategy_equity_curve(sid, is_backtest=True, limit=10000)
            lv = get_strategy_equity_curve(sid, is_backtest=False, limit=10000)
            bt_range = ""
            lv_range = ""
            if bt:
                bt_dates = sorted([r.get('date', '?')[:10] for r in bt])
                bt_range = f"{bt_dates[0]} ~ {bt_dates[-1]}"
            if lv:
                lv_dates = sorted([r.get('date', '?')[:10] for r in lv])
                lv_range = f"{lv_dates[0]} ~ {lv_dates[-1]}"
            print(f"\n  📈 {name}")
            print(f"     回测净值: {len(bt)} pts | {bt_range or '无'}")
            print(f"     实盘净值: {len(lv)} pts | {lv_range or '无'}")

    except Exception as e:
        print(f"验证失败: {e}")

    print("\n" + "=" * 70)
    print("🎉 初始化完成！运行 streamlit run saas_platform/web_frontend/app.py 查看网站".center(70))
    print("=" * 70)


if __name__ == "__main__":
    main()
