#!/usr/bin/env python3
"""
SaaS 数据诊断脚本 - 检查数据管道每个环节

用法：python -m ops_scripts.diagnose_saas
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from saas_platform.saas_config import is_configured, get_config_summary, FRED_API_KEY, FMP_API_KEY, COINMETRICS_API_KEY


def main():
    print("=" * 70)
    print("🔍 SaaS 数据管道诊断".center(70))
    print("=" * 70)

    # ── 1. 配置检查 ──
    print("\n📋 Step 1: 配置检查")
    print("-" * 40)
    if not is_configured():
        print("❌ Supabase 未配置！")
        return
    print("✅ Supabase 已连接")
    print(f"  FRED_API_KEY: {'✅ 已配置' if FRED_API_KEY else '❌ 未配置'}")
    print(f"  FMP_API_KEY: {'✅ 已配置' if FMP_API_KEY else '❌ 未配置'}")
    print(f"  COINMETRICS_API_KEY: {'✅ 已配置' if COINMETRICS_API_KEY else '❌ 未配置'}")

    # ── 2. Supabase 数据统计 ──
    print("\n📋 Step 2: Supabase 数据统计")
    print("-" * 40)

    from saas_platform.database.supabase_client import get_client
    db = get_client()
    if not db:
        print("❌ 无法连接 Supabase")
        return

    # 行情数据
    md = db.select('saas_market_data', columns='symbol,timestamp', limit=50000)
    if md:
        from collections import Counter
        md_sym = Counter(r['symbol'] for r in md)
        print(f"  行情数据 (共 {len(md)} 条):")
        for sym, cnt in md_sym.items():
            sym_records = [r for r in md if r['symbol'] == sym]
            dates = sorted([r['timestamp'][:10] for r in sym_records])
            print(f"    {sym}: {cnt} 条 | {dates[0]} ~ {dates[-1]}")
    else:
        print("  ❌ 无行情数据")

    # 因子数据
    fd = db.select('saas_factor_data', columns='symbol,factor_name,timestamp', limit=50000)
    if fd:
        fd_sym = Counter(r['symbol'] for r in fd)
        fd_name = Counter(r['factor_name'] for r in fd)
        print(f"  因子数据 (共 {len(fd)} 条):")
        for sym, cnt in fd_sym.items():
            print(f"    {sym}: {cnt} 条")
        print(f"  因子列表: {', '.join(sorted(fd_name.keys()))}")
    else:
        print("  ❌ 无因子数据")

    # 策略
    strats = db.select('saas_strategies', columns='id,name,status,target_symbol,backtest_start_date,backtest_end_date,backtest_sharpe,current_target_position', limit=100)
    print(f"\n  策略 (共 {len(strats) if strats else 0} 个):")
    if strats:
        for s in strats:
            print(f"    {s.get('name')} | {s.get('status')} | {s.get('target_symbol')} | bt: {s.get('backtest_start_date')} ~ {s.get('backtest_end_date')} | pos: {s.get('current_target_position')}")

    # 净值数据 - 用 RPC 或直接 count
    if strats:
        print(f"\n  净值数据:")
        for s in strats:
            sid = s['id']
            bt_eq = db.select('saas_equity_curves', columns='date,nav_value',
                             filters={'strategy_id': f'eq.{sid}', 'is_backtest': 'eq.true'},
                             order='date.asc', limit=50000)
            lv_eq = db.select('saas_equity_curves', columns='date,nav_value',
                             filters={'strategy_id': f'eq.{sid}', 'is_backtest': 'eq.false'},
                             order='date.asc', limit=50000)
            bt_count = len(bt_eq) if bt_eq else 0
            lv_count = len(lv_eq) if lv_eq else 0
            bt_range = f"{bt_eq[0]['date'][:10]} ~ {bt_eq[-1]['date'][:10]}" if bt_eq else "无"
            lv_range = f"{lv_eq[0]['date'][:10]} ~ {lv_eq[-1]['date'][:10]}" if lv_eq else "无"
            print(f"    {s.get('name')}:")
            print(f"      回测: {bt_count} 条 | {bt_range}")
            print(f"      实盘: {lv_count} 条 | {lv_range}")

    # ── 3. 宽表诊断 ──
    print("\n📋 Step 3: 宽表诊断")
    print("-" * 40)

    if strats:
        from saas_platform.production_engine.signal_engine import CloudSignalEngine
        engine = CloudSignalEngine()
        for s in strats:
            sym = s.get('target_symbol', 'BTC_USDT')
            df = engine._fetch_wide_table(sym)
            if df.empty:
                print(f"  ❌ {sym}: 宽表为空！无法计算信号")
            else:
                print(f"  ✅ {sym}: 宽表 {df.shape[0]} 行 x {df.shape[1]} 列")
                print(f"    日期范围: {df.index.min()} ~ {df.index.max()}")
                print(f"    列名: {list(df.columns[:15])}{'...' if len(df.columns) > 15 else ''}")

                # 检查策略代码需要的列
                full_s = db.select('saas_strategies', columns='python_code', filters={'id': f"eq.{s['id']}"})
                if full_s and full_s[0].get('python_code'):
                    code = full_s[0]['python_code']
                    import json
                    try:
                        subs = json.loads(code)
                        if isinstance(subs, list):
                            for sub in subs:
                                sub_code = sub.get('code', '')
                                # 检查代码中引用的列名
                                import re
                                refs = set(re.findall(r"df\['(\w+)'\]|df\[(\w+)\]|\.(\w+)", sub_code))
                                col_refs = set()
                                for r1, r2, r3 in refs:
                                    for c in [r1, r2, r3]:
                                        if c and c not in ('iloc', 'loc', 'shape', 'copy', 'values', 'index', 'columns', 'dropna', 'fillna', 'astype', 'shift', 'rolling', 'mean', 'std', 'sum', 'abs', 'max', 'min', 'round', 'clip', 'tail', 'head'):
                                            col_refs.add(c)
                                missing = col_refs - set(df.columns)
                                if missing:
                                    print(f"    ⚠️ 子策略 '{sub.get('name', '?')}' 引用了宽表中不存在的列: {missing}")
                                else:
                                    print(f"    ✅ 子策略 '{sub.get('name', '?')}' 列引用检查通过")
                    except (json.JSONDecodeError, TypeError):
                        pass

    # ── 4. 修复建议 ──
    print("\n📋 Step 4: 修复建议")
    print("-" * 40)

    issues = []
    if not FRED_API_KEY:
        issues.append("FRED_API_KEY 未配置 → 宏观因子无法拉取")
    if not FMP_API_KEY:
        issues.append("FMP_API_KEY 未配置 → SPY/QQQ/GCUSD/BZUSD 行情无法拉取")
    if not COINMETRICS_API_KEY:
        issues.append("COINMETRICS_API_KEY 未配置 → 链上因子无法拉取")

    if md:
        md_sym_set = set(r['symbol'] for r in md)
        if 'SPY' not in md_sym_set:
            issues.append("SPY 行情缺失 → 美股策略无法计算")
        if 'QQQ' not in md_sym_set:
            issues.append("QQQ 行情缺失 → 美股策略无法计算")
    else:
        issues.append("无行情数据 → 需要先运行 data_fetcher")

    if fd:
        fd_name_set = set(r['factor_name'] for r in fd)
        if len(fd_name_set) < 5:
            issues.append(f"因子种类过少 ({len(fd_name_set)} 种) → FRED API 可能失败")
    else:
        issues.append("无因子数据 → 需要先运行 data_fetcher")

    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. ⚠️ {issue}")
    else:
        print("  ✅ 未发现明显问题")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
