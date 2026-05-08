#!/usr/bin/env python3
"""TLT 数据探查脚本 - 扫描现有数据，识别缺失的关键序列"""

import sqlite3
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import DB_PATH

def main():
    import logging
    logging.disable(logging.CRITICAL)

    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()

    print("=" * 70)
    print("  TLT Data Exploration Report")
    print("=" * 70)

    # 1. raw_data 中的 FRED 系列
    c.execute("SELECT DISTINCT source_id FROM raw_data WHERE source_id LIKE 'fred_%' ORDER BY source_id")
    fred_series = [r[0] for r in c.fetchall()]
    print(f"\n[1] FRED series in raw_data: {len(fred_series)}")
    for s in fred_series:
        c.execute("SELECT COUNT(*), MIN(event_timestamp), MAX(event_timestamp) FROM raw_data WHERE source_id=?", (s,))
        cnt, mn, mx = c.fetchone()
        print(f"  {s:<35s}  {cnt:>6d} rows  {str(mn)[:10]} ~ {str(mx)[:10]}")

    # 2. factor_data 中的 MACRO 因子
    c.execute("SELECT DISTINCT factor_name FROM factor_data WHERE symbol='MACRO' ORDER BY factor_name")
    macro_factors = [r[0] for r in c.fetchall()]
    print(f"\n[2] MACRO factors in factor_data: {len(macro_factors)}")
    for f in macro_factors:
        c.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM factor_data WHERE symbol='MACRO' AND factor_name=?", (f,))
        cnt, mn, mx = c.fetchone()
        print(f"  {f:<35s}  {cnt:>6d} rows  {str(mn)[:10]} ~ {str(mx)[:10]}")

    # 3. TLT 行情数据
    c.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM market_data_tlt")
    cnt, mn, mx = c.fetchone()
    print(f"\n[3] TLT market data: {cnt} rows, {str(mn)[:10]} ~ {str(mx)[:10]}")

    # 4. 检查关键缺失数据
    critical_tlt_series = {
        'MOVE': 'ICE BofA US Move Index (债市VIX)',
        'SOFR': '担保隔夜融资利率',
        'FOMC_SENTIMENT': 'FOMC 鹰鸽情绪得分',
    }
    print(f"\n[4] Critical TLT data availability:")
    for series_id, desc in critical_tlt_series.items():
        fred_key = f'fred_{series_id}'
        c.execute("SELECT COUNT(*) FROM raw_data WHERE source_id=?", (fred_key,))
        raw_cnt = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM factor_data WHERE symbol='MACRO' AND factor_name=?", (series_id.lower(),))
        factor_cnt = c.fetchone()[0]
        status = "✅" if (raw_cnt > 0 or factor_cnt > 0) else "❌ MISSING"
        print(f"  {series_id:<20s} {desc:<30s}  raw={raw_cnt}, factor={factor_cnt}  {status}")

    # 5. 检查 TLT 可用的高价值因子
    high_value = [
        'bamlh0a0hym2', 'vixcls', 'gvzcls', 'dgs10', 'dgs2', 'dgs30',
        'dfii10', 'dfii5', 't10y2y', 't10y3m', 'nfci', 'stlfsi4',
        'dtb3', 'dff', 'sofr', 'rrpontsyd',
        'emvmacrointerest', 'emvmacrobroad', 'usepuindxd',
        'bamlc0a4cbbb', 'bamlh0a3hyc',
    ]
    print(f"\n[5] High-value TLT factor availability:")
    for f in high_value:
        c.execute("SELECT COUNT(*) FROM factor_data WHERE symbol='MACRO' AND factor_name=?", (f,))
        factor_cnt = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM raw_data WHERE source_id=?", (f'fred_{f.upper()}',))
        raw_cnt = c.fetchone()[0]
        status = "✅" if (raw_cnt > 0 or factor_cnt > 0) else "❌"
        print(f"  {f:<25s}  factor={factor_cnt:>5d}  raw={raw_cnt:>5d}  {status}")

    # 6. 检查 fomc_sentiment (非 FRED 来源)
    c.execute("SELECT DISTINCT source_id FROM raw_data WHERE source_id LIKE '%fomc%' OR source_id LIKE '%sentiment%'")
    fomc_rows = c.fetchall()
    print(f"\n[6] FOMC/Sentiment data:")
    if fomc_rows:
        for r in fomc_rows:
            c.execute("SELECT COUNT(*), MIN(event_timestamp), MAX(event_timestamp) FROM raw_data WHERE source_id=?", (r[0],))
            cnt, mn, mx = c.fetchone()
            print(f"  {r[0]:<35s}  {cnt:>6d} rows  {str(mn)[:10]} ~ {str(mx)[:10]}")
    else:
        print("  ❌ No FOMC sentiment data found")

    # 7. 检查 TLT volume 数据
    c.execute("SELECT COUNT(*), AVG(volume), MAX(volume) FROM market_data_tlt WHERE volume IS NOT NULL AND volume > 0")
    vol_cnt, vol_avg, vol_max = c.fetchone()
    print(f"\n[7] TLT volume data: {vol_cnt} rows with volume, avg={vol_avg:.0f}, max={vol_max:.0f}")

    conn.close()
    print(f"\n{'='*70}")
    print("  Exploration Complete!")
    print(f"{'='*70}")


if __name__ == '__main__':
    import io
    import sys as _sys
    buf = io.StringIO()
    _old_stdout = _sys.stdout
    _sys.stdout = buf
    try:
        main()
    finally:
        _sys.stdout = _old_stdout
    output = buf.getvalue()
    report_path = os.path.join(os.path.dirname(__file__), 'data_exploration_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(output)
    print(output)
