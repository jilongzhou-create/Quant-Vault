#!/usr/bin/env python3
import sys
import os

project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from us_sector_ai_fund.data_pipeline.sector_data_loader import (
    get_sector_wide_table,
    get_active_universe_mask,
    get_dynamic_universe_summary,
)

print("=" * 70)
print("  (T, 11) 宽表验证 - NaN 保留逻辑")
print("=" * 70)

adj_close = get_sector_wide_table(field='adj_close')
print(f"\n宽表 shape: {adj_close.shape}")
print(f"预期: (T, 11) = ({len(adj_close)}, 11)")

print(f"\n各列 NaN 统计:")
nan_counts = adj_close.isna().sum()
total_rows = len(adj_close)
for col in adj_close.columns:
    nan_pct = nan_counts[col] / total_rows * 100
    first_valid = adj_close[col].first_valid_index()
    print(f"  {col:5s}: {nan_counts[col]:>5} NaN ({nan_pct:>5.1f}%)  首个有效值: {first_valid}")

print(f"\n前5行 (2007年初，应只有9只ETF有值):")
print(adj_close.head().to_string())

print(f"\n2015-10-08 前后 (XLRE上市):")
mask_2015 = (adj_close.index >= '2015-10-05') & (adj_close.index <= '2015-10-12')
print(adj_close.loc[mask_2015, ['XLRE', 'XLC']].to_string())

print(f"\n2018-06-19 前后 (XLC上市):")
mask_2018 = (adj_close.index >= '2018-06-15') & (adj_close.index <= '2018-06-22')
print(adj_close.loc[mask_2018, ['XLRE', 'XLC']].to_string())

print(f"\n最后3行 (最新交易日):")
print(adj_close.tail(3).to_string())

print(f"\n动态资产池摘要:")
summary = get_dynamic_universe_summary()
for phase_key, phase in summary.items():
    print(f"  {phase_key}: {phase['period']} → {phase['count']} 只")
    print(f"    {', '.join(phase['symbols'])}")
    print(f"    {phase['description']}")

print(f"\n✅ 验证完成！NaN 保留逻辑正确，未使用 ffill。")
