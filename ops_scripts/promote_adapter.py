#!/usr/bin/env python3
"""
一键转正脚本 - 将 draft_zone 中的草稿适配器迁移到正式环境

功能：
  1. 扫描 draft_zone/ 中的草稿适配器
  2. 自动提取元数据（FACTOR_NAME, TARGET_SYMBOL, 适配器类型）
  3. 将文件迁移到 data_pipeline/adapters/ 并去除 draft_ 前缀
  4. 自动注册到对应的 sync 脚本（sync_crypto_data / sync_commodities_data）
  5. 可选：更新 data_requirements 状态并唤醒冷冻策略
  6. 可选：删除草稿源文件

适配器分类规则：
  - 文件名含 crypto / coin / defi / onchain → sync_crypto_data.py
  - 文件名含 commodit / oil / crude / gold / term → sync_commodities_data.py
  - 文件名含 stock / us_stock / earnings → sync_us_stock_data.py
  - 其他 → sync_crypto_data.py（默认）
"""

import sys
import os
import re
import ast

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logger_setup import get_logger

logger = get_logger("promote_adapter")

SYNC_SCRIPTS = {
    'crypto': 'sync_crypto_data.py',
    'commodities': 'sync_commodities_data.py',
    'us_stock': 'sync_us_stock_data.py',
}

CATEGORY_KEYWORDS = {
    'crypto': ['crypto', 'coin', 'defi', 'onchain', 'binance', 'deribit',
               'defillama', 'coingecko', 'coinmetrics', 'fiat', 'usdt', 'whale',
               'macro_residual', 'cb_buy'],
    'commodities': ['commodit', 'oil', 'crude', 'term_structure',
                    'brent', 'wti'],
    'us_stock': ['stock', 'earnings', 'buyback', 'sp500', 'nasdaq'],
}


def classify_adapter(filename):
    name_lower = filename.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return 'crypto'


def extract_metadata(content):
    meta = {
        'factor_name': None,
        'target_symbol': None,
        'has_fetch_raw': False,
        'has_process_factor': False,
        'resolved_ids': [],
    }

    m = re.search(r"^FACTOR_NAME\s*=\s*['\"](.+?)['\"]", content, re.MULTILINE)
    if m:
        meta['factor_name'] = m.group(1)

    m = re.search(r"^TARGET_SYMBOL\s*=\s*['\"](.+?)['\"]", content, re.MULTILINE)
    if m:
        meta['target_symbol'] = m.group(1)

    if re.search(r'^def fetch_raw\s*\(', content, re.MULTILINE):
        meta['has_fetch_raw'] = True
    if re.search(r'^def process_factor\s*\(', content, re.MULTILINE):
        meta['has_process_factor'] = True

    m = re.search(r'RESOLVED_REQUIREMENT_IDS\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if m:
        ids_str = m.group(1).strip()
        if ids_str:
            for part in ids_str.split(','):
                part = part.strip()
                if part:
                    try:
                        meta['resolved_ids'].append(int(part))
                    except ValueError:
                        pass

    return meta


def strip_draft_prefix(content, draft_filename, formal_filename):
    docstring_match = re.match(r'(""")[\s\S]*?(""")', content)
    if docstring_match:
        old_doc = docstring_match.group(0)
        new_doc = old_doc.replace(draft_filename, formal_filename).replace('草稿', '正式')
        content = content.replace(old_doc, new_doc, 1)
    return content


def register_to_sync_script(adapter_name, category, meta):
    sync_filename = SYNC_SCRIPTS[category]
    sync_path = os.path.join(project_root, 'data_pipeline', sync_filename)

    if not os.path.exists(sync_path):
        print(f"  [WARN] sync script not found: {sync_path}")
        return False

    with open(sync_path, 'r', encoding='utf-8') as f:
        content = f.read()

    import_alias = adapter_name.replace('.py', '')

    if import_alias in content:
        print(f"  [OK] {import_alias} already registered in {sync_filename}")
        return True

    # --- Step 1: Add import ---
    import_added = False
    multi_import_marker = 'from data_pipeline.adapters import ('
    if multi_import_marker in content:
        idx = content.find(multi_import_marker)
        paren_idx = content.find(')', idx)
        if paren_idx != -1:
            import_block = content[idx:paren_idx + 1]
            lines = import_block.split('\n')
            last_import_i = -1
            for i, line in enumerate(lines):
                s = line.strip()
                if s and not s.startswith('#') and s not in ('(', ')'):
                    last_import_i = i
            if last_import_i >= 0:
                lines.insert(last_import_i + 1, f'    {import_alias},')
                new_block = '\n'.join(lines)
                content = content[:idx] + new_block + content[paren_idx + 1:]
                import_added = True
                print(f"  [OK] Added import for {import_alias} in {sync_filename}")
    else:
        # No multi-line import block found; insert one before the if __name__ block
        new_import_block = f'from data_pipeline.adapters import (\n    {import_alias},\n)\n'
        main_marker = "if __name__ == \"__main__\":"
        if main_marker in content:
            idx = content.find(main_marker)
            content = content[:idx] + new_import_block + '\n' + content[idx:]
            import_added = True
            print(f"  [OK] Created import block for {import_alias} in {sync_filename}")
        else:
            print(f"  [WARN] Could not find insertion point for import in {sync_filename}")
            return False

    # --- Step 2: Add sync call ---
    factor_name = meta.get('factor_name') or import_alias
    display_name = import_alias.replace('_', ' ').title()

    if category == 'crypto':
        sync_code = f"""
    # --- {display_name} ---
    logger.info("\\n--- {display_name} ---")
    _sync_legacy_adapter({import_alias}, "{display_name}", '{import_alias}', '{import_alias}', full_sync, results)
"""
        insert_marker = '# ============================================================\n    # 汇总报告'
        if insert_marker in content:
            content = content.replace(insert_marker, sync_code + '\n    ' + insert_marker)
            print(f"  [OK] Added sync block for {import_alias} in {sync_filename}")
        else:
            print(f"  [WARN] Could not auto-insert sync block in {sync_filename}")
            print(f"  Please manually add the sync call for {import_alias}")

    elif category == 'commodities':
        sync_code = f"""
    # --- {display_name} ---
    print(f"\\n--- {display_name} ---")
    try:
        raw_count = {import_alias}.fetch_raw()
        factor_count = {import_alias}.process_factor()
        print(f"    {factor_name}: raw={{raw_count}}, factor={{factor_count}}")
    except Exception as e:
        print(f"    {factor_name} sync failed: {{e}}")
"""
        insert_marker = 'print("\\n=== 大宗商品数据同步完成 ===")'
        if insert_marker in content:
            content = content.replace(insert_marker, sync_code + '\n' + insert_marker)
            print(f"  [OK] Added sync block for {import_alias} in {sync_filename}")
        else:
            print(f"  [WARN] Could not auto-insert sync block in {sync_filename}")

    elif category == 'us_stock':
        sync_code = f"""
    # --- {display_name} ---
    print(f"\\n--- {display_name} ---")
    try:
        raw_count = {import_alias}.fetch_raw()
        factor_count = {import_alias}.process_factor()
        print(f"    {factor_name}: raw={{raw_count}}, factor={{factor_count}}")
    except Exception as e:
        print(f"    {factor_name} sync failed: {{e}}")
"""
        insert_marker = 'print("=" * 80)\n    print("✅ 所有数据同步完成!")'
        if insert_marker in content:
            content = content.replace(insert_marker, sync_code + '\n' + insert_marker)
            print(f"  [OK] Added sync block for {import_alias} in {sync_filename}")
        else:
            print(f"  [WARN] Could not auto-insert sync block in {sync_filename}")

    with open(sync_path, 'w', encoding='utf-8') as f:
        f.write(content)

    return True


def update_data_requirements(resolved_ids):
    if not resolved_ids:
        print("  No RESOLVED_REQUIREMENT_IDS found, skipping data_requirements update")
        return

    from database.db_manager import update_data_requirement_status, awaken_completed_strategies

    try:
        update_data_requirement_status(resolved_ids, 'COMPLETED',
                                       reason='Promoted by promote_adapter')
        print(f"  [OK] Updated {len(resolved_ids)} requirements to COMPLETED: {resolved_ids}")

        awakened = awaken_completed_strategies()
        if awakened:
            print(f"  [OK] Awakened {len(awakened)} strategies: {awakened}")
        else:
            print("  No strategies to awaken")
    except Exception as e:
        print(f"  [ERROR] Failed to update data_requirements: {e}")


def main():
    print("=" * 80)
    print("Adapter Promote Tool (draft_zone -> data_pipeline/adapters)")
    print("=" * 80)

    draft_zone_path = os.path.join(project_root, "draft_zone")
    adapters_path = os.path.join(project_root, "data_pipeline", "adapters")

    if not os.path.exists(draft_zone_path):
        print(f"\n[ERROR] draft_zone not found: {draft_zone_path}")
        return

    draft_files = sorted([
        f for f in os.listdir(draft_zone_path)
        if f.startswith("draft_") and f.endswith(".py")
    ])

    if not draft_files:
        print("\nNo draft adapters found in draft_zone/")
        return

    print(f"\nDraft adapters found ({len(draft_files)}):")
    print("-" * 80)
    for i, filename in enumerate(draft_files, 1):
        draft_path = os.path.join(draft_zone_path, filename)
        with open(draft_path, 'r', encoding='utf-8') as f:
            content = f.read()
        meta = extract_metadata(content)
        category = classify_adapter(filename)
        sync_target = SYNC_SCRIPTS[category]
        formal_name = filename.replace('draft_', '', 1)
        print(f"  {i}. {filename}")
        print(f"     -> {formal_name}  |  factor={meta['factor_name']}  |  symbol={meta['target_symbol']}  |  sync={sync_target}")
    print("-" * 80)

    adapter_name_input = input("\nEnter adapter name to promote (e.g. commodities_term_structure_adapter): ").strip()
    if not adapter_name_input:
        print("\nCancelled.")
        return

    if adapter_name_input.endswith('.py'):
        adapter_name_input = adapter_name_input[:-3]

    expected_draft = f"draft_{adapter_name_input}.py"
    draft_file_path = os.path.join(draft_zone_path, expected_draft)

    if not os.path.exists(draft_file_path):
        print(f"\n[ERROR] Draft file not found: {expected_draft}")
        return

    with open(draft_file_path, 'r', encoding='utf-8') as f:
        draft_code = f.read()

    meta = extract_metadata(draft_code)
    category = classify_adapter(expected_draft)
    formal_filename = expected_draft.replace('draft_', '', 1)
    formal_file_path = os.path.join(adapters_path, formal_filename)

    print(f"\n{'=' * 60}")
    print(f"Promotion Plan:")
    print(f"  Source:   draft_zone/{expected_draft}")
    print(f"  Target:   data_pipeline/adapters/{formal_filename}")
    print(f"  Category: {category}")
    print(f"  Sync:     data_pipeline/{SYNC_SCRIPTS[category]}")
    print(f"  Factor:   {meta['factor_name']}")
    print(f"  Symbol:   {meta['target_symbol']}")
    print(f"  Resolved: {meta['resolved_ids'] or 'None'}")
    print(f"{'=' * 60}")

    confirm = input("\nProceed? (y/N): ").strip().lower()
    if confirm not in ('y', 'yes'):
        print("Cancelled.")
        return

    migrated_code = strip_draft_prefix(draft_code, expected_draft, formal_filename)

    with open(formal_file_path, 'w', encoding='utf-8') as f:
        f.write(migrated_code)
    print(f"\n[OK] Migrated to {formal_file_path}")

    print(f"\nRegistering to {SYNC_SCRIPTS[category]}...")
    register_to_sync_script(adapter_name_input, category, meta)

    if meta['resolved_ids']:
        print(f"\nUpdating data_requirements...")
        update_data_requirements(meta['resolved_ids'])

    delete_draft = input("\nDelete draft source file? (y/N): ").strip().lower()
    if delete_draft in ('y', 'yes'):
        os.remove(draft_file_path)
        print(f"[OK] Deleted {expected_draft}")
    else:
        print(f"Draft file kept: {expected_draft}")

    print("\n" + "=" * 80)
    print("Promotion complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
