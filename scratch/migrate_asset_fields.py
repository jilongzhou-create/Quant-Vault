#!/usr/bin/env python3
"""
数据迁移脚本：
1. 将 market_data_commodity 中的黄金数据 (GCUSD) 迁移到 market_data_gold
2. 将 market_data_commodity 中的原油数据 (BZUSD/CLUSD) 迁移到 market_data_oil
3. 将 strategy_directions 中的 target_asset 字段统一迁移为新的双字段体系 (target_asset + target_symbol)
4. 将 portfolios 中的 target_asset 字段统一迁移
"""

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from database.db_manager import DB_PATH, ASSET_TABLE_MAP, SYMBOL_ASSET_MAP, init_db
from logger_setup import get_logger
import sqlite3

logger = get_logger(__name__)

GOLD_SYMBOLS = ['GCUSD', 'PAXG_USDT']
OIL_SYMBOLS = ['BZUSD', 'CLUSD']

OLD_TO_NEW_MAP = {
    'crypto':    ('crypto',   'BTC_USDT'),
    'Crypto':    ('crypto',   'BTC_USDT'),
    'BTC_USDT':  ('crypto',   'BTC_USDT'),
    'PAXG_USDT': ('gold',     'PAXG_USDT'),
    'SPY':       ('us_stock', 'SPY'),
    'QQQ':       ('us_stock', 'QQQ'),
    'CLUSD':     ('oil',      'CLUSD'),
    'BZUSD':     ('oil',      'BZUSD'),
    'GCUSD':     ('gold',     'GCUSD'),
    '':          ('crypto',   'BTC_USDT'),
    None:        ('crypto',   'BTC_USDT'),
}


def migrate_commodity_tables():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='market_data_commodity'")
    if not cursor.fetchone():
        logger.info("market_data_commodity 表不存在，跳过迁移")
        conn.close()
        return
    
    cursor.execute("SELECT DISTINCT symbol FROM market_data_commodity")
    symbols = [r[0] for r in cursor.fetchall()]
    logger.info(f"market_data_commodity 中的 symbol: {symbols}")
    
    for symbol in symbols:
        if symbol in GOLD_SYMBOLS:
            target_table = 'market_data_gold'
        elif symbol in OIL_SYMBOLS:
            target_table = 'market_data_oil'
        else:
            logger.warning(f"未知 symbol: {symbol}，跳过")
            continue
        
        cursor.execute(f"SELECT COUNT(*) FROM market_data_commodity WHERE symbol = ?", (symbol,))
        count = cursor.fetchone()[0]
        logger.info(f"迁移 {symbol} ({count} 行) → {target_table}")
        
        cursor.execute(f'''
        INSERT OR IGNORE INTO {target_table} (symbol, timestamp, open, high, low, close, volume, rsi_14, macd, macd_signal, macd_hist)
        SELECT symbol, timestamp, open, high, low, close, volume, rsi_14, macd, macd_signal, macd_hist
        FROM market_data_commodity
        WHERE symbol = ?
        ''', (symbol,))
        
        migrated = cursor.rowcount
        logger.info(f"成功迁移 {migrated} 行 {symbol} → {target_table}")
    
    conn.commit()
    conn.close()
    logger.info("✅ commodity 表迁移完成")


def migrate_strategy_directions():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(strategy_directions)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'target_symbol' not in columns:
        cursor.execute("ALTER TABLE strategy_directions ADD COLUMN target_symbol TEXT DEFAULT 'BTC_USDT'")
        logger.info("已添加 target_symbol 字段到 strategy_directions 表")
    
    cursor.execute("SELECT DISTINCT target_asset FROM strategy_directions")
    old_values = [r[0] for r in cursor.fetchall()]
    logger.info(f"strategy_directions 中现有的 target_asset 值: {old_values}")
    
    for old_val in old_values:
        if old_val in OLD_TO_NEW_MAP:
            new_asset, new_symbol = OLD_TO_NEW_MAP[old_val]
            cursor.execute('''
            UPDATE strategy_directions
            SET target_asset = ?, target_symbol = ?
            WHERE target_asset = ?
            ''', (new_asset, new_symbol, old_val))
            updated = cursor.rowcount
            logger.info(f"target_asset='{old_val}' → ('{new_asset}', '{new_symbol}')，更新 {updated} 行")
        else:
            logger.warning(f"未知的 target_asset 值: '{old_val}'，默认设为 ('crypto', 'BTC_USDT')")
            cursor.execute('''
            UPDATE strategy_directions
            SET target_asset = 'crypto', target_symbol = 'BTC_USDT'
            WHERE target_asset = ?
            ''', (old_val,))
    
    cursor.execute("SELECT DISTINCT target_asset, target_symbol FROM strategy_directions")
    new_values = cursor.fetchall()
    logger.info(f"迁移后 target_asset/target_symbol 分布: {new_values}")
    
    conn.commit()
    conn.close()
    logger.info("✅ strategy_directions 迁移完成")


def migrate_portfolios():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(portfolios)")
    columns = [row[1] for row in cursor.fetchall()]
    
    if 'target_symbol' not in columns:
        cursor.execute("ALTER TABLE portfolios ADD COLUMN target_symbol TEXT DEFAULT 'BTC_USDT'")
        logger.info("已添加 target_symbol 字段到 portfolios 表")
    
    cursor.execute("SELECT DISTINCT target_asset FROM portfolios")
    old_values = [r[0] for r in cursor.fetchall()]
    logger.info(f"portfolios 中现有的 target_asset 值: {old_values}")
    
    for old_val in old_values:
        if old_val in OLD_TO_NEW_MAP:
            new_asset, new_symbol = OLD_TO_NEW_MAP[old_val]
            cursor.execute('''
            UPDATE portfolios
            SET target_asset = ?, target_symbol = ?
            WHERE target_asset = ?
            ''', (new_asset, new_symbol, old_val))
            updated = cursor.rowcount
            logger.info(f"portfolios target_asset='{old_val}' → ('{new_asset}', '{new_symbol}')，更新 {updated} 行")
    
    conn.commit()
    conn.close()
    logger.info("✅ portfolios 迁移完成")


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("=" * 60)
    print("[MIGRATE] 开始数据迁移")
    print("=" * 60)
    
    print("\n[STEP 0] 初始化数据库（创建新表和字段）...")
    init_db()
    
    print("\n[STEP 1] 迁移 commodity 表数据到 gold/oil 表...")
    migrate_commodity_tables()
    
    print("\n[STEP 2] 迁移 strategy_directions 的 target_asset 字段...")
    migrate_strategy_directions()
    
    print("\n[STEP 3] 迁移 portfolios 的 target_asset 字段...")
    migrate_portfolios()
    
    print("\n" + "=" * 60)
    print("[DONE] 数据迁移全部完成！")
    print("=" * 60)
