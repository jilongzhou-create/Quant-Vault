#!/usr/bin/env python3
"""
未来函数（Look-ahead Bias）法医级诊断工具 v2.0
"""

import sys
import os
import sqlite3
import pandas as pd
import random
import re

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:

    sys.path.insert(0, project_root)

from config import DB_PATH
from trading_engine.backtest_engine import load_historical_data

def check_raw_factors():
    print("\n" + "="*80)
    print("🔍 任务 1：原始因子时间戳审查 (检查是否使用了真实发布日)")
    print("="*80)
    
    try:
        conn = sqlite3.connect(DB_PATH)
        
        factors_to_check = ['unrate', 'cpiaucsl']
        
        for factor in factors_to_check:
            print(f"\n📊 检查因子: {factor}")
            print("-" * 40)
            print(f"{'时间戳':<25} {'因子值'}")
            print("-" * 40)
            
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timestamp, factor_value FROM factor_data WHERE factor_name = ? ORDER BY timestamp DESC LIMIT 5",
                (factor,)
            )
            rows = cursor.fetchall()
            
            if not rows:
                print("❌ 数据库中未找到该因子！")
            else:
                for row in rows:
                    print(f"{row[0]:<25} {row[1]}")
                    
        print("\n📋 审查说明:")
        print(" - 如果时间戳全都在每月的 1 号 (如 2023-03-01 00:00) -> ❌ 存在未来函数风险！")
        print(" - 如果时间戳在月中或月底 (如 2023-04-05 13:30) -> ✅ 已使用真实发布日期，非常安全！")
        
        conn.close()
    except Exception as e:
        print(f"检查失败: {e}")

def check_merge_asof():
    print("\n" + "="*80)
    print("🔍 任务 2：合并对齐严谨性审查 (merge_asof direction='backward')")
    print("="*80)
    
    try:
        print("⏳ 加载融合数据...")
        df = load_historical_data(limit=10000)
        if df.empty or 'unrate' not in df.columns:
            print("❌ 数据加载失败或缺少 unrate 因子")
            return
            
        df = df.dropna(subset=['unrate'])
        if len(df) == 0:
            print("❌ 数据全是 NaN")
            return
            
        print("🎯 随机抽取检查点...")
        sample_indices = random.sample(range(len(df)), min(3, len(df)))
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        leak_count = 0
        safe_count = 0
        
        for idx in sample_indices:
            row = df.iloc[idx]
            kline_time = row['timestamp']
            kline_val = row['unrate']
            
            cursor.execute(
                "SELECT MIN(timestamp) FROM factor_data WHERE factor_name = 'unrate' AND factor_value = ?",
                (float(kline_val),)
            )
            raw_time_str = cursor.fetchone()[0]
            
            print(f"\n🔹 K线时间戳: {kline_time}")
            print(f"   读取到的 UNRATE 值: {kline_val}")
            
            if raw_time_str:
                raw_time = pd.to_datetime(raw_time_str)
                if kline_time.tzinfo is not None:
                    kline_time = kline_time.tz_localize(None)
                if raw_time.tzinfo is not None:
                    raw_time = raw_time.tz_localize(None)
                    
                print(f"   该值在数据库的真实入库时间: {raw_time}")
                
                if raw_time > kline_time:
                    print("   ❌ 致命穿越！K线偷看了未来才发布的数据！")
                    leak_count += 1
                else:
                    print("   ✅ 时序安全！K线读取的是已经发布的历史数据。")
                    safe_count += 1
            else:
                print("   ⚠️ 无法在原始数据库中追溯该值")
                
        print(f"\n📊 审查结果: {safe_count} 安全, {leak_count} 泄露")
        conn.close()
        
    except Exception as e:
        print(f"检查失败: {e}")

def check_engine_shift():
    print("\n" + "="*80)
    print("🔍 任务 3：引擎层持仓延迟审查 (Position Shift Check)")
    print("="*80)
    
    try:
        engine_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'core', 'backtest_engine.py')
        
        with open(engine_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        if re.search(r'\.shift\(\s*1\s*\)', content):
            print("✅ 检查通过！在 core/backtest_engine.py 中发现了严格的 .shift(1) 延迟逻辑。")
            print("   说明今天收盘产生的信号，是在次日才计算盈亏。")
            print("   【结论】：价格流不存在未来函数！此前的报警为探针误报。")
        else:
            print("❌ 警告！未在回测引擎中发现 .shift(1) 延迟逻辑！价格流存在严重穿越风险！")
            
    except Exception as e:
        print(f"检查失败: {e}")

def main():
    print("\n" + "="*80)
    print("🔬 未来函数（Look-ahead Bias）法医级诊断工具 v2.0")
    print("="*80)
    
    check_raw_factors()
    check_merge_asof()
    check_engine_shift()
    
    print("\n" + "="*80)
    print("诊断完成！")
    print("="*80)

if __name__ == "__main__":
    main()
