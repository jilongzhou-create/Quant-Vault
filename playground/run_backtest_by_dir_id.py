#!/usr/bin/env python3
"""
策略方向单次回测工具
功能：
1. 输入 dir_id，从 strategy_directions 表获取策略信息
2. 生成策略代码并 debug
3. 执行回测
4. 创建 .md 文档记录结果
5. 创建临时表记录每日数据
"""

import sys
import os
import json
import sqlite3
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:

    sys.path.insert(0, project_root)

from database.db_manager import (
    DB_PATH,
    get_factor_statistics_summary
)
from agents.base_llm_client import (
    build_init_prompt,
    call_llm,
    check_code_safety
)
from trading_engine.backtest_engine import compile_strategy
from trading_engine.backtest_engine import (
    load_historical_data,
    load_daily_data_directly,
    run_backtest
)
from logger_setup import get_logger

logger = get_logger(__name__)


# 临时表名
TEMP_TABLE_NAME = 'temp_backtest_records'
# 结果文档名
RESULT_FILE = 'backtest_result.md'


def get_strategy_direction_by_id(dir_id):
    """根据 dir_id 获取策略方向信息"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT dir_id, name, description, timeframe
        FROM strategy_directions
        WHERE dir_id = ?
        ''', (dir_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return {
            'dir_id': row[0],
            'name': row[1],
            'description': row[2],
            'timeframe': row[3]
        }
    except Exception as e:
        logger.error(f"获取策略方向失败: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()


def get_strategy_version_by_id(ver_id):
    """根据 ver_id 获取策略版本信息"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT ver_id, dir_id, code_content, params_json, timeframe
        FROM strategy_versions
        WHERE ver_id = ?
        ''', (ver_id,))
        row = cursor.fetchone()
        if not row:
            return None
        # 同时需要获取策略方向的 name 和 description
        dir_id = row[1]
        cursor.execute('''
        SELECT name, description
        FROM strategy_directions
        WHERE dir_id = ?
        ''', (dir_id,))
        dir_row = cursor.fetchone()
        
        return {
            'ver_id': row[0],
            'dir_id': dir_id,
            'name': dir_row[0] if dir_row else f'Version {ver_id}',
            'description': dir_row[1] if dir_row else '',
            'code_content': row[2],
            'params_json': row[3],
            'timeframe': row[4]
        }
    except Exception as e:
        logger.error(f"获取策略版本失败: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()


def init_temp_table():
    """初始化临时表"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 检查临时表是否存在
        cursor.execute(f"""
        SELECT name FROM sqlite_master WHERE type='table' AND name='{TEMP_TABLE_NAME}';
        """)
        if not cursor.fetchone():
            # 创建临时表（与 portfolio_daily_records 结构相同，加 btc_price）
            cursor.execute(f"""
            CREATE TABLE {TEMP_TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id TEXT,
                timestamp DATETIME,
                total_equity REAL,
                daily_return REAL,
                cum_return REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                holdings TEXT,
                signals TEXT,
                run_phase TEXT,
                btc_price REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            logger.info(f"创建临时表 {TEMP_TABLE_NAME} 成功")
        else:
            # 检查是否有 btc_price 列，没有就添加
            cursor.execute(f"PRAGMA table_info({TEMP_TABLE_NAME})")
            columns = [row[1] for row in cursor.fetchall()]
            if 'btc_price' not in columns:
                cursor.execute(f"ALTER TABLE {TEMP_TABLE_NAME} ADD COLUMN btc_price REAL")
                logger.info(f"临时表 {TEMP_TABLE_NAME} 添加 btc_price 列成功")
            
            # 清空临时表
            cursor.execute(f"DELETE FROM {TEMP_TABLE_NAME}")
            logger.info(f"清空临时表 {TEMP_TABLE_NAME} 成功")
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"初始化临时表失败: {e}")


def save_daily_records(records_df):
    """保存每日记录到临时表"""
    try:
        conn = sqlite3.connect(DB_PATH)
        
        # 插入数据
        records_df.to_sql(TEMP_TABLE_NAME, conn, if_exists='append', index=False)
        
        conn.commit()
        conn.close()
        logger.info(f"保存每日记录到 {TEMP_TABLE_NAME} 成功")
    except Exception as e:
        logger.error(f"保存每日记录失败: {e}")


def generate_strategy_code(name, description, timeframe):
    """生成策略代码"""
    try:
        # 获取因子统计特征
        logger.info("获取因子统计特征")
        factor_stats_str = get_factor_statistics_summary()
        
        # 构建提示词
        logger.info("构建提示词")
        prompt = build_init_prompt(name, description, timeframe, factor_stats_str)
        
        # 调用 LLM
        logger.info("调用 LLM 生成代码")
        llm_result = call_llm(prompt)
        
        code_content = llm_result.get('code_content')
        params_json = llm_result.get('params_json')
        
        if not code_content or not params_json:
            raise Exception("LLM 生成失败")
        
        return code_content, params_json
    except Exception as e:
        logger.error(f"生成策略代码失败: {e}")
        raise


def debug_strategy(code_content, params_json, df, timeframe):
    """调试策略并执行回测，返回结果和每日记录"""
    try:
        # 编译策略
        logger.info("编译策略")
        strategy_func = compile_strategy(code_content)
        
        # 静态安检
        logger.info("静态安检")
        is_safe, message = check_code_safety(code_content)
        if not is_safe:
            raise Exception(f"代码安全检查失败: {message}")
        
        # 解析参数
        try:
            params = json.loads(params_json)
        except:
            params = {}
        
        # ============ 复制 backtest_engine 中的回测逻辑，收集每日数据 ============
        from trading_engine.backtest_engine import resample_data, calculate_metrics
        
        # 步骤A: 降采样数据
        resampled_df = resample_data(df, timeframe)
        
        # 步骤B: 获取目标持仓状态
        logger.info(f"调用策略函数，输入数据形状: {resampled_df.shape}")
        signals = strategy_func(resampled_df, params)
        
        # 步骤C: 严谨的向量化盈亏计算
        commission_rate = params.get('commission_rate', 0.0005)
        positions = signals.shift(1).fillna(0)
        returns = resampled_df['close'].pct_change().fillna(0)
        trades = positions.diff().fillna(0).abs()
        costs = trades * commission_rate
        strategy_returns = (positions * returns) - costs
        
        # 步骤D: 构建每日记录DataFrame（只计算临时表需要的字段）
        daily_df = resampled_df.copy()
        daily_df['signals'] = positions
        
        # 计算累计净值和基本指标（仅用于临时表）
        daily_df['strategy_returns'] = strategy_returns
        daily_df['total_equity'] = (1 + daily_df['strategy_returns']).cumprod()
        daily_df['daily_return'] = daily_df['strategy_returns']
        daily_df['cum_return'] = daily_df['total_equity'] - 1
        
        # 准备保存到临时表的记录
        daily_records = daily_df.reset_index()
        if 'timestamp' not in daily_records.columns and 'index' in daily_records.columns:
            daily_records = daily_records.rename(columns={'index': 'timestamp'})
        
        # 只保留需要的字段
        save_records = pd.DataFrame({
            'portfolio_id': 'single_backtest',
            'timestamp': daily_records['timestamp'],
            'total_equity': daily_records['total_equity'],
            'daily_return': daily_records['daily_return'],
            'cum_return': daily_records['cum_return'],
            'holdings': daily_records['signals'].apply(lambda x: json.dumps({'BTC_USDT': float(x)})),
            'signals': daily_records['signals'].apply(lambda x: json.dumps({'BTC_USDT': float(x)})),
            'run_phase': 'BACKTEST',
            'btc_price': daily_records['close']
        })
        
        # 步骤E: 计算详细的性能指标
        new_columns = pd.DataFrame({
            'signals': positions,
            'returns': returns,
            'strategy_returns': strategy_returns,
            'timeframe': timeframe
        }, index=resampled_df.index)
        full_df = pd.concat([resampled_df, new_columns], axis=1)
        metrics = calculate_metrics(full_df)
        metrics['timeframe'] = timeframe
        
        return metrics, save_records
        
    except Exception as e:
        logger.error(f"调试策略失败: {e}")
        raise


def write_result_to_md(dir_id, strategy_info, code_content, params_json, result):
    """写入结果到 Markdown 文件（使用字段名）"""
    try:
        # 清空文件
        with open(RESULT_FILE, 'w', encoding='utf-8') as f:
            f.write('')
        
        # 写入内容
        with open(RESULT_FILE, 'a', encoding='utf-8') as f:
            f.write(f"# Strategy Backtest Result\n\n")
            f.write(f"## Strategy Info\n\n")
            f.write(f"- **dir_id**: {dir_id}\n")
            f.write(f"- **name**: {strategy_info['name']}\n")
            f.write(f"- **description**: {strategy_info['description']}\n")
            f.write(f"- **timeframe**: {strategy_info['timeframe']}\n\n")
            
            f.write(f"## Strategy Code\n\n")
            f.write(f"```python\n{code_content}\n```\n\n")
            
            f.write(f"## Strategy Params\n\n")
            f.write(f"```json\n{params_json}\n```\n\n")
            
            f.write(f"## Backtest Result\n\n")
            for key, value in sorted(result.items()):
                if isinstance(value, float):
                    if 'return' in key or 'drawdown' in key or 'rate' in key and key != 'sharpe_ratio':
                        f.write(f"- **{key}**: {value*100:.4f}%\n")
                    else:
                        f.write(f"- **{key}**: {value:.4f}\n")
                else:
                    f.write(f"- **{key}**: {value}\n")
        
        logger.info(f"结果已写入 {RESULT_FILE}")
    except Exception as e:
        logger.error(f"写入结果文件失败: {e}")


def print_backtest_result(result):
    """打印回测结果（使用 calculate_metrics 返回的字段名）"""
    print("\n" + "=" * 100)
    print("STRATEGY BACKTEST RESULT".center(100))
    print("=" * 100)
    print()
    
    for key, value in sorted(result.items()):
        if isinstance(value, float):
            if 'return' in key or 'drawdown' in key or 'rate' in key and key != 'sharpe_ratio':
                print(f"  {key:<30} {value*100:.4f}%")
            else:
                print(f"  {key:<30} {value:.4f}")
        else:
            print(f"  {key:<30} {value}")
    
    print()
    print("=" * 100)


def main():
    print("=" * 100)
    print("🚀 策略单次回测工具".center(100))
    print("=" * 100)
    
    # 1. 获取输入ID，询问是 dir_id 还是 ver_id
    if len(sys.argv) > 1:
        input_id = sys.argv[1]
        # 如果参数中包含 --ver 或 --dir，优先根据参数判断
        mode = 'dir'
        if '--ver' in sys.argv or len(sys.argv) > 2:
            mode = 'ver'
    else:
        # 交互式询问
        print("\n请选择回测模式:")
        print("1. dir_id 模式（生成新策略代码）")
        print("2. ver_id 模式（使用已有策略代码）")
        choice = input("请输入选项 (1 或 2): ").strip()
        
        if choice == '2':
            input_id = input("请输入要回测的策略版本 ver_id: ").strip()
            mode = 'ver'
        else:
            input_id = input("请输入要回测的策略方向 dir_id: ").strip()
            mode = 'dir'
    
    if not input_id:
        print("❌ ID 不能为空！")
        return
    
    # 2. 根据模式获取数据
    if mode == 'ver':
        print(f"\n📌 正在获取策略版本: {input_id}")
        version_info = get_strategy_version_by_id(input_id)
        
        if not version_info:
            print(f"❌ 未找到 ver_id 为 {input_id} 的策略版本！")
            return
        
        print("\n✅ 策略版本信息:")
        print(f"   ver_id: {version_info['ver_id']}")
        print(f"   dir_id: {version_info['dir_id']}")
        print(f"   名称: {version_info['name']}")
        print(f"   描述: {version_info['description']}")
        print(f"   时间周期: {version_info['timeframe']}")
        
        strategy_info = version_info
        use_existing_code = True
        code_content = version_info['code_content']
        params_json = version_info['params_json']
        dir_id = version_info['dir_id']
    else:
        print(f"\n📌 正在获取策略方向: {input_id}")
        strategy_info = get_strategy_direction_by_id(input_id)
        
        if not strategy_info:
            print(f"❌ 未找到 dir_id 为 {input_id} 的策略方向！")
            return
        
        print("\n✅ 策略方向信息:")
        print(f"   dir_id: {strategy_info['dir_id']}")
        print(f"   名称: {strategy_info['name']}")
        print(f"   描述: {strategy_info['description']}")
        print(f"   时间周期: {strategy_info['timeframe']}")
        
        use_existing_code = False
        code_content = None
        params_json = None
        dir_id = input_id
    
    # 3. 初始化临时表
    print("\n⏳ 初始化临时表...")
    init_temp_table()
    
    # 4. 加载历史数据
    print("\n⏳ 正在加载历史数据...")
    
    try:
        timeframe = strategy_info['timeframe']
        
        if timeframe == '1d':
            df = load_daily_data_directly(symbol="BTC_USDT")
        else:
            df = load_historical_data(symbol="BTC_USDT")
        
        if df.empty:
            print("❌ 历史数据加载失败！")
            return
        
        print(f"✅ 历史数据加载成功，共 {len(df)} 条记录")
        
        # 5. 生成策略代码（如果不是使用已有代码）
        if not use_existing_code:
            print("\n🚀 正在生成策略代码...")
            code_content, params_json = generate_strategy_code(
                strategy_info['name'],
                strategy_info['description'],
                strategy_info['timeframe']
            )
            print("✅ 策略代码生成成功！")
        else:
            print("\n✅ 使用已有策略代码")
        
        # 6. 调试策略并回测
        print("\n🚀 正在调试策略并执行回测...")
        result, daily_records = debug_strategy(code_content, params_json, df, timeframe)
        
        # 7. 打印结果
        print_backtest_result(result)
        
        # 8. 写入结果到 Markdown
        print("\n📝 正在写入结果到 Markdown 文件...")
        write_result_to_md(dir_id, strategy_info, code_content, params_json, result)
        
        # 9. 保存每日记录到临时表
        print("\n💾 正在保存每日记录到临时表...")
        if not daily_records.empty:
            save_daily_records(daily_records)
            print(f"✅ 每日记录已保存到 {TEMP_TABLE_NAME} 表，共 {len(daily_records)} 条")
        else:
            print("⚠️  没有每日记录可保存")
        
        print("\n✅ 任务完成！")
        print(f"   结果文件: {RESULT_FILE}")
        print(f"   临时表: {TEMP_TABLE_NAME}")
        
    except Exception as e:
        logger.error(f"执行过程出错: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n❌ 执行出错: {e}")


if __name__ == "__main__":
    main()