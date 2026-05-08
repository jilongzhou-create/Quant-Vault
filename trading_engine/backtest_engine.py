import sqlite3
import numpy as np
import pandas as pd
import os
import sys
from logger_setup import get_logger
from config import DB_PATH
from database.db_manager import ASSET_TABLE_MAP
import json
from datetime import datetime

# 【性能与内存优化】：限制 OpenBLAS 线程数以防止 Windows 上的内存分配失败 (10054/OOM)
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# 初始化日志记录器
logger = get_logger("backtest_engine")

def load_historical_data(symbol="BTC_USDT", batch_size=100000, limit=None, target_asset="crypto", factor_names=None):
    """
    从数据库加载历史K线数据并融合因子数据（支持分批加载）
    
    参数:
        symbol (str): 交易对符号，默认为"BTC_USDT"
        batch_size (int): 每批加载的数据条数，默认为100000
        limit (int): 限制加载的数据总条数，默认为None（加载所有数据）
        target_asset (str): 目标资产类别 (crypto/gold/oil/us_stock)
        factor_names (list): 指定只加载这些因子名。None 表示加载全部（可能很慢）
    
    返回:
        pd.DataFrame: 包含历史K线数据和因子数据的DataFrame，确保timestamp为datetime格式，价格为float
    """
    try:
        logger.info(f"开始加载历史数据，交易对: {symbol}, 资产类别: {target_asset}")
        
        table_name = ASSET_TABLE_MAP.get(target_asset, 'market_data_crypto')
        
        # 连接数据库
        conn = sqlite3.connect(DB_PATH)
        
        # 获取数据总量
        count_query = f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?"
        total_rows = conn.execute(count_query, (symbol,)).fetchone()[0]
        logger.info(f"数据总量: {total_rows} 条")
        
        # 确定实际需要加载的数据量
        if limit is not None and limit < total_rows:
            total_rows = limit
            logger.info(f"限制加载数据量为: {total_rows} 条")
        
        # 分批加载数据
        all_data = []
        offset = 0
        
        while offset < total_rows:
            current_batch_size = min(batch_size, total_rows - offset)
            logger.info(f"加载批次: 从 {offset} 到 {offset + current_batch_size}")
            
            # 构建SQL查询语句
            query = f"""
            SELECT * FROM {table_name} 
            WHERE symbol = ? 
            ORDER BY timestamp ASC 
            LIMIT ? OFFSET ?
            """
            
            # 执行查询并加载到DataFrame
            batch_df = pd.read_sql_query(query, conn, params=(symbol, current_batch_size, offset))
            if not batch_df.empty:
                all_data.append(batch_df)
            
            offset += current_batch_size
        
        if not all_data:
            logger.warning(f"未找到 {symbol} 的历史数据 (资产类别: {target_asset})")
            conn.close()
            return pd.DataFrame()
        
        df = pd.concat(all_data, ignore_index=True)
        
        # 确保timestamp为datetime格式
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 确保价格相关列为float32类型以节省内存
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].astype('float32')
        
        # 数据融合：加载因子数据
        if not df.empty:
            logger.info("开始融合因子数据")
            
            if factor_names:
                placeholders = ','.join(['?'] * len(factor_names))
                factor_query = f"""
                SELECT timestamp, factor_name, factor_value 
                FROM factor_data 
                WHERE (symbol = ? OR symbol = 'MACRO')
                  AND factor_name IN ({placeholders})
                ORDER BY timestamp ASC
                """
                factor_df = pd.read_sql_query(factor_query, conn, params=[symbol] + list(factor_names))
                logger.info(f"指定加载 {len(factor_names)} 个因子: {factor_names}")
            else:
                factor_query = """
                SELECT timestamp, factor_name, factor_value 
                FROM factor_data 
                WHERE symbol = ? OR symbol = 'MACRO'
                ORDER BY timestamp ASC
                """
                factor_df = pd.read_sql_query(factor_query, conn, params=(symbol,))
            
            if not factor_df.empty:
                logger.info(f"加载因子数据成功，共 {len(factor_df)} 条")
                
                factor_df['timestamp'] = pd.to_datetime(factor_df['timestamp'], format='mixed')
                factor_df = factor_df.drop_duplicates(subset=['timestamp', 'factor_name'], keep='last')
                
                factor_df['factor_value'] = factor_df['factor_value'].astype('float32')
                
                pivot_df = factor_df.pivot_table(
                    index='timestamp', columns='factor_name',
                    values='factor_value', aggfunc='last'
                )
                pivot_df = pivot_df.sort_index().ffill()
                pivot_df = pivot_df.reset_index()

                df.attrs['_factor_df_cache'] = pivot_df
                logger.info(f"因子数据已挂载至 df 私有属性中 (形状: {pivot_df.shape})，等待降采样后融合...")
            else:
                logger.info("没有找到因子数据")
        
        # 关闭数据库连接
        conn.close()
        
        # 数据验证
        logger.info("数据验证...")
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"价格范围: 最小值={df['close'].min():.2f}, 最大值={df['close'].max():.2f}")
        logger.info(f"价格变化范围: 最小值={df['close'].pct_change().min():.4f}, 最大值={df['close'].pct_change().max():.4f}")
        
        logger.info(f"数据加载完成，共 {len(df)} 条记录")
        return df
    except Exception as e:
        logger.error(f"加载历史数据时出错: {e}")
        raise

def compile_strategy(code_string):
    """
    编译策略代码字符串，提取generate_signals函数
    
    参数:
        code_string (str): 包含generate_signals函数的Python代码字符串
    
    返回:
        function: 提取的generate_signals函数
    
    异常:
        ValueError: 如果代码字符串中没有找到generate_signals函数
    """
    try:
        logger.info("开始编译策略代码")
        
        # 创建命名空间，预导入常用库
        namespace = {}
        import pandas as pd
        import numpy as np
        namespace['pd'] = pd
        namespace['np'] = np
        
        # 执行代码字符串，将变量和函数加载到命名空间
        exec(code_string, globals(), namespace)
        
        # 提取generate_signals函数
        strategy_func = namespace.get('generate_signals')
        
        if strategy_func is None:
            error_msg = "策略代码中未找到generate_signals函数"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("策略代码编译成功")
        return strategy_func
    except Exception as e:
        logger.error(f"编译策略代码时出错: {e}")
        raise

def resample_data(df, timeframe):
    """
    对数据进行降采样
    
    参数:
        df (pd.DataFrame): 包含历史K线数据的DataFrame
        timeframe (str): 时间周期，如 '1h', '1d'
    
    返回:
        pd.DataFrame: 降采样后的数据
    """
    try:
        logger.info(f"开始降采样数据，时间周期: {timeframe}")
        logger.info(f"原始数据列数: {len(df.columns)}, 列名: {df.columns.tolist()}")
        
        # 兼容 Pandas 的 offset 别名 (将小写 d 转为 D, m 转为 T，h 保持不变)
        pandas_tf = timeframe.replace('d', 'D').replace('m', 'T')
        
        # 构建聚合字典（在设置index之前）
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # 添加技术指标列（取最后一个值）
        tech_indicators = ['rsi_14', 'macd', 'macd_signal', 'macd_hist']
        for col in tech_indicators:
            if col in df.columns:
                agg_dict[col] = 'last'
        
        # 添加所有因子数据列（取最后一个值）
        # 自动检测所有不是基础量价和技术指标的列作为因子列
        base_columns = ['open', 'high', 'low', 'close', 'volume', 'rsi_14', 'macd', 'macd_signal', 'macd_hist', 'symbol', 'timestamp']
        factor_columns = []
        for col in df.columns:
            if col not in base_columns and col not in agg_dict:
                agg_dict[col] = 'last'
                factor_columns.append(col)
        
        logger.info(f"检测到 {len(factor_columns)} 个因子列: {factor_columns}")
        logger.info(f"聚合字典包含 {len(agg_dict)} 个列")
        
        # 【究极内存架构优化：延迟融合执行】
        # 在 resample 之前，先将因子缓存从 attrs 中取出并清除，
        # 避免 concat 时 attrs 中的 DataFrame 触发 "ambiguous truth value" 错误
        factor_df = df.attrs.pop('_factor_df_cache', None)
        
        # 确保DataFrame的index是timestamp
        if df.index.name != 'timestamp':
            # 检查是否有timestamp列
            if 'timestamp' in df.columns:
                # 转换为datetime格式并设置为index
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                raise ValueError("DataFrame中没有timestamp列")
        
        # 【内存优化】：分批次进行聚合，防止 pandas 在 resample 后对 112 列进行全量 consolidate 导致 OOM
        resampler = df.resample(pandas_tf)
        all_resampled_parts = []
        cols = list(agg_dict.keys())
        batch_size = 20  # 每批处理 20 列，约消耗 200MB 内存
        
        for i in range(0, len(cols), batch_size):
            batch_cols = cols[i:i+batch_size]
            batch_agg = {c: agg_dict[c] for c in batch_cols}
            # 只选取当前批次的列进行聚合，避免 1.1GB 的全矩阵 consolidate
            resampled_batch = resampler[batch_cols].agg(batch_agg)
            all_resampled_parts.append(resampled_batch)
            
        resampled_df = pd.concat(all_resampled_parts, axis=1)
        
        # 剔除无交易的空档期（只根据close列判断）
        resampled_df = resampled_df.dropna(subset=['close'])
        
        # 补回被 resample 聚合时丢弃的基础 symbol 列
        if 'symbol' in df.columns:
            resampled_df['symbol'] = df['symbol'].iloc[0] if len(df) > 0 else None
            
        # 【究极内存架构优化：延迟融合执行】
        # 使用之前从 attrs 中取出的因子缓存
        if factor_df is not None:
            logger.info("检测到挂载的因子数据，开始与降采样后的 K 线进行极速轻量级融合...")
            
            # 将 timestamp 从 index 中释放出来用于对齐
            resampled_df = resampled_df.reset_index()
            resampled_df = resampled_df.sort_values('timestamp')
            factor_df = factor_df.sort_values('timestamp')
            
            resampled_df = pd.merge_asof(
                resampled_df,
                factor_df,
                on='timestamp',
                direction='backward'
            )
            resampled_df.ffill(inplace=True)
            resampled_df.fillna(0, inplace=True)
            
            # 重新设为 index
            resampled_df.set_index('timestamp', inplace=True)
            logger.info("延迟融合因子数据完成！")
            
        logger.info(f"处理最终完成，降采样后行数: {len(resampled_df)}, 列数: {len(resampled_df.columns)}")
        return resampled_df
    except Exception as e:
        logger.error(f"降采样数据时出错: {e}")
        raise

def load_resampled_data(symbol="BTC_USDT", target_asset="crypto", timeframe="1d"):
    """
    极速、极低内存加载指定时间周期的数据。
    仅加载 OHLCV 6列，在内存中降采样后计算技术指标并融合宏观因子。
    内存占用相比 load_historical_data 降低 99%+。
    
    参数:
        symbol (str): 交易对符号
        target_asset (str): 目标资产类别
        timeframe (str): 目标时间周期，如 '1d', '4h', '1h'
    
    返回:
        pd.DataFrame: 降采样后的数据，timestamp 为 index
    """
    try:
        logger.info(f"极速加载数据: {symbol}, 资产: {target_asset}, 周期: {timeframe}")
        table_name = ASSET_TABLE_MAP.get(target_asset, 'market_data_crypto')
        conn = sqlite3.connect(DB_PATH)
        
        logger.info("1. 读取基础 K 线...")
        query = f"SELECT timestamp, open, high, low, close, volume FROM {table_name} WHERE symbol = ?"
        df = pd.read_sql_query(query, conn, params=(symbol,))
        
        if df.empty:
            logger.error("基础数据为空")
            conn.close()
            return pd.DataFrame()
            
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        logger.info(f"2. 降采样到 {timeframe} (原始数据 {len(df)} 条)...")
        pandas_tf = timeframe.replace('d', 'D').replace('m', 'T')
        df_resampled = df.resample(pandas_tf).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna(subset=['close'])
        
        logger.info(f"降采样完成，共 {len(df_resampled)} 条。开始计算技术指标...")
        
        del df
        
        delta = df_resampled['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss
        df_resampled['rsi_14'] = 100 - (100 / (1 + rs))
        
        exp1 = df_resampled['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_resampled['close'].ewm(span=26, adjust=False).mean()
        df_resampled['macd'] = exp1 - exp2
        df_resampled['macd_signal'] = df_resampled['macd'].ewm(span=9, adjust=False).mean()
        df_resampled['macd_hist'] = df_resampled['macd'] - df_resampled['macd_signal']
        
        df_resampled = df_resampled.reset_index()
        
        logger.info("3. 获取并融合宏观因子...")
        factor_query = "SELECT timestamp, factor_name, factor_value FROM factor_data WHERE symbol = ? OR symbol = 'MACRO' ORDER BY timestamp ASC"
        factor_df = pd.read_sql_query(factor_query, conn, params=(symbol,))
        
        if not factor_df.empty:
            factor_df['timestamp'] = pd.to_datetime(factor_df['timestamp'], format='mixed')
            factor_df = factor_df.drop_duplicates(subset=['timestamp', 'factor_name'], keep='last')
            factor_df['factor_value'] = factor_df['factor_value'].astype('float32')
            
            all_timestamps = factor_df['timestamp'].unique()
            pivot_dict = {'timestamp': all_timestamps}
            
            for fn in factor_df['factor_name'].unique():
                sub = factor_df[factor_df['factor_name'] == fn]
                series = sub.set_index('timestamp')['factor_value']
                pivot_dict[fn] = series.reindex(all_timestamps).values
            
            pivot_df = pd.DataFrame(pivot_dict).sort_values('timestamp').ffill()
            
            df_resampled = pd.merge_asof(
                df_resampled,
                pivot_df,
                on='timestamp',
                direction='backward'
            )
            
        df_resampled.ffill(inplace=True)
        df_resampled.fillna(0, inplace=True)
        
        df_resampled['symbol'] = symbol
        df_resampled.set_index('timestamp', inplace=True)
        
        conn.close()
        logger.info(f"极速加载完成！最终数据形状: {df_resampled.shape}")
        return df_resampled
        
    except Exception as e:
        logger.error(f"极速加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def load_daily_data_directly(symbol="BTC_USDT", target_asset="crypto"):
    return load_resampled_data(symbol=symbol, target_asset=target_asset, timeframe="1d")

def calculate_metrics(df):
    """
    计算回测性能指标（状态机终极版：完美处理未平仓、多空反转、复利对齐）
    """
    try:
        import numpy as np
        logger.info("=== 开始计算回测指标 (状态机终极版) ===")
        
        timeframe = df['timeframe'].iloc[0] if 'timeframe' in df.columns else '1d'
        start_datetime = df.index.min()
        end_datetime = df.index.max()
        
        days_in_backtest = max((end_datetime - start_datetime).total_seconds() / 86400, 1.0)
        years_in_backtest = days_in_backtest / 365.25
        
        strategy_returns = df['strategy_returns'].fillna(0)
        market_returns = df['returns'].fillna(0)
        
        total_return = (1 + strategy_returns).prod() - 1
        market_total_return = (1 + market_returns).prod() - 1
        
        cumulative_strategy = (1 + strategy_returns).cumprod()
        peak = cumulative_strategy.expanding(min_periods=1).max()
        max_drawdown = ((cumulative_strategy - peak) / peak).min() if len(cumulative_strategy) > 0 else 0
        
        # === 状态机：精准逐笔交易统计 ===
        positions_array = df['signals'].fillna(0).values
        returns_array = strategy_returns.values
        
        trades_list = []
        current_pos = 0
        trade_ret = 1.0  
        trade_bars = 0   
        
        for pos, ret in zip(positions_array, returns_array):
            if current_pos == 0:
                if pos != 0: # 发生开仓 (0 -> 1 或 0 -> -1)
                    current_pos = pos
                    trade_ret = 1 + ret
                    trade_bars = 1
            else:
                if pos == current_pos: # 持续持仓中
                    trade_ret *= (1 + ret)
                    trade_bars += 1
                elif pos == 0: # 发生平仓 (1 -> 0 或 -1 -> 0)
                    trade_ret *= (1 + ret)
                    trades_list.append({'profit': trade_ret - 1, 'bars': trade_bars})
                    current_pos = 0
                    trade_ret = 1.0
                    trade_bars = 0
                else: # 多空反转 (1 -> -1 或 -1 -> 1)
                    trade_ret *= (1 + ret)
                    trades_list.append({'profit': trade_ret - 1, 'bars': trade_bars})
                    current_pos = pos
                    trade_ret = 1.0
                    trade_bars = 0
                    
        # 【关键修复】处理回测结束时，仍未平仓的最后一笔交易
        if current_pos != 0:
            trades_list.append({'profit': trade_ret - 1, 'bars': trade_bars})
            
        # === 汇总交易统计 ===
        total_trades = len(trades_list)
        win_rate, win_loss_ratio, avg_profit_loss_per_trade, avg_hold_period = 0.0, 0.0, 0.0, 0.0
        
        if total_trades > 0:
            wins = [t['profit'] for t in trades_list if t['profit'] > 0]
            losses = [t['profit'] for t in trades_list if t['profit'] < 0]
            
            win_rate = len(wins) / total_trades
            avg_win = sum(wins) / len(wins) if wins else 0
            avg_loss = abs(sum(losses) / len(losses)) if losses else 0
            
            win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else (999.0 if avg_win > 0 else 0.0)
            avg_profit_loss_per_trade = sum(t['profit'] for t in trades_list) / total_trades
            avg_hold_period = sum(t['bars'] for t in trades_list) / total_trades
            
        # === 年化与夏普计算 ===
        annual_return = (1 + total_return) ** (1 / years_in_backtest) - 1 if total_return > -1 else -1
        market_annual_return = (1 + market_total_return) ** (1 / years_in_backtest) - 1 if market_total_return > -1 else -1
        
        daily_std = strategy_returns.std()
        sharpe_ratio = 0.0
        if daily_std > 1e-6:
            periods_per_year = 365.25 * 24 if timeframe in ['1h', 'H'] else 365.25
            sharpe_ratio = (strategy_returns.mean() / daily_std) * np.sqrt(periods_per_year)

        # === 返回结果 ===
        metrics = {
            'days_in_backtest': float(days_in_backtest),
            'years_in_backtest': float(years_in_backtest),
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'win_loss_ratio': float(win_loss_ratio),
            'annual_return': float(annual_return),
            'sharpe_ratio': float(sharpe_ratio),
            'excess_return': float(total_return - market_total_return),
            'excess_annual_return': float(annual_return - market_annual_return),
            'market_return': float(market_total_return),
            'market_annual_return': float(market_annual_return),
            'total_profit_loss': float(total_return), # 【修复】总盈亏等同于复利收益
            'avg_profit_loss_per_trade': float(avg_profit_loss_per_trade),
            'total_trades': int(total_trades),
            'avg_hold_period': float(avg_hold_period),
            'start_date': start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'end_date': end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'timeframe': timeframe
        }
        return metrics

    except Exception as e:
        logger.error(f"计算指标时出错: {e}")
        raise

def run_backtest(df, code_string, params, timeframe):
    """
    运行回测，计算策略性能指标
    
    参数:
        df (pd.DataFrame): 历史K线数据
        code_string (str): 包含generate_signals函数的Python代码字符串
        params (dict): 策略参数
        timeframe (str): 时间周期，如 '1h', '1d'
    
    返回:
        dict: 回测结果字典，包含详细的性能指标
    """
    try:
        logger.info(f"开始运行回测，时间周期: {timeframe}")
        
        # 步骤A: 降采样数据
        resampled_df = resample_data(df, timeframe)
        
        # 步骤B: 编译策略代码，获取策略函数
        strategy_func = compile_strategy(code_string)
        
        # 步骤C: 获取目标持仓状态
        logger.info(f"调用策略函数，输入数据形状: {resampled_df.shape}, 列: {resampled_df.columns.tolist()}")
        signals = strategy_func(resampled_df, params)
        logger.info(f"策略函数返回信号: {signals.value_counts().to_dict() if hasattr(signals, 'value_counts') else signals}")
        logger.info(f"信号总个数: {len(signals)}, 非零信号数: {(signals != 0).sum() if hasattr(signals, '__len__') else 'N/A'}")
        
        # 步骤D: 严谨的向量化盈亏计算
        # 手续费率改为万分之五
        commission_rate = params.get('commission_rate', 0.0005)
        
        # 仓位延迟一根K线生效
        positions = signals.shift(1).fillna(0)
        
        # 基础收益率
        returns = resampled_df['close'].pct_change().fillna(0)
        
        # 状态变更检测（交易发生）
        trades = positions.diff().fillna(0).abs()
        
        # 手续费损耗
        costs = trades * commission_rate
        
        # 策略净收益
        strategy_returns = (positions * returns) - costs
        
        # 一次性添加所有新列，避免DataFrame碎片化
        new_columns = pd.DataFrame({
            'signals': positions,
            'returns': returns,
            'strategy_returns': strategy_returns,
            'timeframe': timeframe
        }, index=resampled_df.index)
        
        resampled_df = pd.concat([resampled_df, new_columns], axis=1)
        
        # 步骤E: 计算详细的性能指标
        metrics = calculate_metrics(resampled_df)
        
        # 确保时间周期在结果中
        metrics['timeframe'] = timeframe
        
        logger.info(f"回测完成，结果: {metrics}")
        return metrics
    except Exception as e:
        logger.error(f"运行回测时出错: {e}")
        raise


def run_sensitivity_test(df, code_string, params, timeframe, base_sharpe,
                         sharpe_drop_threshold=0.4, max_tests_per_param=3):
    """
    参数敏感度（高原）测试：对策略的数值型参数进行微小偏移并重测，
    判断策略是否处于"过拟合悬崖"。

    Args:
        df: 历史K线数据
        code_string: 策略代码
        params: 当前参数字典
        timeframe: 时间周期
        base_sharpe: 基准夏普率
        sharpe_drop_threshold: 夏普率跌幅阈值（默认0.4，即40%）
        max_tests_per_param: 每个参数最多偏移测试次数

    Returns:
        dict: {
            'passed': bool, 是否通过敏感度测试,
            'base_sharpe': float, 基准夏普率,
            'details': list, 每个参数的测试详情,
            'cliff_params': list, 处于悬崖的参数名列表,
            'summary': str, 人类可读的摘要
        }
    """
    try:
        logger.info(f"开始参数敏感度测试，基准夏普率: {base_sharpe:.2f}")

        numeric_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float)) and key not in ('commission_rate',):
                numeric_params[key] = value

        if not numeric_params:
            logger.info("无数值型参数可测试，自动通过")
            return {
                'passed': True,
                'base_sharpe': base_sharpe,
                'details': [],
                'cliff_params': [],
                'summary': '无数值型参数，自动通过敏感度测试'
            }

        details = []
        cliff_params = []
        total_tests = 0

        for param_name, param_value in numeric_params.items():
            param_detail = {
                'param_name': param_name,
                'base_value': param_value,
                'perturbations': []
            }

            if isinstance(param_value, int):
                offsets = [1, -1]
                if param_value > 5:
                    offsets.append(2)
            else:
                offsets = [param_value * 0.1, -param_value * 0.1]
                if abs(param_value) > 0.5:
                    offsets.append(param_value * 0.2)

            offsets = offsets[:max_tests_per_param]

            for offset in offsets:
                new_value = param_value + offset

                if isinstance(param_value, int):
                    new_value = int(round(new_value))
                    if new_value == param_value or new_value <= 0:
                        continue
                else:
                    if abs(new_value - param_value) < 1e-9 or new_value <= 0:
                        continue

                perturbed_params = dict(params)
                perturbed_params[param_name] = new_value

                try:
                    perturbed_metrics = run_backtest(df, code_string, perturbed_params, timeframe)
                    perturbed_sharpe = perturbed_metrics.get('sharpe_ratio', 0)
                    drop_ratio = (base_sharpe - perturbed_sharpe) / abs(base_sharpe) if abs(base_sharpe) > 1e-6 else 0

                    pert_result = {
                        'offset': offset,
                        'new_value': new_value,
                        'sharpe': perturbed_sharpe,
                        'drop_ratio': drop_ratio,
                        'is_cliff': drop_ratio > sharpe_drop_threshold
                    }
                    param_detail['perturbations'].append(pert_result)

                    if pert_result['is_cliff']:
                        cliff_params.append(param_name)
                        logger.warning(
                            f"参数敏感度悬崖: {param_name}={param_value} -> {new_value}, "
                            f"夏普率 {base_sharpe:.2f} -> {perturbed_sharpe:.2f} (跌幅 {drop_ratio:.1%})"
                        )

                    total_tests += 1
                except Exception as e:
                    pert_result = {
                        'offset': offset,
                        'new_value': new_value,
                        'sharpe': None,
                        'drop_ratio': 1.0,
                        'is_cliff': True,
                        'error': str(e)
                    }
                    param_detail['perturbations'].append(pert_result)
                    cliff_params.append(param_name)
                    logger.warning(f"参数敏感度测试报错: {param_name}={param_value} -> {new_value}, 错误: {e}")
                    total_tests += 1

            details.append(param_detail)

        passed = len(cliff_params) == 0

        cliff_params_unique = list(dict.fromkeys(cliff_params))

        if passed:
            summary = f"参数敏感度测试通过！共测试 {total_tests} 次偏移，所有参数均处于高原区域。"
        else:
            summary = (
                f"参数敏感度测试未通过！共测试 {total_tests} 次偏移，"
                f"以下参数处于过拟合悬崖: {', '.join(cliff_params_unique)}。"
                f"偏移后夏普率跌幅超过 {sharpe_drop_threshold:.0%}，策略高度依赖精确参数值。"
            )

        logger.info(summary)

        return {
            'passed': passed,
            'base_sharpe': base_sharpe,
            'details': details,
            'cliff_params': cliff_params_unique,
            'summary': summary
        }

    except Exception as e:
        logger.error(f"参数敏感度测试异常: {e}")
        return {
            'passed': False,
            'base_sharpe': base_sharpe,
            'details': [],
            'cliff_params': [],
            'summary': f'敏感度测试异常: {e}'
        }


def create_temp_test_table():
    """
    创建临时测试表用于存储回测结果
    """
    try:
        logger.info("创建临时测试表")
        
        # 连接数据库
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 首先检查表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='backtest_test_results'")
        table_exists = cursor.fetchone() is not None
        
        if not table_exists:
            # 创建新表
            create_table_sql = """
            CREATE TABLE backtest_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                start_date TEXT,
                end_date TEXT,
                total_trades INTEGER,
                total_profit_loss REAL,
                avg_profit_loss_per_trade REAL,
                win_rate REAL,
                win_loss_ratio REAL,
                total_return REAL,
                annual_return REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                excess_return REAL,
                excess_annual_return REAL,
                market_return REAL,
                market_annual_return REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_table_sql)
            logger.info("临时测试表创建成功")
        else:
            # 表已存在，检查并添加缺失的字段
            fields_to_add = [
                ('excess_annual_return', 'REAL'),
                ('market_return', 'REAL'),
                ('market_annual_return', 'REAL')
            ]
            
            for field_name, field_type in fields_to_add:
                # 检查字段是否存在
                cursor.execute("PRAGMA table_info(backtest_test_results)")
                existing_fields = [row[1] for row in cursor.fetchall()]
                
                if field_name not in existing_fields:
                    # 添加字段
                    alter_sql = f"ALTER TABLE backtest_test_results ADD COLUMN {field_name} {field_type}"
                    cursor.execute(alter_sql)
                    logger.info(f"添加字段: {field_name}")
        
        conn.commit()
        conn.close()
        
        logger.info("临时测试表准备完成")
    except Exception as e:
        logger.error(f"创建临时测试表时出错: {e}")
        raise

def save_backtest_result_to_temp_table(result, strategy_name="test_strategy"):
    """
    将回测结果保存到临时测试表
    
    参数:
        result (dict): 回测结果字典
        strategy_name (str): 策略名称
    """
    try:
        logger.info("保存回测结果到临时测试表")
        
        # 连接数据库
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 插入数据
        insert_sql = """
        INSERT INTO backtest_test_results (
            strategy_name, start_date, end_date, total_trades, total_profit_loss, 
            avg_profit_loss_per_trade, win_rate, win_loss_ratio, total_return, 
            annual_return, max_drawdown, sharpe_ratio, excess_return, 
            excess_annual_return, market_return, market_annual_return
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        cursor.execute(insert_sql, (
            strategy_name,
            result.get('start_date', ''),
            result.get('end_date', ''),
            result.get('total_trades', 0),
            result.get('total_profit_loss', 0),
            result.get('avg_profit_loss_per_trade', 0),
            result.get('win_rate', 0),
            result.get('win_loss_ratio', 0),
            result.get('total_return', 0),
            result.get('annual_return', 0),
            result.get('max_drawdown', 0),
            result.get('sharpe_ratio', 0),
            result.get('excess_return', 0),
            result.get('excess_annual_return', 0),
            result.get('market_return', 0),
            result.get('market_annual_return', 0)
        ))
        
        conn.commit()
        conn.close()
        
        logger.info("回测结果保存成功")
    except Exception as e:
        logger.error(f"保存回测结果时出错: {e}")
        raise

def update_strategy_versions_table():
    """
    更新strategy_versions表结构，添加所需的字段
    """
    try:
        logger.info("更新strategy_versions表结构")
        
        # 连接数据库
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 检查并添加缺失的字段
        fields_to_add = [
            ('start_date', 'TEXT'),
            ('end_date', 'TEXT'),
            ('total_trades', 'INTEGER'),
            ('total_profit_loss', 'REAL'),
            ('avg_profit_loss_per_trade', 'REAL'),
            ('win_rate', 'REAL'),
            ('win_loss_ratio', 'REAL'),
            ('annual_return', 'REAL'),
            ('max_drawdown', 'REAL'),
            ('excess_return', 'REAL')
        ]
        
        for field_name, field_type in fields_to_add:
            # 检查字段是否存在
            cursor.execute(f"PRAGMA table_info(strategy_versions)")
            existing_fields = [row[1] for row in cursor.fetchall()]
            
            if field_name not in existing_fields:
                # 添加字段
                alter_sql = f"ALTER TABLE strategy_versions ADD COLUMN {field_name} {field_type}"
                cursor.execute(alter_sql)
                logger.info(f"添加字段: {field_name}")
        
        conn.commit()
        conn.close()
        
        logger.info("strategy_versions表结构更新完成")
    except Exception as e:
        logger.error(f"更新strategy_versions表结构时出错: {e}")
        raise

def save_backtest_result_to_strategy_versions(ver_id, result):
    """
    将回测结果保存到strategy_versions表
    
    参数:
        ver_id (str): 策略版本ID
        result (dict): 回测结果字典
    """
    try:
        logger.info(f"保存回测结果到strategy_versions表，版本ID: {ver_id}")
        
        # 连接数据库
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 更新数据
        update_sql = """
        UPDATE strategy_versions SET 
            metric_sharpe = ?, 
            metric_return = ?, 
            metric_max_drawdown = ?, 
            metric_win_rate = ?, 
            metric_profit_loss_ratio = ?, 
            metric_total_trades = ?, 
            metric_annualized_return = ?, 
            metric_excess_return = ?, 
            timeframe = ?, 
            run_status = 'completed'
        WHERE ver_id = ?
        """
        
        cursor.execute(update_sql, (
            result.get('sharpe_ratio', 0),
            result.get('total_return', 0),
            result.get('max_drawdown', 0),
            result.get('win_rate', 0),
            result.get('win_loss_ratio', 0),
            result.get('total_trades', 0),
            result.get('annual_return', 0),
            result.get('excess_return', 0),
            result.get('timeframe', '1h'),
            ver_id
        ))
        
        conn.commit()
        conn.close()
        
        logger.info("回测结果保存成功")
    except Exception as e:
        logger.error(f"保存回测结果到strategy_versions表时出错: {e}")
        raise
