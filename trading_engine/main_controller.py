#!/usr/bin/env python3
# 系统总控制器 - 串联前4步的所有组件

import pandas as pd
import json
import traceback

# 导入日志设置
from logger_setup import get_logger
logger = get_logger("MainController")

# 导入数据库管理函数
from database.db_manager import (
    init_db,
    insert_strategy_direction,
    insert_strategy_version,
    update_strategy_version_status,
    update_strategy_version_metrics,
    update_strategy_direction_best_version,
    get_strategy_version_by_id,
    get_strategy_versions_by_dir,
    get_overfitted_feedback,
    get_factor_statistics_summary,
    get_available_factor_names
)

# 导入回测引擎
from trading_engine.backtest_engine import run_backtest, load_historical_data, run_sensitivity_test

# 导入大模型代理
from agents.base_llm_client import (
    build_init_prompt,
    build_debug_prompt,
    build_tune_prompt,
    call_llm,
    check_code_safety
)


def is_better_metrics(current_metrics, best_metrics):
    """
    判断当前指标是否比最佳指标更好
    
    优先比较夏普率，其次比较收益率
    
    Args:
        current_metrics (dict): 当前回测指标
        best_metrics (dict): 最佳回测指标
    
    Returns:
        bool: 是否更好
    """
    if not best_metrics:
        return True
    
    current_sharpe = current_metrics.get('sharpe_ratio', 0) or 0
    best_sharpe = best_metrics.get('sharpe_ratio', 0) or 0
    
    if current_sharpe > best_sharpe:
        return True
    elif current_sharpe == best_sharpe:
        current_return = current_metrics.get('total_return', 0) or 0
        best_return = best_metrics.get('total_return', 0) or 0
        return current_return > best_return
    else:
        return False


def process_new_direction(dir_id, name, description, timeframe, target_asset='crypto', target_symbol='BTC_USDT'):
    """
    处理新的策略方向，执行完整的工作流
    
    完整工作流程：
    1. 生成初始代码
    2. 静态安检
    3. 存入数据库
    4. Debug 阶段（最多3次）
    5. 优化阶段（最多10次迭代）
    
    Args:
        dir_id (str): 策略方向 ID
        name (str): 策略方向名称
        description (str): 策略方向描述
        timeframe (str): 时间周期
        target_asset (str): 资产大类 (crypto/gold/oil/us_stock)，用于路由到数据库表
        target_symbol (str): 具体交易标的 (BTC_USDT/GCUSD/BZUSD/SPY/QQQ)，用于回测
    
    Returns:
        tuple: (是否成功, 最佳版本ID, 最佳版本指标)
    """
    logger.info(f"开始处理策略方向: {name}, dir_id: {dir_id}, target_asset: {target_asset}, target_symbol: {target_symbol}")
    
    best_ver_id = None
    best_metrics = None
    
    # 预先加载历史数据（只加载一次，提高效率）
    logger.info(f"预先加载历史数据 (资产: {target_asset}, 标的: {target_symbol})")
    
    available_factors = get_available_factor_names(target_symbol=target_symbol)
    logger.info(f"可用因子数量: {len(available_factors)}, 因子列表: {available_factors[:10]}{'...' if len(available_factors) > 10 else ''}")
    
    df = load_historical_data(symbol=target_symbol, target_asset=target_asset, factor_names=available_factors)
    
    if df.empty:
        logger.warning(f"{target_symbol} 无历史数据，跳过该策略")
        return False, None, None
    
    logger.info(f"历史数据加载成功，共 {len(df)} 行")
    
    # ==================== 阶段 1: 初始生成 ====================
    print(f"\n{'='*60}")
    print(f"阶段 1: 初始策略生成")
    print(f"{'='*60}")
    
    # 获取因子统计特征
    logger.info("获取因子统计特征")
    factor_stats_str = get_factor_statistics_summary(target_symbol=target_symbol, factor_names=available_factors)
    logger.info(f"因子统计特征获取完成，长度: {len(factor_stats_str)}")
    
    # 获取过拟合反馈（如果之前有版本被标记为 OVERFITTED）
    overfitted_feedback = get_overfitted_feedback(dir_id)
    if overfitted_feedback:
        logger.info(f"检测到过拟合反馈，将传给 LLM 进行重写")
        print(f"\n⚠️ 检测到之前的过拟合记录，将指导 LLM 避免重蹈覆辙")
    
    code_content = None
    params_json = None
    
    try:
        logger.info("生成初始代码")
        prompt = build_init_prompt(
            name, description, timeframe, factor_stats_str,
            target_symbol=target_symbol,
            overfitted_feedback=overfitted_feedback
        )
        llm_result = call_llm(prompt)
        code_content = llm_result.get('code_content')
        params_json = llm_result.get('params_json')
        
        if not code_content or not params_json:
            logger.error("大模型返回结果不完整")
            return False, None, None
        
        logger.info("初始代码生成成功")
        
    except Exception as e:
        logger.error(f"初始代码生成失败: {e}")
        traceback.print_exc()
        return False, None, None
    
    # ==================== 阶段 2: Debug（最多3次） ====================
    print(f"\n{'='*60}")
    print(f"阶段 2: Debug 阶段（最多3次）")
    print(f"{'='*60}")
    
    debug_success = False
    final_code = code_content
    final_params = params_json
    
    for debug_attempt in range(3):
        print(f"\n--- Debug 尝试 {debug_attempt + 1}/3 ---")
        
        # 步骤 1: 静态安检
        logger.info("静态代码安全检查")
        is_safe, message = check_code_safety(final_code)
        if not is_safe:
            logger.error(f"代码安全检查失败: {message}")
            print(f"代码安全检查失败: {message}")
            
            # 调用大模型修复代码
            try:
                logger.info("调用大模型修复代码")
                debug_prompt = build_debug_prompt(final_code, message)
                debug_result = call_llm(debug_prompt)
                final_code = debug_result.get('code_content', final_code)
                logger.info("代码修复成功")
                continue
            except Exception as e:
                logger.error(f"代码修复失败: {e}")
                traceback.print_exc()
                return False, None, None
        
        logger.info("代码安全检查通过")
        
        # 步骤 2: 存入数据库
        logger.info("存入数据库")
        iteration_type = 'INIT' if debug_attempt == 0 else f'DEBUG_{debug_attempt}'
        ver_id = insert_strategy_version(
            dir_id=dir_id,
            code_content=final_code,
            params_json=final_params,
            iteration_type=iteration_type,
            run_status='PENDING'
        )
        logger.info(f"存入数据库成功，ver_id: {ver_id}")
        
        # 步骤 3: 运行回测
        logger.info("运行回测")
        try:
            metrics = run_backtest(df, final_code, final_params, timeframe)
            logger.info("回测成功")
            print(f"✓ 回测成功！")
            print(f"  - 收益率: {metrics.get('total_return', 0):.4%}")
            print(f"  - 夏普率: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  - 最大回撤: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  - 胜率: {metrics.get('win_rate', 0):.2%}")
            print(f"  - 交易次数: {metrics.get('total_trades', 0)}")
            
            # 更新数据库
            update_strategy_version_status(ver_id, 'SUCCESS')
            update_strategy_version_metrics(ver_id, metrics)
            
            # 更新最佳版本
            if is_better_metrics(metrics, best_metrics):
                best_metrics = metrics
                best_ver_id = ver_id
                update_strategy_direction_best_version(dir_id, best_ver_id)
                print(f"✓ 发现新的最佳版本！ver_id: {best_ver_id}")
            
            debug_success = True
            break
            
        except Exception as e:
            error_stack = traceback.format_exc()
            logger.error(f"回测失败: {e}")
            logger.error(f"错误堆栈: {error_stack}")
            print(f"✗ 回测失败: {e}")
            
            # 更新数据库状态
            update_strategy_version_status(ver_id, 'ERROR', error_stack)
            
            # 如果不是最后一次尝试，调用大模型修复
            if debug_attempt < 2:
                try:
                    logger.info("调用大模型修复代码")
                    debug_prompt = build_debug_prompt(final_code, error_stack)
                    debug_result = call_llm(debug_prompt)
                    final_code = debug_result.get('code_content', final_code)
                    logger.info("代码修复成功")
                except Exception as repair_e:
                    logger.error(f"代码修复失败: {repair_e}")
                    traceback.print_exc()
                    return False, None, None
    
    if not debug_success:
        logger.error("Debug 阶段失败，超过最大尝试次数")
        print("✗ Debug 阶段失败，超过最大尝试次数")
        return False, None, None
    
    # ==================== 阶段 3: 参数优化（最多10次） ====================
    print(f"\n{'='*60}")
    print(f"阶段 3: 参数优化阶段（最多10次迭代）")
    print(f"{'='*60}")
    
    current_code = final_code
    current_params = final_params
    current_metrics = best_metrics
    
    for iteration in range(10):
        print(f"\n--- 优化迭代 {iteration + 1}/10 ---")
        
        # 调用大模型调优参数
        try:
            logger.info("调用大模型调优参数")
            tune_prompt = build_tune_prompt(
                description=description,
                current_params=current_params,
                current_metrics=current_metrics
            )
            tune_result = call_llm(tune_prompt)
            new_params = tune_result.get('params_json')
            
            if not new_params:
                logger.error("大模型未返回新参数")
                continue
            
            logger.info(f"新参数: {new_params}")
            print(f"新参数: {new_params}")
            
        except Exception as e:
            logger.error(f"参数调优调用失败: {e}")
            traceback.print_exc()
            continue
        
        # 存入数据库
        iteration_type = f'TUNE_{iteration + 1}'
        ver_id = insert_strategy_version(
            dir_id=dir_id,
            code_content=current_code,
            params_json=new_params,
            iteration_type=iteration_type,
            run_status='PENDING'
        )
        logger.info(f"存入数据库成功，ver_id: {ver_id}")
        
        # 运行回测
        try:
            metrics = run_backtest(df, current_code, new_params, timeframe)
            logger.info("回测成功")
            print(f"✓ 回测成功！")
            print(f"  - 收益率: {metrics.get('total_return', 0):.4%}")
            print(f"  - 夏普率: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  - 最大回撤: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  - 胜率: {metrics.get('win_rate', 0):.2%}")
            print(f"  - 交易次数: {metrics.get('total_trades', 0)}")
            
            # 更新数据库
            update_strategy_version_status(ver_id, 'SUCCESS')
            update_strategy_version_metrics(ver_id, metrics)
            
            # 检查是否更好
            if is_better_metrics(metrics, best_metrics):
                best_metrics = metrics
                best_ver_id = ver_id
                current_params = new_params
                current_metrics = metrics
                update_strategy_direction_best_version(dir_id, best_ver_id)
                print(f"✓ 发现新的最佳版本！ver_id: {best_ver_id}")
            else:
                print(f"- 本次优化效果不如最佳版本，保持当前最佳")
                
        except Exception as e:
            error_stack = traceback.format_exc()
            logger.error(f"回测失败: {e}")
            print(f"✗ 回测失败: {e}")
            update_strategy_version_status(ver_id, 'ERROR', error_stack)
    
    # ==================== 完成 ====================
    
    # 最终敏感度测试：对最佳版本进行参数高原检验
    overfitting_detected = False
    sensitivity_summary = ""
    
    if best_metrics and best_metrics.get('sharpe_ratio', 0) > 0.5:
        print(f"\n{'='*60}")
        print(f"阶段 4: 参数敏感度（高原）测试")
        print(f"{'='*60}")
        
        try:
            sensitivity_result = run_sensitivity_test(
                df=df,
                code_string=current_code,
                params=current_params,
                timeframe=timeframe,
                base_sharpe=best_metrics.get('sharpe_ratio', 0)
            )
            
            sensitivity_summary = sensitivity_result['summary']
            print(f"\n{sensitivity_summary}")
            
            if not sensitivity_result['passed']:
                overfitting_detected = True
                cliff_info = ', '.join(sensitivity_result['cliff_params'])
                
                print(f"\n[OVERFITTED] 策略未通过参数敏感度测试！")
                print(f"  悬崖参数: {cliff_info}")
                print(f"  该策略高度依赖精确参数值，属于过拟合。")
                
                if best_ver_id:
                    overfit_log = (
                        f"参数敏感度测试未通过。{sensitivity_summary} "
                        f"悬崖参数: {cliff_info}。"
                        f"建议：简化条件分支（每个if最多3个and/or），"
                        f"使用宽泛阈值替代精确值，降低参数敏感度。"
                    )
                    update_strategy_version_status(best_ver_id, 'OVERFITTED', overfit_log)
                    logger.info(f"策略版本 {best_ver_id} 标记为 OVERFITTED")
                
                update_strategy_direction_best_version(dir_id, None)
                logger.info(f"已清除策略方向 {dir_id} 的 best_version_id，策略将回到未处理状态等待重写")
            else:
                print(f"\n[PASSED] 策略通过参数敏感度测试，参数处于高原区域。")
                
        except Exception as e:
            logger.error(f"参数敏感度测试异常: {e}")
            print(f"参数敏感度测试异常: {e}")
    
    print(f"\n{'='*60}")
    print(f"策略方向处理完成！")
    print(f"{'='*60}")
    if best_metrics:
        print(f"最佳版本 ver_id: {best_ver_id}")
        print(f"最佳回测结果:")
        print(f"  - 收益率: {best_metrics.get('total_return', 0):.4%}")
        print(f"  - 夏普率: {best_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  - 最大回撤: {best_metrics.get('max_drawdown', 0):.2f}%")
        print(f"  - 胜率: {best_metrics.get('win_rate', 0):.2f}%")
        print(f"  - 交易次数: {best_metrics.get('total_trades', 0)}")
        if overfitting_detected:
            print(f"  - [OVERFITTED] 未通过参数敏感度测试")
        elif sensitivity_summary:
            print(f"  - [PASSED] 参数敏感度测试通过")
    
    logger.info(f"策略方向处理完成，dir_id: {dir_id}, overfitting={overfitting_detected}")
    return True, best_ver_id, best_metrics


if __name__ == "__main__":
    """
    测试总控制器
    """
    print("===== 测试总控制器 ======")
    
    try:
        # 初始化数据库
        print("初始化数据库...")
        init_db()
        print("数据库初始化完成")
        
        # 构造测试数据
        test_dir = {
            "name": "均线交叉验证",
            "description": "短周期上穿长周期做多，跌破做空",
            "timeframe": "1h"
        }
        
        # 插入策略方向
        print("插入策略方向...")
        dir_id = insert_strategy_direction(
            name=test_dir['name'],
            description=test_dir['description'],
            timeframe=test_dir['timeframe']
        )
        print(f"策略方向插入成功，dir_id: {dir_id}")
        
        # 处理策略方向
        print("处理策略方向...")
        success, best_ver_id, best_metrics = process_new_direction(
            dir_id=dir_id,
            name=test_dir['name'],
            description=test_dir['description'],
            timeframe=test_dir['timeframe'],
            target_asset='crypto',
            target_symbol='BTC_USDT'
        )
        
        if success:
            print("\n✓ 测试成功！策略处理完成")
        else:
            print("\n✗ 测试失败！策略处理失败")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

