#!/usr/bin/env python3
"""
测试 draft adapter 测试工具 - 支持单个或批量测试
"""

import sys
import os
import importlib.util

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if project_root not in sys.path:

    sys.path.insert(0, project_root)

from logger_setup import get_logger

logger = get_logger("test_draft_adapters")


def load_draft_adapter(file_path):
    """
    加载 draft adapter 文件
    
    Args:
        file_path (str): draft adapter 文件路径
        
    Returns:
        module: 加载的模块对象
    """
    try:
        module_name = os.path.splitext(os.path.basename(file_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logger.error(f"加载 draft adapter 失败: {file_path}, 错误: {e}")
        return None


def test_single_adapter(file_path):
    """
    测试单个 draft adapter
    
    Args:
        file_path (str): draft adapter 文件路径
        
    Returns:
        bool: 测试是否成功
    """
    print("=" * 80)
    print(f"测试 draft adapter: {os.path.basename(file_path)}")
    print("=" * 80)
    
    module = load_draft_adapter(file_path)
    if not module:
        return False
    
    try:
        # 检查必要的函数
        has_fetch = hasattr(module, 'fetch_raw_test')
        has_process = hasattr(module, 'process_factor_test')
        
        if not has_fetch:
            print(f"❌ 缺少 fetch_raw_test() 函数")
            return False
        
        if not has_process:
            print(f"❌ 缺少 process_factor_test() 函数")
            return False
        
        print(f"\n✅ 找到必要的函数: fetch_raw_test(), process_factor_test()")
        
        # 运行 fetch_raw_test()
        print(f"\n--- 步骤 1: 运行 fetch_raw_test()...")
        raw_data = module.fetch_raw_test()
        print(f"✅ fetch_raw_test() 返回: {type(raw_data)}")
        
        # 运行 process_factor_test()
        print(f"\n--- 步骤 2: 运行 process_factor_test()...")
        if hasattr(raw_data, '__len__') and len(raw_data) > 0:
            factor_df = module.process_factor_test(raw_data)
            print(f"✅ process_factor_test() 返回: {type(factor_df)}")
        else:
            print(f"⚠️  raw_data 为空，跳过 process_factor_test()")
        
        print(f"\n✅ {os.path.basename(file_path)} 测试通过！")
        return True
        
    except Exception as e:
        logger.error(f"测试 draft adapter 失败: {file_path}, 错误: {e}")
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    draft_zone_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "draft_zone")
    
    if not os.path.exists(draft_zone_path):
        print(f"❌ draft_zone 目录不存在: {draft_zone_path}")
        return
    
    # 列出所有 draft_*.py 文件
    draft_files = []
    for filename in os.listdir(draft_zone_path):
        if filename.startswith("draft_") and filename.endswith(".py"):
            draft_files.append(os.path.join(draft_zone_path, filename))
    
    if not draft_files:
        print(f"📭 draft_zone 目录下没有找到 draft_*.py 文件")
        return
    
    print("=" * 80)
    print("draft adapter 测试工具")
    print("=" * 80)
    
    print(f"\n找到 {len(draft_files)} 个 draft adapter:")
    for i, file_path in enumerate(draft_files, 1):
        print(f"  {i}. {os.path.basename(file_path)}")
    
    print("\n请选择测试模式:")
    print("  1. 测试单个指定的 adapter")
    print("  2. 批量测试所有 adapter")
    
    choice = input("\n请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        # 测试单个
        num = input("\n请输入要测试的 adapter 编号 (1-{}): ".format(len(draft_files))).strip()
        try:
            idx = int(num) - 1
            if 0 <= idx < len(draft_files):
                test_single_adapter(draft_files[idx])
            else:
                print("❌ 无效的编号")
        except ValueError:
            print("❌ 请输入有效的数字")
    
    elif choice == "2":
        # 批量测试
        print("\n" + "=" * 80)
        print("开始批量测试所有 adapter")
        print("=" * 80)
        
        success_count = 0
        fail_count = 0
        
        for i, file_path in enumerate(draft_files, 1):
            print(f"\n[{i}/{len(draft_files)} 测试: {os.path.basename(file_path)}")
            result = test_single_adapter(file_path)
            if result:
                success_count += 1
            else:
                fail_count += 1
        
        print("\n" + "=" * 80)
        print("批量测试完成！")
        print("=" * 80)
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        print("=" * 80)
    
    else:
        print("❌ 无效的选择")


if __name__ == "__main__":
    main()
