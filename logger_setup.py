import logging
import os


def get_logger(name):
    """
    获取配置好的日志记录器
    
    Args:
        name: 日志记录器的名称，通常使用 __name__
        
    Returns:
        logging.Logger: 配置好的日志记录器实例
    """
    # 动态导入 config 以避免循环依赖
    try:
        from config import LOG_LEVEL
    except ImportError:
        # 如果 config 尚未准备好，默认使用 INFO
        LOG_LEVEL = logging.INFO

    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)
    
    # 检查是否已经添加了处理器，避免重复添加
    if not logger.handlers:
        # 确保logs目录存在
        os.makedirs('./logs', exist_ok=True)
        
        # 创建文件处理器，输出到logs/system.log
        file_handler = logging.FileHandler('./logs/system.log', encoding='utf-8')
        file_handler.setLevel(LOG_LEVEL)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(LOG_LEVEL)
        
        # 定义日志格式
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # 添加处理器到日志记录器
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger



if __name__ == "__main__":
    """
    测试日志模块
    """
    # 获取日志记录器
    logger = get_logger(__name__)
    
    # 测试不同级别的日志
    logger.debug("这是一个debug级别日志")
    logger.info("这是一个info级别日志")
    logger.warning("这是一个warning级别日志")
    logger.error("这是一个error级别日志")
    logger.critical("这是一个critical级别日志")
    
    print("日志测试完成，查看logs/system.log文件和控制台输出")