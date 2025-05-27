import logging
import logging.handlers
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

def setup_logger(name, level=logging.DEBUG):
    """Setup and return a logger instance"""
    # 检查是否已经存在该名称的logger
    logger = logging.getLogger(name)
    
    # 如果logger已经有handler，说明已经初始化过，直接返回
    if logger.handlers:
        return logger
        
    # 确保日志目录存在
    log_dir = Path('log')
    log_dir.mkdir(parents=True, exist_ok=True)

    # set logger level
    logger.setLevel(logging.DEBUG)

    # stdout handler
    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(logging.Formatter("\n- ThreadId: %(thread)d "
                                                "\n- Level: %(levelname)s "
                                                "\n- FileName:%(filename)s "
                                                "\n- Lineno:%(lineno)d "
                                                "\n- Message:\n--------------------------------\n%(message)s\n--------------------------------\n"))
    stream_handler.setLevel(level)

    # file output handler
    file_handler = TimedRotatingFileHandler('log/' + name,
                                            when='H',
                                            interval=1,
                                            backupCount=2,
                                            delay=True)
    file_handler.suffix = "%Y-%m-%d_%H-%M-%S.log"
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s "
                                                "\n- ProcessId:%(process)d "
                                                "\n- ThreadId:%(thread)d "
                                                "\n- ThreadName:%(threadName)s "
                                                "\n- Level:%(levelname)s "
                                                "\n- FileName:%(filename)s "
                                                "\n- Lineno:%(lineno)d "
                                                "\n- FunctionName:%(funcName)s "
                                                "\n- Message:\n--------------------------------\n%(message)s\n--------------------------------\n"))

    # add handlers
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # 只在第一次初始化时打印信息
    logger.info(f'{name} Logger has been initialized')
    
    return logger

# 为了保持向后兼容，保留MyLogger类但改为使用setup_logger
class MyLogger:
    def __new__(cls, name, level=logging.DEBUG):
        return setup_logger(name, level)


        