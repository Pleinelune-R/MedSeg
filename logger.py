import logging
import logging.handlers
import os
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import pwd


# This module should be used in >=3.4 version


class MyLogger(logging.Logger):
    """Create my self.log from python stdlib self.log module"""

    def __init__(self, name, level=logging.DEBUG):
        #
        super().__init__(name)

        # 确保日志目录存在
        log_dir = Path('log')
        log_dir.mkdir(parents=True, exist_ok=True)  # 自动创建目录

        # set logger level
        self.setLevel(logging.DEBUG)

        # get cwd
        current_directory = Path.cwd()

        # get folders things
        files_and_folders = current_directory.iterdir()
        LOG_PATH_CONTENT = "\r\n".join(item.name for item in files_and_folders)

        LOG_PATH = os.getcwd()
        USERNAME = pwd.getpwuid(os.getuid()).pw_name

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
                                                when='S',
                                                interval=5,
                                                backupCount=1,
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

        # add Logger to handler
        self.addHandler(file_handler)
        self.addHandler(stream_handler)

        # logger add more info
        self.info(f'{name} Logger has been initialized')
        self.info(f'{USERNAME} LOG_PATH is {LOG_PATH}\r\n{LOG_PATH_CONTENT}')
