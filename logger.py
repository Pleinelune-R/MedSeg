import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from colorama import init, Fore, Back, Style

init(autoreset=True)  # Auto reset terminal colors


# Custom colored Formatter
class ColorFormatter(logging.Formatter):
    """Colored log formatter (applies to console only)."""

    # Map log levels to colors
    LEVEL_COLORS = {
        logging.DEBUG: Fore.CYAN,  # cyan
        logging.INFO: Fore.GREEN,  # green
        logging.WARNING: Fore.YELLOW,  # yellow
        logging.ERROR: Fore.RED,  # red
        logging.CRITICAL: Fore.RED + Back.WHITE + Style.BRIGHT,  # red background, white bright text
    }

    def format(self, record):
        original_message = super().format(record)
        # get color
        color = self.LEVEL_COLORS.get(record.levelno, Fore.RESET)
        # replace [LEVEL] with color
        colored_level = (
            f"{color}{record.levelname}{Fore.RESET}"
        )
        formatted_record = original_message.replace(f"{record.levelname}", colored_level)

        # multilines helper
        if '\n' in record.message:
            # calculate prefix length
            first_line = formatted_record.split('\n')[0]
            # len(f"{color}{Fore.RESET}") == 10
            prefix_length = len(first_line) - len(record.message.split('\n')[0]) - 10
            # split into sever lines
            lines = record.message.split('\n')
            # add first line
            formatted_lines = [first_line]

            # add prefix
            for line in lines[1:]:
                aligned_prefix = ' ' * prefix_length
                formatted_lines.append(f"{aligned_prefix}{line}")

            # rebuild lines
            formatted_record = '\n'.join(formatted_lines)

        return formatted_record


# Log level mapping
LOG_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
}


class LoggerManager:
    """logger manager to control the logger"""

    def __init__(self):
        self.loggers = {}
        self.log_dir = "logs"
        self.log_level = logging.INFO
        self.max_bytes = 10 * 1024 * 1024  # 10MB
        self.backup_count = 5
        self.initialized = False

    def init_app(self, log_level="info", log_dir='logs'):
        """init logger system"""
        # set logger level
        self.log_level = LOG_LEVELS.get(log_level.lower(), logging.INFO)

        # set logger dir
        if log_dir:
            self.log_dir = log_dir

        # make sure logger dir exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.initialized = True
        return self

    def get_logger(self, name) -> logging.Logger:
        """get or create specify logger"""
        if not self.initialized:
            self.init_app()

        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self.log_level)
            # Attach handlers only to the custom logger; do not affect the root logger
            if not logger.handlers:
                # Console handler
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(self.log_level)
                console_formatter = ColorFormatter(
                    '%(levelname)s:     %(asctime)s %(lineno)d %(name)s - %(message)s',
                    datefmt='%y-%m-%d %H:%M'
                )
                console_handler.setFormatter(console_formatter)
                logger.addHandler(console_handler)
                # File handler
                file_path = os.path.join(self.log_dir, f"{name}.log")
                file_handler = RotatingFileHandler(
                    file_path,
                    maxBytes=self.max_bytes,
                    backupCount=self.backup_count
                )
                file_handler.setLevel(self.log_level)
                formatter = logging.Formatter(
                    '%(levelname)s: %(asctime)s %(lineno)d  %(name)s - %(message)s'
                )
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            self.loggers[name] = logger

        return self.loggers[name]

    def set_level(self, level):
        """dynamic change logger level"""
        if isinstance(level, str):
            level = LOG_LEVELS.get(level.lower(), logging.INFO)

        # update logger level
        self.log_level = level

        # update all logger
        for logger in self.loggers.values():
            logger.setLevel(level)
            for handler in logger.handlers:
                handler.setLevel(level)


# Create logger manager singleton instance
logger_manager = LoggerManager()

LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# Initialize logger manager
logger_manager.init_app(log_level=LOG_LEVEL, log_dir=os.getenv("LOG_DIR", "logs"))


# Convenience function to get a logger
def get_logger(name="app"):
    return logger_manager.get_logger(name)