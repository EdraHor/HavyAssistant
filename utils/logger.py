"""
Настройка логирования с цветным выводом
"""

import logging
import colorlog
from utils.config_loader import config

def setup_logger():
    """Настройка глобального логгера"""
    
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format')
    log_file = config.get('logging.file')
    console_enabled = config.get('logging.console', True)
    
    # Цветной формат для консоли
    color_format = '%(log_color)s%(asctime)s - %(name)-20s - %(levelname)-8s%(reset)s - %(message)s'
    
    # Создаем root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    
    # Очищаем существующие handlers
    logger.handlers = []
    
    # Консольный handler с цветами
    if console_enabled:
        console_handler = colorlog.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level))
        console_formatter = colorlog.ColoredFormatter(
            color_format,
            datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Файловый handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level))
        file_formatter = logging.Formatter(log_format, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger
