"""
Настройка логирования с цветным выводом
С улучшенной обработкой ошибок
"""

import logging
import sys
from pathlib import Path

try:
    import colorlog
    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False
    print("WARNING: colorlog не установлен, цветной вывод недоступен")

from utils.config_loader import config

def setup_logger():
    """Настройка глобального логгера"""

    try:
        log_level = config.get('logging.level', 'INFO')
        log_format = config.get('logging.format')
        log_file = config.get('logging.file')
        console_enabled = config.get('logging.console', True)

        # Создаем root logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, log_level))

        # Очищаем существующие handlers
        logger.handlers = []

        # Консольный handler с цветами (если доступен colorlog)
        if console_enabled:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, log_level))

            if COLORLOG_AVAILABLE:
                # Цветной формат
                color_format = '%(log_color)s%(asctime)s - %(name)-20s - %(levelname)-8s%(reset)s - %(message)s'
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
            else:
                # Обычный формат без цветов
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s',
                    datefmt='%H:%M:%S'
                )

            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)

        # Файловый handler
        if log_file:
            try:
                # Создаем директорию для логов если нужно
                log_path = Path(log_file)
                log_path.parent.mkdir(parents=True, exist_ok=True)

                file_handler = logging.FileHandler(log_file, encoding='utf-8')
                file_handler.setLevel(getattr(logging, log_level))
                file_formatter = logging.Formatter(
                    log_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)

                logger.info(f"✓ Логирование в файл: {log_file}")

            except Exception as e:
                print(f"WARNING: Не удалось создать файловый handler: {e}", file=sys.stderr)

        # Логирование некритичных ошибок из библиотек
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)

        logger.info(f"✓ Логирование настроено (уровень: {log_level})")

        return logger

    except Exception as e:
        # Fallback на базовую настройку
        print(f"ERROR: Не удалось настроить логирование: {e}", file=sys.stderr)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        logger = logging.getLogger()
        logger.error(f"Ошибка настройки логирования: {e}", exc_info=True)
        return logger