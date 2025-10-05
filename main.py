"""
Голосовой ассистент с распознаванием речи на GPU
Точка входа приложения с глобальной обработкой ошибок
"""

import sys
import logging
import traceback
from PyQt5.QtWidgets import QApplication, QMessageBox
from PyQt5.QtCore import Qt
from gui.main_window import MainWindow
from utils.logger import setup_logger
from utils.model_downloader import check_models, get_models_info

# Глобальный логгер
logger = None

def exception_hook(exctype, value, tb):
    """Глобальный обработчик необработанных исключений"""
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))

    if logger:
        logger.critical(f"Необработанное исключение:\n{error_msg}")
    else:
        print(f"CRITICAL ERROR:\n{error_msg}", file=sys.stderr)

    # Показываем диалог пользователю если возможно
    try:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Критическая ошибка")
        msg.setText(f"Произошла непредвиденная ошибка:\n\n{exctype.__name__}: {value}")
        msg.setDetailedText(error_msg)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    except:
        pass

    # Вызываем стандартный обработчик
    sys.__excepthook__(exctype, value, tb)

def setup_exception_handling():
    """Настройка глобальной обработки исключений"""
    # Python exceptions
    sys.excepthook = exception_hook

    # Qt exceptions (в слотах и сигналах)
    def qt_exception_handler(exc_type, exc_value, exc_tb):
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        if logger:
            logger.error(f"Qt исключение:\n{error_msg}")
        return False  # Не подавляем исключение

    # Подключаем Qt exception handler если доступен
    try:
        from PyQt5.QtCore import qInstallMessageHandler

        def qt_message_handler(mode, context, message):
            if mode == 3:  # QtCriticalMsg
                if logger:
                    logger.error(f"Qt Critical: {message}")
            elif mode == 2:  # QtWarningMsg
                if logger:
                    logger.warning(f"Qt Warning: {message}")

        qInstallMessageHandler(qt_message_handler)
    except:
        pass

def main():
    global logger

    try:
        # Настройка логирования
        logger = setup_logger()
        logger.info("=" * 60)
        logger.info("=== Запуск голосового ассистента ===")
        logger.info("=" * 60)

        # Настройка обработки исключений
        setup_exception_handling()
        logger.info("✓ Глобальная обработка ошибок включена")

        # Проверка моделей при запуске
        status = check_models()
        if not all(status.values()):
            logger.info("Требуется загрузка моделей")
            logger.info(get_models_info())

        # Создание приложения
        app = QApplication(sys.argv)
        app.setApplicationName("Голосовой Ассистент")
        app.setStyle('Fusion')

        # Информация о моделях если нужно
        missing = [name for name, ready in status.items() if not ready]
        if missing:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setWindowTitle("Загрузка моделей")
            msg.setText(get_models_info())
            msg.setInformativeText("Нажмите OK для продолжения. Модели загрузятся автоматически.")
            msg.exec_()

        # Главное окно с обработкой ошибок
        try:
            window = MainWindow()
            window.show()
            logger.info("✓ Главное окно создано")
        except Exception as e:
            logger.critical(f"Ошибка создания главного окна: {e}", exc_info=True)
            QMessageBox.critical(None, "Ошибка запуска",
                f"Не удалось создать главное окно:\n\n{str(e)}\n\nПроверьте логи для деталей.")
            return 1

        # Запуск event loop
        logger.info("✓ Запуск главного цикла приложения")
        exit_code = app.exec_()

        logger.info(f"Приложение завершено с кодом {exit_code}")
        return exit_code

    except Exception as e:
        if logger:
            logger.critical(f"Критическая ошибка в main(): {e}", exc_info=True)
        else:
            print(f"CRITICAL: {e}", file=sys.stderr)
            traceback.print_exc()

        try:
            QMessageBox.critical(None, "Критическая ошибка",
                f"Не удалось запустить приложение:\n\n{str(e)}")
        except:
            pass

        return 1

if __name__ == "__main__":
    sys.exit(main())