"""
Голосовой ассистент с распознаванием речи на GPU
Точка входа приложения
"""

import sys
import logging
from PyQt5.QtWidgets import QApplication, QMessageBox
from gui.main_window import MainWindow
from utils.logger import setup_logger
from utils.model_downloader import check_models, get_models_info

def main():
    # Настройка логирования
    setup_logger()
    logger = logging.getLogger(__name__)
    logger.info("=== Запуск голосового ассистента ===")

    # Проверка моделей при запуске
    status = check_models()
    if not all(status.values()):
        logger.info("Требуется загрузка моделей")
        logger.info(get_models_info())

    # Создание приложения
    app = QApplication(sys.argv)
    app.setApplicationName("Голосовой Ассистент")
    app.setStyle('Fusion')  # Современный стиль

    # Информация о моделях если нужно
    missing = [name for name, ready in status.items() if not ready]
    if missing:
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Загрузка моделей")
        msg.setText(get_models_info())
        msg.setInformativeText("Нажмите OK для продолжения. Модели загрузятся автоматически.")
        msg.exec_()

    # Главное окно
    window = MainWindow()
    window.show()

    # Запуск event loop
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()