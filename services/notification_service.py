"""
Сервис системных уведомлений
"""

import logging
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)

class NotificationService(QObject):
    """Сервис для показа уведомлений"""
    
    # Сигналы для потокобезопасности
    show_info_signal = pyqtSignal(str, str)
    show_error_signal = pyqtSignal(str, str)
    
    def __init__(self):
        super().__init__()
        
        # Подключаем сигналы к слотам
        self.show_info_signal.connect(self._show_info_box)
        self.show_error_signal.connect(self._show_error_box)
    
    def show_notification(self, title: str, message: str):
        """
        Показать информационное уведомление
        
        Args:
            title: Заголовок
            message: Сообщение
        """
        logger.info(f"Уведомление: {title} - {message}")
        self.show_info_signal.emit(title, message)
    
    def show_error(self, title: str, message: str):
        """
        Показать уведомление об ошибке
        
        Args:
            title: Заголовок
            message: Сообщение об ошибке
        """
        logger.error(f"Ошибка: {title} - {message}")
        self.show_error_signal.emit(title, message)
    
    def _show_info_box(self, title: str, message: str):
        """Показать MessageBox с информацией"""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
    
    def _show_error_box(self, title: str, message: str):
        """Показать MessageBox с ошибкой"""
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec_()
