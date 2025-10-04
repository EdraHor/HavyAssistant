"""
Оверлей загрузки - показывается поверх окна при длительных операциях
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPainter, QColor


class LoadingOverlay(QWidget):
    """Виджет оверлея с индикатором загрузки"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Настройка виджета
        self.setVisible(False)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        # Layout
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        # Текст загрузки
        self.loading_label = QLabel("Загрузка...")
        self.loading_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.loading_label.setStyleSheet("""
            QLabel {
                color: white;
                background-color: rgba(13, 115, 119, 220);
                padding: 20px 40px;
                border-radius: 10px;
            }
        """)
        self.loading_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.loading_label)

    def paintEvent(self, event):
        """Рисуем полупрозрачный фон"""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 150))

    def show_loading(self, message: str = "Загрузка..."):
        """Показать оверлей с сообщением"""
        self.loading_label.setText(message)
        self.setGeometry(self.parent().rect())
        self.raise_()
        self.show()

        # Блокируем главное окно
        if self.parent():
            self.parent().setEnabled(False)

    def hide_loading(self):
        """Скрыть оверлей"""
        self.hide()

        # Разблокируем главное окно
        if self.parent():
            self.parent().setEnabled(True)