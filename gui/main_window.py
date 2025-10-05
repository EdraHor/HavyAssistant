"""
Главное окно приложения (PyQt5) - только UI
С реалтайм отображением калибровки и чувствительности
"""

import logging
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QLineEdit, QSlider, QPushButton,
                             QProgressBar, QTextEdit, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSettings
from PyQt5.QtGui import QFont, QPainter, QColor
from PyQt5.QtWidgets import QStyle

from gui.loading_overlay import LoadingOverlay
from services.assistant_controller import VoiceAssistantController, AssistantState
from services.notification_service import NotificationService
from utils.config_loader import config

logger = logging.getLogger(__name__)

class ThresholdProgressBar(QProgressBar):
    """Прогресс-бар с линией порога"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.threshold_value = 0
        self.noise_floor_value = 0

    def set_threshold(self, threshold: float):
        """Установить значение порога (0.0-1.0)"""
        self.threshold_value = min(int(threshold * 300), 100)
        self.update()

    def set_noise_floor(self, noise_floor: float):
        """Установить значение фона (0.0-1.0)"""
        self.noise_floor_value = min(int(noise_floor * 300), 100)
        self.update()

    def paintEvent(self, event):
        """Рисуем прогресс-бар с линиями порога и фона"""
        super().paintEvent(event)

        if self.threshold_value > 0 or self.noise_floor_value > 0:
            painter = QPainter(self)

            # Линия фона (серая, пунктирная)
            if self.noise_floor_value > 0:
                noise_x = int(self.width() * self.noise_floor_value / 100)
                painter.setPen(QColor(150, 150, 150))
                for y in range(0, self.height(), 4):
                    painter.drawLine(noise_x, y, noise_x, y + 2)

            # Линия порога (красная, сплошная)
            if self.threshold_value > 0:
                threshold_x = int(self.width() * self.threshold_value / 100)
                painter.setPen(QColor(255, 100, 100, 200))
                painter.drawLine(threshold_x, 0, threshold_x, self.height())

            painter.end()

class MainWindow(QMainWindow):
    """Главное окно голосового ассистента - только отображение"""

    # Сигналы для thread-safe обновления UI
    update_status_signal = pyqtSignal(str, str)
    update_audio_level_signal = pyqtSignal(float, str)
    add_log_signal = pyqtSignal(str)
    update_history_signal = pyqtSignal(int)
    update_noise_floor_signal = pyqtSignal(float)
    update_threshold_signal = pyqtSignal(float)
    show_notification_signal = pyqtSignal(str, str)
    show_error_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()

        try:
            # Контроллер (вся логика здесь)
            self.controller = VoiceAssistantController()

            # Сервис уведомлений
            self.notification_service = NotificationService()

            # QSettings для сохранения настроек
            self.settings = QSettings("VoiceAssistant", "HavyAssistant")

            # Флаг инициализации
            self._is_initializing = True

            # Текущие значения для отображения
            self.current_noise_floor = 0.0
            self.current_threshold = 0.0

            # Настройка окна
            self.setWindowTitle("Голосовой Ассистент (GPU)")
            self.setGeometry(100, 100,
                             config.get('ui.window_width', 700),
                             config.get('ui.window_height', 550))

            # Создание UI
            self._init_ui()
            self._apply_theme()

            # Оверлей загрузки
            self.loading_overlay = LoadingOverlay(self)

            # Подключение сигналов
            self._connect_signals()

            # Подключение к контроллеру
            self._connect_controller()

            # Загрузка настроек
            self._load_settings()

            # Конец инициализации
            self._is_initializing = False

            # Обновление истории
            history_count = self.controller.get_history_length()
            self.history_label.setText(f"История: {history_count} сообщений")
            if history_count > 0:
                self._add_log(f"Загружена история: {history_count} сообщений")

            logger.info("✓ GUI инициализирован")

        except Exception as e:
            logger.error(f"Ошибка инициализации GUI: {e}", exc_info=True)
            raise

    def _init_ui(self):
        """Создание интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 1. Выбор микрофона
        mic_group = QGroupBox("Микрофон")
        mic_layout = QVBoxLayout()

        self.device_combo = QComboBox()
        devices = self.controller.get_audio_devices()
        for device in devices:
            self.device_combo.addItem(device['name'], device['index'])
        mic_layout.addWidget(self.device_combo)

        mic_group.setLayout(mic_layout)
        layout.addWidget(mic_group)

        # 2. Ключевое слово
        wake_group = QGroupBox("Ключевое слово")
        wake_layout = QVBoxLayout()

        self.wake_word_input = QLineEdit()
        wake_layout.addWidget(self.wake_word_input)

        wake_group.setLayout(wake_layout)
        layout.addWidget(wake_group)

        # 3. Чувствительность
        sens_group = QGroupBox("Чувствительность (работает в реальном времени)")
        sens_layout = QVBoxLayout()

        sens_header = QHBoxLayout()
        sens_header.addWidget(QLabel("Чувствительность:"))
        self.sens_value_label = QLabel("5/10")
        sens_header.addStretch()
        sens_header.addWidget(self.sens_value_label)
        sens_layout.addLayout(sens_header)

        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(10)
        self.sensitivity_slider.setValue(config.get('speech_recognition.sensitivity', 5))
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_slider.setTickInterval(1)
        sens_layout.addWidget(self.sensitivity_slider)

        hint_label = QLabel("← Менее чувствительно  |  Более чувствительно →")
        hint_label.setStyleSheet("color: gray; font-size: 10px;")
        hint_label.setAlignment(Qt.AlignCenter)
        sens_layout.addWidget(hint_label)

        sens_group.setLayout(sens_layout)
        layout.addWidget(sens_group)

        # 4. Индикатор звука с калибровкой
        audio_group = QGroupBox("Уровень звука и калибровка")
        audio_layout = QVBoxLayout()

        # Заголовок с информацией
        audio_header = QHBoxLayout()
        audio_header.addWidget(QLabel("Текущий уровень:"))

        # Информация о калибровке (расширенная)
        calibration_info = QVBoxLayout()
        calibration_info.setSpacing(2)

        self.noise_floor_label = QLabel("Фон: не откалиброван")
        self.noise_floor_label.setStyleSheet("color: #888; font-size: 11px;")
        calibration_info.addWidget(self.noise_floor_label)

        self.threshold_label = QLabel("Порог: не установлен")
        self.threshold_label.setStyleSheet("color: #888; font-size: 11px;")
        calibration_info.addWidget(self.threshold_label)

        audio_header.addLayout(calibration_info)
        audio_header.addStretch()

        # Легенда
        legend_layout = QVBoxLayout()
        legend_layout.setSpacing(2)

        legend_noise = QLabel("━━ Фон (серый)")
        legend_noise.setStyleSheet("color: #888; font-size: 10px;")
        legend_layout.addWidget(legend_noise)

        legend_threshold = QLabel("━━ Порог (красный)")
        legend_threshold.setStyleSheet("color: #ff6464; font-size: 10px;")
        legend_layout.addWidget(legend_threshold)

        audio_header.addLayout(legend_layout)
        audio_layout.addLayout(audio_header)

        # Прогресс-бар с линиями порога
        self.audio_progress = ThresholdProgressBar()
        self.audio_progress.setMaximum(100)
        self.audio_progress.setTextVisible(True)
        self.audio_progress.setFormat("Тишина")
        self.audio_progress.setMinimumHeight(30)
        audio_layout.addWidget(self.audio_progress)

        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)

        # 5. Статус
        status_group = QGroupBox("Статус")
        status_layout = QVBoxLayout()

        status_header = QHBoxLayout()
        self.status_label = QLabel("Готов к работе")
        self.status_label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        self.status_label.setStyleSheet("color: gray;")
        status_header.addWidget(self.status_label)

        self.history_label = QLabel("История: 0 сообщений")
        self.history_label.setStyleSheet("color: gray; font-size: 10px;")
        status_header.addStretch()
        status_header.addWidget(self.history_label)

        status_layout.addLayout(status_header)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 6. Лог команд
        log_group = QGroupBox("Распознанные команды")
        log_layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # 7. Кнопки управления
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("Запустить")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        btn_layout.addWidget(self.start_btn)

        self.calibrate_btn = QPushButton("Калибровка (2с)")
        self.calibrate_btn.setMinimumHeight(40)
        self.calibrate_btn.setEnabled(False)
        btn_layout.addWidget(self.calibrate_btn)

        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.clear_btn)

        self.reset_context_btn = QPushButton("Сброс контекста")
        self.reset_context_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.reset_context_btn)

        layout.addLayout(btn_layout)

        central_widget.setLayout(layout)

    def _connect_signals(self):
        """Подключение сигналов UI"""
        # Кнопки
        self.start_btn.clicked.connect(self._on_start_stop_clicked)
        self.calibrate_btn.clicked.connect(self._on_calibrate_clicked)
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        self.reset_context_btn.clicked.connect(self._on_reset_context_clicked)

        # Настройки (только после загрузки)
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)

        # Thread-safe сигналы
        self.update_status_signal.connect(self._set_status)
        self.update_audio_level_signal.connect(self._set_audio_level)
        self.add_log_signal.connect(self._add_log)
        self.update_history_signal.connect(self._update_history)
        self.update_noise_floor_signal.connect(self._update_noise_floor)
        self.update_threshold_signal.connect(self._update_threshold)
        self.show_notification_signal.connect(self._show_notification)
        self.show_error_signal.connect(self._show_error)

    def _connect_controller(self):
        """Подключение к контроллеру"""
        self.controller.set_callback('on_state_changed', self._on_state_changed)
        self.controller.set_callback('on_status_update', lambda text, color: self.update_status_signal.emit(text, color))
        self.controller.set_callback('on_audio_level', lambda rms, status: self.update_audio_level_signal.emit(rms, status))
        self.controller.set_callback('on_log', lambda msg: self.add_log_signal.emit(msg))
        self.controller.set_callback('on_history_updated', lambda count: self.update_history_signal.emit(count))
        self.controller.set_callback('on_noise_floor_calibrated', lambda floor: self.update_noise_floor_signal.emit(floor))
        self.controller.set_callback('on_threshold_updated', lambda threshold: self.update_threshold_signal.emit(threshold))
        self.controller.set_callback('on_llm_response', self._on_llm_response)
        self.controller.set_callback('on_llm_error', lambda error: self.show_error_signal.emit("Ошибка Gemini", error))
        self.controller.set_callback('on_error', lambda error: self.show_error_signal.emit("Ошибка", error))

    def _load_settings(self):
        """Загрузка настроек"""
        try:
            # Микрофон
            saved_device = self.settings.value("device_index", type=int)
            if saved_device is not None:
                for i in range(self.device_combo.count()):
                    if self.device_combo.itemData(i) == saved_device:
                        self.device_combo.setCurrentIndex(i)
                        break

            # Чувствительность
            saved_sensitivity = self.settings.value("sensitivity", type=int)
            if saved_sensitivity is not None:
                self.sensitivity_slider.setValue(saved_sensitivity)

            # Ключевое слово
            saved_wake_word = self.settings.value("wake_word")
            if saved_wake_word:
                self.wake_word_input.setText(str(saved_wake_word))
            else:
                default_wake_word = config.get('wake_word.keyword', 'привет ассистент')
                self.wake_word_input.setText(default_wake_word)

            # Подключаем сигналы ПОСЛЕ загрузки
            self.wake_word_input.editingFinished.connect(self._save_settings)
            self.device_combo.currentIndexChanged.connect(self._save_settings)

            logger.info("✓ Настройки загружены")

        except Exception as e:
            logger.error(f"Ошибка загрузки настроек: {e}", exc_info=True)

    def _save_settings(self):
        """Сохранение настроек"""
        if self._is_initializing:
            return

        try:
            self.settings.setValue("device_index", self.device_combo.currentData())
            self.settings.setValue("sensitivity", self.sensitivity_slider.value())
            self.settings.setValue("wake_word", self.wake_word_input.text())
            self.settings.sync()
            logger.debug("✓ Настройки сохранены")
        except Exception as e:
            logger.error(f"Ошибка сохранения настроек: {e}", exc_info=True)

    def _apply_theme(self):
        """Применить тему"""
        theme = config.get('ui.theme', 'dark')
        if theme == 'dark':
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #2b2b2b;
                    color: #ffffff;
                }
                QGroupBox {
                    border: 1px solid #444;
                    border-radius: 5px;
                    margin-top: 10px;
                    padding-top: 10px;
                    font-weight: bold;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px;
                }
                QLineEdit, QComboBox, QTextEdit {
                    background-color: #3c3c3c;
                    border: 1px solid #555;
                    border-radius: 3px;
                    padding: 5px;
                    color: #ffffff;
                }
                QPushButton {
                    background-color: #0d7377;
                    border: none;
                    border-radius: 5px;
                    color: white;
                    padding: 8px;
                }
                QPushButton:hover {
                    background-color: #14ffec;
                    color: #000;
                }
                QPushButton:disabled {
                    background-color: #444;
                    color: #888;
                }
                QProgressBar {
                    border: 1px solid #555;
                    border-radius: 3px;
                    background-color: #3c3c3c;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #0d7377;
                }
                QSlider::groove:horizontal {
                    border: 1px solid #555;
                    height: 8px;
                    background: #3c3c3c;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #14ffec;
                    border: 1px solid #0d7377;
                    width: 18px;
                    margin: -5px 0;
                    border-radius: 9px;
                }
            """)

    # === Обработчики UI событий ===

    def _on_start_stop_clicked(self):
        """Запуск/остановка"""
        try:
            if self.controller.state == AssistantState.STOPPED:
                device_index = self.device_combo.currentData()
                if device_index is None:
                    QMessageBox.warning(self, "Ошибка", "Выберите микрофон")
                    return

                wake_word = self.wake_word_input.text()
                self.loading_overlay.show_loading("Инициализация...")

                success = self.controller.start(device_index, wake_word)
                if not success:
                    self.loading_overlay.hide_loading()
            else:
                self.controller.stop()
        except Exception as e:
            logger.error(f"Ошибка start/stop: {e}", exc_info=True)
            self.loading_overlay.hide_loading()

    def _on_calibrate_clicked(self):
        """Калибровка"""
        try:
            self.controller.calibrate_noise_floor(2.0)
        except Exception as e:
            logger.error(f"Ошибка калибровки: {e}", exc_info=True)

    def _on_clear_clicked(self):
        """Очистка лога"""
        self.log_text.clear()

    def _on_reset_context_clicked(self):
        """Сброс контекста"""
        try:
            history_length = self.controller.get_history_length()
            reply = QMessageBox.question(
                self,
                "Сброс контекста",
                f"Создать новую сессию?\n\nТекущая история: {history_length} сообщений\nСтарая сессия сохранится в базе данных.",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.controller.reset_context()
        except Exception as e:
            logger.error(f"Ошибка сброса контекста: {e}", exc_info=True)

    def _on_sensitivity_changed(self, value: int):
        """Изменение чувствительности - работает в реалтайм"""
        try:
            self.sens_value_label.setText(f"{value}/10")

            # Обновляем в контроллере (работает даже во время записи!)
            self.controller.update_sensitivity(value)

            # Обновляем отображение порога
            self._update_threshold_display()

            if not self._is_initializing:
                self._save_settings()
        except Exception as e:
            logger.error(f"Ошибка изменения чувствительности: {e}", exc_info=True)

    def _update_threshold_display(self):
        """Обновить отображение порога на основе текущих значений"""
        try:
            # Получаем актуальный порог из сервиса
            threshold = self.controller.speech_service.get_voice_threshold()
            self.current_threshold = threshold

            # Обновляем label
            if threshold > 0:
                self.threshold_label.setText(f"Порог: {threshold:.4f}")
                self.threshold_label.setStyleSheet("color: #ff6464; font-size: 11px;")
            else:
                self.threshold_label.setText(f"Порог: не установлен")
                self.threshold_label.setStyleSheet("color: #888; font-size: 11px;")

            # Обновляем линию на прогресс-баре
            self.audio_progress.set_threshold(threshold)

            logger.debug(f"Порог обновлен: {threshold:.4f}")

        except Exception as e:
            logger.error(f"Ошибка обновления порога: {e}", exc_info=True)

    # === Обработчики событий контроллера ===

    def _on_state_changed(self, new_state: AssistantState):
        """Изменение состояния"""
        try:
            if new_state == AssistantState.STOPPED:
                self.start_btn.setText("Запустить")
                self.calibrate_btn.setEnabled(False)
                self.loading_overlay.hide_loading()
            elif new_state == AssistantState.INITIALIZING:
                self.loading_overlay.show_loading("Загрузка моделей...")
            elif new_state == AssistantState.LISTENING_WAKE_WORD:
                self.start_btn.setText("Остановить")
                self.calibrate_btn.setEnabled(True)
                self.loading_overlay.hide_loading()
                # Обновляем порог при старте
                self._update_threshold_display()
            elif new_state == AssistantState.RECORDING_COMMAND:
                pass  # Индикация в статусе
            elif new_state == AssistantState.PROCESSING_LLM:
                pass  # Индикация в статусе
        except Exception as e:
            logger.error(f"Ошибка обработки состояния: {e}", exc_info=True)

    def _on_llm_response(self, response: str):
        """Ответ LLM"""
        try:
            self.show_notification_signal.emit("Ответ ассистента", response)
        except Exception as e:
            logger.error(f"Ошибка обработки ответа: {e}", exc_info=True)

    # === UI обновления (thread-safe) ===

    def _set_status(self, text: str, color: str):
        """Установить статус"""
        try:
            self.status_label.setText(text)
            self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        except Exception as e:
            logger.error(f"Ошибка установки статуса: {e}", exc_info=True)

    def _set_audio_level(self, rms: float, status: str):
        """Установить уровень звука"""
        try:
            level = min(int(rms * 300), 100)
            self.audio_progress.setValue(level)
            self.audio_progress.setFormat(f"{status} ({rms:.4f})")

            color = "gray" if status == "Тишина" else "orange" if status == "Шум" else "#0d7377"
            self.audio_progress.setStyleSheet(f"""
                QProgressBar::chunk {{
                    background-color: {color};
                }}
            """)
        except Exception as e:
            logger.error(f"Ошибка установки уровня звука: {e}", exc_info=True)

    def _add_log(self, message: str):
        """Добавить в лог"""
        try:
            self.log_text.append(message)
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception as e:
            logger.error(f"Ошибка добавления в лог: {e}", exc_info=True)

    def _update_history(self, count: int):
        """Обновить индикатор истории"""
        try:
            self.history_label.setText(f"История: {count} сообщений")
        except Exception as e:
            logger.error(f"Ошибка обновления истории: {e}", exc_info=True)

    def _update_noise_floor(self, floor: float):
        """Обновить индикатор калибровки"""
        try:
            self.current_noise_floor = floor

            if floor > 0:
                self.noise_floor_label.setText(f"Фон: {floor:.4f} ✓")
                self.noise_floor_label.setStyleSheet("color: #4CAF50; font-size: 11px; font-weight: bold;")
            else:
                self.noise_floor_label.setText(f"Фон: не откалиброван")
                self.noise_floor_label.setStyleSheet("color: #888; font-size: 11px;")

            # Обновляем линию фона на прогресс-баре
            self.audio_progress.set_noise_floor(floor)

            # Также обновляем порог (он зависит от фона)
            self._update_threshold_display()

            logger.info(f"✓ Noise floor обновлен в UI: {floor:.4f}")

        except Exception as e:
            logger.error(f"Ошибка обновления noise floor: {e}", exc_info=True)

    def _update_threshold(self, threshold: float):
        """Обновить порог (вызывается при изменении чувствительности)"""
        try:
            self.current_threshold = threshold

            if threshold > 0:
                self.threshold_label.setText(f"Порог: {threshold:.4f}")
                self.threshold_label.setStyleSheet("color: #ff6464; font-size: 11px; font-weight: bold;")
            else:
                self.threshold_label.setText(f"Порог: не установлен")
                self.threshold_label.setStyleSheet("color: #888; font-size: 11px;")

            self.audio_progress.set_threshold(threshold)

            logger.info(f"✓ Порог обновлен в UI: {threshold:.4f}")

        except Exception as e:
            logger.error(f"Ошибка обновления порога: {e}", exc_info=True)

    def _show_notification(self, title: str, message: str):
        """Показать уведомление"""
        try:
            self.notification_service.show_notification(title, message)
        except Exception as e:
            logger.error(f"Ошибка показа уведомления: {e}", exc_info=True)

    def _show_error(self, title: str, message: str):
        """Показать ошибку"""
        try:
            self.notification_service.show_error(title, message)
        except Exception as e:
            logger.error(f"Ошибка показа ошибки: {e}", exc_info=True)

    # === Закрытие ===

    def closeEvent(self, event):
        """Закрытие окна"""
        try:
            self._save_settings()
            if self.controller.state != AssistantState.STOPPED:
                self.loading_overlay.show_loading("Остановка...")
                QTimer.singleShot(100, lambda: self._finalize_close(event))
            else:
                event.accept()
        except Exception as e:
            logger.error(f"Ошибка при закрытии: {e}", exc_info=True)
            event.accept()

    def _finalize_close(self, event):
        """Финальное закрытие"""
        try:
            self.controller.stop()
            self.loading_overlay.hide_loading()
            event.accept()
        except Exception as e:
            logger.error(f"Ошибка финального закрытия: {e}", exc_info=True)
            event.accept()