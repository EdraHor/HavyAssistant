"""
Главное окно приложения (PyQt5)
"""

import logging
from gui.loading_overlay import LoadingOverlay
from datetime import datetime
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QComboBox, QLineEdit, QSlider, QPushButton,
                             QProgressBar, QTextEdit, QGroupBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSettings
from PyQt5.QtGui import QFont, QColor

from utils.config_loader import config
from services.audio_capture import AudioCaptureService
from services.wake_word import WakeWordService
from services.speech_recognition import SpeechRecognitionService
from services.llm_service import GeminiService
from services.notification_service import NotificationService

logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Главное окно голосового ассистента"""
    calibration_done = pyqtSignal()

    def __init__(self):
        super().__init__()

        # Сервисы
        self.audio_service = AudioCaptureService()
        self.wake_word_service = WakeWordService()
        self.speech_service = SpeechRecognitionService()
        self.gemini_service = GeminiService()
        self.notification_service = NotificationService()

        # Состояние
        self.is_listening = False
        self.selected_device = None

        # QSettings для сохранения настроек
        self.settings = QSettings("VoiceAssistant", "HavyAssistant")

        # Флаг: НЕ сохранять во время инициализации
        self._is_initializing = True

        # Настройка окна
        self.setWindowTitle("Голосовой Ассистент (GPU)")
        self.setGeometry(100, 100,
                         config.get('ui.window_width', 700),
                         config.get('ui.window_height', 550))

        # Создание UI
        self._init_ui()
        self._setup_callbacks()
        self._apply_theme()

        # Оверлей загрузки
        self.loading_overlay = LoadingOverlay(self)

        # Загрузка сохраненных настроек
        self._load_settings()

        # ВАЖНО: Теперь можно сохранять
        self._is_initializing = False

        # Обновляем счетчик истории
        if hasattr(self, 'history_label'):
            history_count = self.gemini_service.get_history_length()
            self.history_label.setText(f"История: {history_count} сообщений")
            if history_count > 0:
                self._add_log(f"Загружена история: {history_count} сообщений")

        logger.info("GUI инициализирован")

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
        devices = self.audio_service.get_audio_devices()
        for device in devices:
            self.device_combo.addItem(device['name'], device['index'])
        mic_layout.addWidget(self.device_combo)
        self.device_combo.currentIndexChanged.connect(lambda: self._save_settings())

        mic_group.setLayout(mic_layout)
        layout.addWidget(mic_group)

        # 2. Ключевое слово
        wake_group = QGroupBox("Ключевое слово")
        wake_layout = QVBoxLayout()

        self.wake_word_input = QLineEdit()
        self.wake_word_input.editingFinished.connect(self._save_settings)
        wake_layout.addWidget(self.wake_word_input)

        wake_group.setLayout(wake_layout)
        layout.addWidget(wake_group)

        # 3. Чувствительность
        sens_group = QGroupBox("Чувствительность")
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
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        sens_layout.addWidget(self.sensitivity_slider)

        hint_label = QLabel("← Менее чувствительно  |  Более чувствительно →")
        hint_label.setStyleSheet("color: gray; font-size: 10px;")
        hint_label.setAlignment(Qt.AlignCenter)
        sens_layout.addWidget(hint_label)

        sens_group.setLayout(sens_layout)
        layout.addWidget(sens_group)

        # 4. Индикатор звука
        audio_group = QGroupBox("Уровень звука")
        audio_layout = QVBoxLayout()

        audio_header = QHBoxLayout()
        audio_header.addWidget(QLabel("Уровень звука:"))
        self.noise_floor_label = QLabel("Фон не откалиброван")
        self.noise_floor_label.setStyleSheet("color: gray; font-size: 11px;")
        audio_header.addStretch()
        audio_header.addWidget(self.noise_floor_label)
        audio_layout.addLayout(audio_header)

        self.audio_progress = QProgressBar()
        self.audio_progress.setMaximum(100)
        self.audio_progress.setTextVisible(True)
        self.audio_progress.setFormat("Тишина")
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

        # Индикатор истории диалога
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
        self.start_btn.clicked.connect(self._toggle_listening)
        btn_layout.addWidget(self.start_btn)

        self.calibrate_btn = QPushButton("Калибровка")
        self.calibrate_btn.setMinimumHeight(40)
        self.calibrate_btn.setEnabled(False)
        self.calibrate_btn.clicked.connect(self._calibrate)
        btn_layout.addWidget(self.calibrate_btn)

        self.clear_btn = QPushButton("Очистить")
        self.clear_btn.setMinimumHeight(40)
        self.clear_btn.clicked.connect(self._clear_log)
        btn_layout.addWidget(self.clear_btn)

        self.reset_context_btn = QPushButton("Сброс контекста")
        self.reset_context_btn.setMinimumHeight(40)
        self.reset_context_btn.clicked.connect(self._reset_context)
        btn_layout.addWidget(self.reset_context_btn)

        layout.addLayout(btn_layout)

        central_widget.setLayout(layout)

    def _setup_callbacks(self):
        """Настройка callback'ов для сервисов"""
        # Wake Word
        self.wake_word_service.set_wake_word_callback(self._on_wake_word_detected)
        self.wake_word_service.set_audio_level_callback(self._on_audio_level_wake)

        # Speech Recognition
        self.speech_service.set_speech_recognized_callback(self._on_speech_recognized)
        self.speech_service.set_audio_level_callback(self._on_audio_level_speech)
        self.speech_service.set_noise_floor_callback(self._on_noise_floor_calibrated)

        # Gemini
        self.gemini_service.set_response_callback(self._on_gemini_response)
        self.gemini_service.set_error_callback(self._on_gemini_error)

    def _load_settings(self):
        """Загрузка сохраненных настроек"""
        logger.info("Загрузка настроек из реестра...")

        # Микрофон
        saved_device = self.settings.value("device_index", type=int)
        if saved_device is not None:
            for i in range(self.device_combo.count()):
                if self.device_combo.itemData(i) == saved_device:
                    self.device_combo.setCurrentIndex(i)
                    logger.info(f"Загружен микрофон: index={saved_device}")
                    break

        # Чувствительность
        saved_sensitivity = self.settings.value("sensitivity", type=int)
        if saved_sensitivity is not None:
            self.sensitivity_slider.setValue(saved_sensitivity)
            logger.info(f"Загружена чувствительность: {saved_sensitivity}")

        # Ключевое слово
        saved_wake_word = self.settings.value("wake_word")
        if saved_wake_word:
            self.wake_word_input.setText(str(saved_wake_word))
            logger.info(f"Загружено ключевое слово: '{saved_wake_word}'")
        else:
            default_wake_word = config.get('wake_word.keyword', 'привет ассистент')
            self.wake_word_input.setText(default_wake_word)
            logger.info(f"Дефолтное ключевое слово: '{default_wake_word}'")

        # Подключаем сигналы ПОСЛЕ загрузки всех значений
        self.wake_word_input.editingFinished.connect(self._save_settings)
        self.device_combo.currentIndexChanged.connect(self._save_settings)

    def _save_settings(self):
        """Сохранение текущих настроек"""
        # НЕ сохраняем во время инициализации!
        if hasattr(self, '_is_initializing') and self._is_initializing:
            return

        device = self.device_combo.currentData()
        sensitivity = self.sensitivity_slider.value()
        wake_word = self.wake_word_input.text()

        logger.info(f"Сохранение: device={device}, sensitivity={sensitivity}, wake_word='{wake_word}'")

        self.settings.setValue("device_index", device)
        self.settings.setValue("sensitivity", sensitivity)
        self.settings.setValue("wake_word", wake_word)
        self.settings.sync()

        logger.debug("Настройки сохранены")

    def _apply_theme(self):
        """Применить темную тему"""
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

    def _toggle_listening(self):
        """Переключение состояния прослушивания"""
        if self.is_listening:
            self._stop_listening()
        else:
            self._start_listening()

    def _start_listening(self):
        """Запуск прослушивания"""
        device_index = self.device_combo.currentData()

        if device_index is None:
            QMessageBox.warning(self, "Ошибка", "Выберите микрофон")
            return

        try:
            # Показываем оверлей
            self.loading_overlay.show_loading("Загрузка моделей...")

            # Инициализация моделей
            self._set_status("Загрузка Vosk...", "orange")
            QTimer.singleShot(100, lambda: self._init_vosk(device_index))

        except Exception as e:
            self.loading_overlay.hide_loading()
            logger.error(f"Ошибка запуска: {e}")
            QMessageBox.critical(self, "Ошибка", f"Не удалось запустить:\n{str(e)}")
            self._set_status(f"Ошибка: {str(e)}", "red")

    def _reset_context(self):
        """Сброс контекста - создание новой сессии"""
        history_length = self.gemini_service.get_history_length()

        reply = QMessageBox.question(
            self,
            "Сброс контекста",
            f"Создать новую сессию?\n\nТекущая история: {history_length} сообщений\nСтарая сессия сохранится в базе данных.",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.gemini_service.clear_history()
            self._add_log(f"Новая сессия создана (старая сохранена)")

            # Обновляем индикатор
            if hasattr(self, 'history_label'):
                self.history_label.setText("История: 0 сообщений")

    def _init_vosk(self, device_index):
        """Инициализация Vosk в отдельном потоке"""
        from PyQt5.QtCore import QThread, pyqtSignal

        class VoskLoaderThread(QThread):
            success = pyqtSignal()
            error = pyqtSignal(str)
            progress = pyqtSignal(str)

            def __init__(self, service, wake_word):
                super().__init__()
                self.service = service
                self.wake_word = wake_word

            def run(self):
                try:
                    self.progress.emit("Загрузка Vosk...")
                    self.service.initialize(self.wake_word)
                    self.success.emit()
                except Exception as e:
                    self.error.emit(str(e))

        wake_word = self.wake_word_input.text()
        self.loader_thread = VoskLoaderThread(self.wake_word_service, wake_word)
        self.loader_thread.progress.connect(lambda msg: self.loading_overlay.show_loading(msg))
        self.loader_thread.success.connect(lambda: self._init_whisper(device_index, wake_word))
        self.loader_thread.error.connect(self._on_init_error)
        self.loader_thread.start()

    def _on_init_error(self, error_msg):
        """Обработка ошибки инициализации"""
        self.loading_overlay.hide_loading()
        logger.error(f"Ошибка инициализации: {error_msg}")
        QMessageBox.critical(self, "Ошибка", f"Не удалось инициализировать:\n{error_msg}")
        self._set_status("Ошибка", "red")

    def _init_whisper(self, device_index, wake_word):
        """Инициализация Whisper в отдельном потоке"""
        from PyQt5.QtCore import QThread, pyqtSignal

        class WhisperLoaderThread(QThread):
            success = pyqtSignal()
            error = pyqtSignal(str)
            progress = pyqtSignal(str)

            def __init__(self, service, sensitivity):
                super().__init__()
                self.service = service
                self.sensitivity = sensitivity

            def run(self):
                try:
                    model_name = config.get('speech_recognition.model_name', 'large-v3')
                    self.progress.emit(f"Загрузка Whisper {model_name}...")
                    self.service.initialize()
                    self.service.update_sensitivity(self.sensitivity)
                    self.success.emit()
                except Exception as e:
                    self.error.emit(str(e))

        self.whisper_thread = WhisperLoaderThread(self.speech_service, self.sensitivity_slider.value())
        self.whisper_thread.progress.connect(lambda msg: self.loading_overlay.show_loading(msg))
        self.whisper_thread.success.connect(lambda: self._finalize_startup(device_index, wake_word))
        self.whisper_thread.error.connect(self._on_init_error)
        self.whisper_thread.start()

    def _finalize_startup(self, device_index, wake_word):
        """Завершение запуска после загрузки моделей"""
        try:
            # Скрываем оверлей
            self.loading_overlay.hide_loading()

            # Запуск захвата
            self.audio_service.start_capture(device_index, self._on_audio_data)

            self.is_listening = True
            self.start_btn.setText("Остановить")
            self.calibrate_btn.setEnabled(True)

            self._set_status(f"Слушаю... Скажите '{wake_word}'", "green")
            self._add_log(f"✅ Запущено! Микрофон: {self.device_combo.currentText()}")

            # Автокалибровка (если включена в конфиге)
            auto_calibrate = config.get('speech_recognition.auto_calibrate', True)
            if auto_calibrate:
                QTimer.singleShot(2000, self._calibrate)
            else:
                logger.info("Автокалибровка отключена")

        except Exception as e:
            logger.error(f"Ошибка запуска: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка запуска:\n{str(e)}")
            self._set_status("Ошибка", "red")

    def _stop_listening(self):
        """Остановка прослушивания"""
        try:
            # Останавливаем все сервисы в правильном порядке
            self.audio_service.stop_capture()
            self.speech_service.stop()
            self.wake_word_service.stop()

            # Отменяем все таймеры и потоки
            if hasattr(self, 'loader_thread') and self.loader_thread:
                if self.loader_thread.isRunning():
                    self.loader_thread.quit()
                    self.loader_thread.wait(1000)  # Ждем 1 секунду

            if hasattr(self, 'whisper_thread') and self.whisper_thread:
                if self.whisper_thread.isRunning():
                    self.whisper_thread.quit()
                    self.whisper_thread.wait(1000)

            # Сброс состояния
            self.is_listening = False
            self.start_btn.setText("Запустить")
            self.calibrate_btn.setEnabled(False)

            self._set_status("Остановлено", "gray")
            self._set_audio_level(0, "Тишина", "gray")
            self._add_log("⏹️ Остановлено")

        except Exception as e:
            logger.error(f"Ошибка при остановке: {e}")

    def _calibrate(self):
        """Калибровка фонового шума"""
        if not self.is_listening:
            return

        self._add_log("🔧 КАЛИБРОВКА: замолчите на 2 секунды...")
        self._set_status("Калибровка... Не говорите!", "orange")

        import threading
        def run_calibration():
            self.speech_service.calibrate_noise_floor(2.0)
            self.calibration_done.emit()  # Thread-safe сигнал

        threading.Thread(target=run_calibration, daemon=True).start()

    def _after_calibration(self):
        """Вызывается после калибровки"""
        wake_word = self.wake_word_input.text()
        self._set_status(f"Слушаю... Скажите '{wake_word}'", "green")

    def _on_audio_data(self, audio_data: bytes):
        """Обработка аудио данных"""
        if not self.wake_word_service.is_recording:
            self.wake_word_service.process_audio(audio_data)
        else:
            self.speech_service.process_audio(audio_data)

    def _on_wake_word_detected(self):
        """Wake Word обнаружено"""
        wake_word = self.wake_word_input.text()
        self._set_status("🎤 Ключевое слово обнаружено! Слушаю команду...", "blue")
        self._add_log("🔑 WAKE WORD обнаружено → переключение на Whisper")

        self.speech_service.start_recording()

    def _on_speech_recognized(self, text: str):
        """Речь распознана"""
        if text not in ["[тишина]", "[ошибка]"]:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._add_log(f"[{timestamp}] 💬 {text}")

            # Отправка в Gemini
            self._send_to_gemini(text)

        wake_word = self.wake_word_input.text()
        self._set_status(f"Слушаю... Скажите '{wake_word}'", "green")
        self.wake_word_service.stop()

    def _on_audio_level_wake(self, rms: float, status: str):
        """Уровень звука от Wake Word"""
        if not self.wake_word_service.is_recording:
            self._set_audio_level(rms, status)

    def _on_audio_level_speech(self, rms: float, status: str):
        """Уровень звука от Speech Recognition"""
        if self.speech_service.is_recording:
            self._set_audio_level(rms, status)

    def _on_noise_floor_calibrated(self, noise_floor: float):
        """Калибровка завершена"""
        self.noise_floor_label.setText(f"Фон: {noise_floor:.4f}")
        self._add_log(f"✅ КАЛИБРОВКА: фоновый уровень = {noise_floor:.4f}")

    def _send_to_gemini(self, query: str):
        """Отправка запроса в Gemini"""
        self._add_log(f"📤 Отправка в Gemini: {query}")
        self._set_status("Обработка запроса...", "orange")

        # Запуск в отдельном потоке
        import threading
        threading.Thread(
            target=self.gemini_service.send_query,
            args=(query,),
            daemon=True
        ).start()

    def _on_gemini_response(self, response: str):
        """Ответ от Gemini"""
        self._add_log(f"📥 Ответ Gemini: {response}")
        wake_word = self.wake_word_input.text()
        self._set_status(f"Слушаю... Скажите '{wake_word}'", "green")

        # Обновляем индикатор истории
        history_count = self.gemini_service.get_history_length()
        self.history_label.setText(f"История: {history_count} сообщений")

        # Показать уведомление
        self.notification_service.show_notification("Ответ ассистента", response)

    def _on_gemini_error(self, error: str):
        """Ошибка Gemini"""
        self._add_log(f"❌ Ошибка Gemini: {error}")
        self._set_status("Ошибка обработки запроса", "red")
        self.notification_service.show_error("Ошибка", f"Не удалось получить ответ:\n{error}")

    def _on_sensitivity_changed(self, value: int):
        """Изменение чувствительности"""
        self.sens_value_label.setText(f"{value}/10")
        self.speech_service.update_sensitivity(value)

        # НЕ сохраняем во время инициализации
        if not (hasattr(self, '_is_initializing') and self._is_initializing):
            self._save_settings()

    def _set_status(self, text: str, color: str):
        """Установить статус"""
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _set_audio_level(self, rms: float, status: str, color: str = None):
        """Установить уровень звука"""
        level = min(int(rms * 300), 100)
        self.audio_progress.setValue(level)
        self.audio_progress.setFormat(status)

        if color is None:
            if status == "Тишина":
                color = "gray"
            elif status == "Шум":
                color = "orange"
            else:
                color = "#0d7377"

        self.audio_progress.setStyleSheet(f"""
            QProgressBar::chunk {{
                background-color: {color};
            }}
        """)

    def _add_log(self, message: str):
        """Добавить запись в лог"""
        self.log_text.append(message)
        # Прокрутка вниз
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _clear_log(self):
        """Очистить лог"""
        self.log_text.clear()

    def closeEvent(self, event):
        """Закрытие окна"""
        self._save_settings()
        if self.is_listening:
            self.loading_overlay.show_loading("Остановка сервисов...")
            QTimer.singleShot(100, lambda: self._finalize_close(event))
        else:
            event.accept()

    def _finalize_close(self, event):
        """Финальное закрытие после остановки"""
        self._stop_listening()
        self._save_settings()
        self.loading_overlay.hide_loading()
        event.accept()
        self.close()