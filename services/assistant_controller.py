"""
Контроллер голосового ассистента - вся бизнес-логика
Может работать без UI
"""

import logging
import threading
from typing import Optional, Callable, Dict
from enum import Enum

from utils.config_loader import config
from services.audio_capture import AudioCaptureService
from services.wake_word import WakeWordService
from services.speech_recognition import SpeechRecognitionService
from services.llm_service import GeminiService

logger = logging.getLogger(__name__)

class AssistantState(Enum):
    """Состояния ассистента"""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    LISTENING_WAKE_WORD = "listening_wake_word"
    RECORDING_COMMAND = "recording_command"
    PROCESSING_LLM = "processing_llm"
    ERROR = "error"

class VoiceAssistantController:
    """
    Основной контроллер голосового ассистента
    Управляет всеми сервисами и координирует их работу
    """
    
    def __init__(self):
        # Сервисы
        self.audio_service = AudioCaptureService()
        self.wake_word_service = WakeWordService()
        self.speech_service = SpeechRecognitionService()
        self.gemini_service = GeminiService()
        
        # Состояние
        self.state = AssistantState.STOPPED
        self.selected_device_index = None
        self.wake_word = config.get('wake_word.keyword', 'привет ассистент')
        self.sensitivity = config.get('speech_recognition.sensitivity', 5)
        
        # Callbacks для UI или других компонентов
        self.callbacks = {
            'on_state_changed': None,
            'on_status_update': None,
            'on_audio_level': None,
            'on_wake_word_detected': None,
            'on_speech_recognized': None,
            'on_llm_response': None,
            'on_llm_error': None,
            'on_error': None,
            'on_log': None,
            'on_noise_floor_calibrated': None,
            'on_history_updated': None,
            'on_threshold_updated': None,
        }

        # Настройка внутренних коллбэков
        self._setup_service_callbacks()

        logger.info("✓ VoiceAssistantController инициализирован")

    def _setup_service_callbacks(self):
        """Настройка callbacks от сервисов"""
        # Wake Word
        self.wake_word_service.set_wake_word_callback(self._on_wake_word_detected)
        self.wake_word_service.set_audio_level_callback(self._on_audio_level_wake)

        # Speech Recognition
        self.speech_service.set_speech_recognized_callback(self._on_speech_recognized)
        self.speech_service.set_audio_level_callback(self._on_audio_level_speech)
        self.speech_service.set_noise_floor_callback(self._on_noise_floor_calibrated)

        # Gemini
        self.gemini_service.set_response_callback(self._on_llm_response)
        self.gemini_service.set_error_callback(self._on_llm_error)

    def set_callback(self, event: str, callback: Callable):
        """Установить callback для события"""
        if event in self.callbacks:
            self.callbacks[event] = callback
            logger.debug(f"Установлен callback для '{event}'")
        else:
            logger.warning(f"Неизвестный callback: '{event}'")

    def _emit(self, event: str, *args, **kwargs):
        """Вызвать callback если установлен"""
        try:
            if self.callbacks.get(event):
                self.callbacks[event](*args, **kwargs)
        except Exception as e:
            logger.error(f"Ошибка в callback '{event}': {e}", exc_info=True)

    def _set_state(self, new_state: AssistantState):
        """Изменить состояние"""
        old_state = self.state
        self.state = new_state
        logger.info(f"Состояние: {old_state.value} -> {new_state.value}")
        self._emit('on_state_changed', new_state)

    def _log(self, message: str):
        """Вывести лог"""
        logger.info(message)
        self._emit('on_log', message)

    def get_audio_devices(self):
        """Получить список аудио устройств"""
        try:
            return self.audio_service.get_audio_devices()
        except Exception as e:
            logger.error(f"Ошибка получения устройств: {e}", exc_info=True)
            return []

    def start(self, device_index: int, wake_word: Optional[str] = None):
        """
        Запустить ассистент

        Args:
            device_index: Индекс аудио устройства
            wake_word: Ключевое слово (опционально)
        """
        if self.state != AssistantState.STOPPED:
            logger.warning("Ассистент уже запущен")
            return False

        try:
            self._set_state(AssistantState.INITIALIZING)
            self.selected_device_index = device_index

            if wake_word:
                self.wake_word = wake_word

            # Запуск инициализации в отдельном потоке
            init_thread = threading.Thread(
                target=self._initialize_services,
                daemon=True
            )
            init_thread.start()

            return True

        except Exception as e:
            logger.error(f"Ошибка запуска: {e}", exc_info=True)
            self._set_state(AssistantState.ERROR)
            self._emit('on_error', f"Ошибка запуска: {str(e)}")
            return False

    def _initialize_services(self):
        """Инициализация сервисов (в отдельном потоке)"""
        try:
            # Vosk
            self._emit('on_status_update', "Загрузка Vosk...", "orange")
            self.wake_word_service.initialize(self.wake_word)
            logger.info("✓ Vosk инициализирован")

            # Whisper
            model_name = config.get('speech_recognition.model_name', 'large-v3')
            self._emit('on_status_update', f"Загрузка Whisper {model_name}...", "orange")
            self.speech_service.initialize()
            self.speech_service.update_sensitivity(self.sensitivity)
            logger.info("✓ Whisper инициализирован")

            # Запуск захвата аудио
            self.audio_service.start_capture(
                self.selected_device_index,
                self._on_audio_data
            )
            logger.info("✓ Захват аудио запущен")

            # Готово
            self._set_state(AssistantState.LISTENING_WAKE_WORD)
            self._emit('on_status_update', f"Слушаю... Скажите '{self.wake_word}'", "green")
            self._log(f"✅ Запущено! Ключевое слово: '{self.wake_word}'")

            # Автокалибровка
            if config.get('speech_recognition.auto_calibrate', True):
                threading.Timer(2.0, self.calibrate_noise_floor).start()

        except Exception as e:
            logger.error(f"Ошибка инициализации: {e}", exc_info=True)
            self._set_state(AssistantState.ERROR)
            self._emit('on_error', f"Ошибка инициализации: {str(e)}")
            self.stop()

    def stop(self):
        """Остановить ассистент"""
        try:
            logger.info("Остановка ассистента...")

            # Останавливаем сервисы
            self.audio_service.stop_capture()
            self.speech_service.stop()
            self.wake_word_service.stop()

            self._set_state(AssistantState.STOPPED)
            self._emit('on_status_update', "Остановлено", "gray")
            self._log("⏹️ Остановлено")

            logger.info("✓ Ассистент остановлен")

        except Exception as e:
            logger.error(f"Ошибка остановки: {e}", exc_info=True)

    def calibrate_noise_floor(self, duration: float = 2.0):
        """Калибровка фонового шума"""
        if self.state not in [AssistantState.LISTENING_WAKE_WORD]:
            logger.warning("Калибровка доступна только в режиме ожидания")
            return

        try:
            self._log(f"🔧 КАЛИБРОВКА: замолчите на {duration} секунд...")
            self._emit('on_status_update', "Калибровка... Не говорите!", "orange")

            def run_calibration():
                try:
                    self.speech_service.calibrate_noise_floor(duration)
                    self._emit('on_status_update', f"Слушаю... Скажите '{self.wake_word}'", "green")
                except Exception as e:
                    logger.error(f"Ошибка калибровки: {e}", exc_info=True)

            threading.Thread(target=run_calibration, daemon=True).start()

        except Exception as e:
            logger.error(f"Ошибка запуска калибровки: {e}", exc_info=True)

    def update_sensitivity(self, level: int):
        """Обновить чувствительность (1-10) - работает в реалтайм"""
        try:
            self.sensitivity = level
            self.speech_service.update_sensitivity(level)

            # Получаем новый порог для отображения
            new_threshold = self.speech_service.get_voice_threshold()
            self._emit('on_threshold_updated', new_threshold)

            logger.info(f"✓ Чувствительность: {level}/10, новый порог: {new_threshold:.4f}")
        except Exception as e:
            logger.error(f"Ошибка обновления чувствительности: {e}", exc_info=True)

    def reset_context(self):
        """Сброс контекста LLM"""
        try:
            old_count = self.gemini_service.get_history_length()
            self.gemini_service.clear_history()
            new_count = self.gemini_service.get_history_length()

            self._log(f"Новая сессия создана (старая с {old_count} сообщениями сохранена)")
            self._emit('on_history_updated', new_count)

            logger.info(f"✓ Контекст сброшен: {old_count} -> {new_count}")

        except Exception as e:
            logger.error(f"Ошибка сброса контекста: {e}", exc_info=True)

    def get_history_length(self) -> int:
        """Получить длину истории"""
        try:
            return self.gemini_service.get_history_length()
        except Exception as e:
            logger.error(f"Ошибка получения истории: {e}", exc_info=True)
            return 0

    def _on_audio_data(self, audio_data: bytes):
        """Обработка аудио данных"""
        try:
            # Если идет калибровка - данные идут в speech_service
            if self.speech_service.is_calibrating:
                self.speech_service.process_audio(audio_data)
            elif self.state == AssistantState.LISTENING_WAKE_WORD:
                self.wake_word_service.process_audio(audio_data)
            elif self.state == AssistantState.RECORDING_COMMAND:
                self.speech_service.process_audio(audio_data)
        except Exception as e:
            logger.error(f"Ошибка обработки аудио: {e}", exc_info=True)

    def _on_wake_word_detected(self):
        """Wake Word обнаружено"""
        try:
            self._set_state(AssistantState.RECORDING_COMMAND)
            self._emit('on_status_update', "🎤 Ключевое слово обнаружено! Слушаю команду...", "blue")
            self._log("🔑 WAKE WORD обнаружено → переключение на Whisper")
            self._emit('on_wake_word_detected')

            self.speech_service.start_recording()

        except Exception as e:
            logger.error(f"Ошибка обработки wake word: {e}", exc_info=True)
            self._set_state(AssistantState.LISTENING_WAKE_WORD)

    def _on_speech_recognized(self, text: str):
        """Речь распознана"""
        try:
            if text not in ["[тишина]", "[ошибка]"]:
                from datetime import datetime
                timestamp = datetime.now().strftime("%H:%M:%S")
                self._log(f"[{timestamp}] 💬 {text}")
                self._emit('on_speech_recognized', text)

                # Отправка в LLM
                self._send_to_llm(text)

            # Возврат к ожиданию wake word
            self._set_state(AssistantState.LISTENING_WAKE_WORD)
            self._emit('on_status_update', f"Слушаю... Скажите '{self.wake_word}'", "green")
            self.wake_word_service.stop()

        except Exception as e:
            logger.error(f"Ошибка обработки распознанной речи: {e}", exc_info=True)
            self._set_state(AssistantState.LISTENING_WAKE_WORD)

    def _on_audio_level_wake(self, rms: float, status: str):
        """Уровень звука от Wake Word"""
        if self.state == AssistantState.LISTENING_WAKE_WORD:
            self._emit('on_audio_level', rms, status)

    def _on_audio_level_speech(self, rms: float, status: str):
        """Уровень звука от Speech Recognition"""
        if self.state == AssistantState.RECORDING_COMMAND:
            self._emit('on_audio_level', rms, status)

    def _on_noise_floor_calibrated(self, noise_floor: float):
        """Калибровка завершена"""
        try:
            self._log(f"✅ КАЛИБРОВКА: фоновый уровень = {noise_floor:.4f}")
            self._emit('on_noise_floor_calibrated', noise_floor)

            # Также отправляем обновленный порог
            new_threshold = self.speech_service.get_voice_threshold()
            self._emit('on_threshold_updated', new_threshold)
            self._log(f"✅ Новый порог голоса: {new_threshold:.4f}")

        except Exception as e:
            logger.error(f"Ошибка в callback калибровки: {e}", exc_info=True)

    def _send_to_llm(self, query: str):
        """Отправка запроса в LLM"""
        try:
            self._set_state(AssistantState.PROCESSING_LLM)
            self._log(f"📤 Отправка в Gemini: {query}")
            self._emit('on_status_update', "Обработка запроса...", "orange")

            def process():
                try:
                    self.gemini_service.send_query(query)
                except Exception as e:
                    logger.error(f"Ошибка запроса к LLM: {e}", exc_info=True)
                    self._on_llm_error(f"Ошибка запроса: {str(e)}")

            threading.Thread(target=process, daemon=True).start()

        except Exception as e:
            logger.error(f"Ошибка отправки в LLM: {e}", exc_info=True)
            self._set_state(AssistantState.LISTENING_WAKE_WORD)

    def _on_llm_response(self, response: str):
        """Ответ от LLM"""
        try:
            self._log(f"📥 Ответ Gemini: {response}")
            self._emit('on_llm_response', response)

            history_count = self.gemini_service.get_history_length()
            self._emit('on_history_updated', history_count)

            self._set_state(AssistantState.LISTENING_WAKE_WORD)
            self._emit('on_status_update', f"Слушаю... Скажите '{self.wake_word}'", "green")

        except Exception as e:
            logger.error(f"Ошибка обработки ответа LLM: {e}", exc_info=True)

    def _on_llm_error(self, error: str):
        """Ошибка LLM"""
        try:
            self._log(f"❌ Ошибка Gemini: {error}")
            self._emit('on_llm_error', error)

            self._set_state(AssistantState.LISTENING_WAKE_WORD)
            self._emit('on_status_update', "Ошибка обработки запроса", "red")

        except Exception as e:
            logger.error(f"Ошибка обработки ошибки LLM: {e}", exc_info=True)