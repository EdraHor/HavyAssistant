"""
Контроллер голосового ассистента с поддержкой TTS
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
from tts import TTSService  # НОВОЕ

logger = logging.getLogger(__name__)

class AssistantState(Enum):
    """Состояния ассистента"""
    STOPPED = "stopped"
    INITIALIZING = "initializing"
    LISTENING_WAKE_WORD = "listening_wake_word"
    RECORDING_COMMAND = "recording_command"
    PROCESSING_LLM = "processing_llm"
    SPEAKING = "speaking"  # НОВОЕ
    ERROR = "error"

class VoiceAssistantController:
    """Основной контроллер с TTS"""

    def __init__(self):
        # Сервисы
        self.audio_service = AudioCaptureService()
        self.wake_word_service = WakeWordService()
        self.speech_service = SpeechRecognitionService()
        self.gemini_service = GeminiService()
        self.tts_service = TTSService()  # НОВОЕ

        # Состояние
        self.state = AssistantState.STOPPED
        self.selected_device_index = None
        self.wake_word = config.get('wake_word.keyword', 'привет ассистент')
        self.sensitivity = config.get('speech_recognition.sensitivity', 5)

        # Callbacks
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
            'on_tts_start': None,  # НОВОЕ
            'on_tts_finish': None,  # НОВОЕ
        }

        # Настройка callbacks
        self._setup_service_callbacks()

        logger.info("✓ VoiceAssistantController инициализирован")

    def _setup_service_callbacks(self):
        """Настройка callbacks"""
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

        # TTS - НОВОЕ
        self.tts_service.set_callbacks(
            on_start=self._on_tts_start,
            on_finish=self._on_tts_finish
        )

    def set_callback(self, event: str, callback: Callable):
        """Установить callback"""
        if event in self.callbacks:
            self.callbacks[event] = callback
            logger.debug(f"Установлен callback для '{event}'")
        else:
            logger.warning(f"Неизвестный callback: '{event}'")

    def _emit(self, event: str, *args, **kwargs):
        """Вызвать callback"""
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
        """Получить аудио устройства"""
        try:
            return self.audio_service.get_audio_devices()
        except Exception as e:
            logger.error(f"Ошибка получения устройств: {e}", exc_info=True)
            return []

    def start(self, device_index: int, wake_word: Optional[str] = None):
        """Запустить ассистент"""
        if self.state != AssistantState.STOPPED:
            logger.warning("Ассистент уже запущен")
            return False

        try:
            self._set_state(AssistantState.INITIALIZING)
            self.selected_device_index = device_index

            if wake_word:
                self.wake_word = wake_word

            # Инициализация в потоке
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
        """Инициализация всех сервисов"""
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

            # TTS - НОВОЕ
            if config.get('tts.enabled', True):
                tts_engine = config.get('tts.engine', 'silero')
                self._emit('on_status_update', f"Загрузка TTS ({tts_engine})...", "orange")
                self.tts_service.initialize()
                logger.info("✓ TTS инициализирован")

            # Аудио захват
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

            self.audio_service.stop_capture()
            self.speech_service.stop()
            self.wake_word_service.stop()
            self.tts_service.cleanup()  # НОВОЕ

            self._set_state(AssistantState.STOPPED)
            self._emit('on_status_update', "Остановлено", "gray")
            self._log("⏹️ Остановлено")

            logger.info("✓ Ассистент остановлен")

        except Exception as e:
            logger.error(f"Ошибка остановки: {e}", exc_info=True)

    def calibrate_noise_floor(self, duration: float = 2.0):
        """Калибровка"""
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
        """Обновить чувствительность"""
        try:
            self.sensitivity = level
            self.speech_service.update_sensitivity(level)

            new_threshold = self.speech_service.get_voice_threshold()
            self._emit('on_threshold_updated', new_threshold)

            logger.info(f"✓ Чувствительность: {level}/10, порог: {new_threshold:.4f}")
        except Exception as e:
            logger.error(f"Ошибка обновления чувствительности: {e}", exc_info=True)

    def reset_context(self):
        """Сброс контекста"""
        try:
            old_count = self.gemini_service.get_history_length()
            self.gemini_service.clear_history()
            new_count = self.gemini_service.get_history_length()

            self._log(f"Новая сессия (старая с {old_count} сообщениями сохранена)")
            self._emit('on_history_updated', new_count)

            logger.info(f"✓ Контекст сброшен: {old_count} -> {new_count}")

        except Exception as e:
            logger.error(f"Ошибка сброса контекста: {e}", exc_info=True)

    def get_history_length(self) -> int:
        """Длина истории"""
        try:
            return self.gemini_service.get_history_length()
        except Exception as e:
            logger.error(f"Ошибка получения истории: {e}", exc_info=True)
            return 0

    def _on_audio_data(self, audio_data: bytes):
        """Обработка аудио"""
        try:
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
            self._emit('on_status_update', "🎤 Ключевое слово! Слушаю...", "blue")
            self._log("🔑 WAKE WORD → Whisper")
            self._emit('on_wake_word_detected')

            self.speech_service.start_recording()

        except Exception as e:
            logger.error(f"Ошибка wake word: {e}", exc_info=True)
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

            # Возврат к wake word
            self._set_state(AssistantState.LISTENING_WAKE_WORD)
            self._emit('on_status_update', f"Слушаю... '{self.wake_word}'", "green")
            self.wake_word_service.stop()

        except Exception as e:
            logger.error(f"Ошибка распознавания: {e}", exc_info=True)
            self._set_state(AssistantState.LISTENING_WAKE_WORD)

    def _on_audio_level_wake(self, rms: float, status: str):
        """Уровень от Wake Word"""
        if self.state == AssistantState.LISTENING_WAKE_WORD:
            self._emit('on_audio_level', rms, status)

    def _on_audio_level_speech(self, rms: float, status: str):
        """Уровень от Speech"""
        if self.state == AssistantState.RECORDING_COMMAND:
            self._emit('on_audio_level', rms, status)

    def _on_noise_floor_calibrated(self, noise_floor: float):
        """Калибровка завершена"""
        try:
            self._log(f"✅ КАЛИБРОВКА: {noise_floor:.4f}")
            self._emit('on_noise_floor_calibrated', noise_floor)

            new_threshold = self.speech_service.get_voice_threshold()
            self._emit('on_threshold_updated', new_threshold)
            self._log(f"✅ Порог: {new_threshold:.4f}")

        except Exception as e:
            logger.error(f"Ошибка callback калибровки: {e}", exc_info=True)

    def _send_to_llm(self, query: str):
        """Отправка в LLM"""
        try:
            self._set_state(AssistantState.PROCESSING_LLM)
            self._log(f"📤 Gemini: {query}")
            self._emit('on_status_update', "Обработка...", "orange")

            def process():
                try:
                    self.gemini_service.send_query(query)
                except Exception as e:
                    logger.error(f"Ошибка LLM: {e}", exc_info=True)
                    self._on_llm_error(f"Ошибка: {str(e)}")

            threading.Thread(target=process, daemon=True).start()

        except Exception as e:
            logger.error(f"Ошибка отправки LLM: {e}", exc_info=True)
            self._set_state(AssistantState.LISTENING_WAKE_WORD)

    def _on_llm_response(self, response: str):
        """Ответ от LLM - ОБНОВЛЕНО"""
        try:
            self._log(f"📥 Gemini: {response}")
            self._emit('on_llm_response', response)

            history_count = self.gemini_service.get_history_length()
            self._emit('on_history_updated', history_count)

            # НОВОЕ: Озвучивание ответа
            if config.get('tts.enabled', True):
                self._set_state(AssistantState.SPEAKING)
                self._emit('on_status_update', "🔊 Озвучивание ответа...", "blue")

                # Озвучка в отдельном потоке
                def speak():
                    try:
                        self.tts_service.speak(response)
                        # После озвучки возврат к wake word
                        self._set_state(AssistantState.LISTENING_WAKE_WORD)
                        self._emit('on_status_update', f"Слушаю... '{self.wake_word}'", "green")
                    except Exception as e:
                        logger.error(f"Ошибка озвучки: {e}", exc_info=True)
                        self._set_state(AssistantState.LISTENING_WAKE_WORD)

                threading.Thread(target=speak, daemon=True).start()
            else:
                # Если TTS выключен
                self._set_state(AssistantState.LISTENING_WAKE_WORD)
                self._emit('on_status_update', f"Слушаю... '{self.wake_word}'", "green")

        except Exception as e:
            logger.error(f"Ошибка обработки ответа: {e}", exc_info=True)

    def _on_llm_error(self, error: str):
        """Ошибка LLM"""
        try:
            self._log(f"❌ Gemini: {error}")
            self._emit('on_llm_error', error)

            self._set_state(AssistantState.LISTENING_WAKE_WORD)
            self._emit('on_status_update', "Ошибка", "red")

        except Exception as e:
            logger.error(f"Ошибка обработки ошибки: {e}", exc_info=True)

    # НОВЫЕ callbacks для TTS
    def _on_tts_start(self, text: str):
        """TTS начал озвучивание"""
        self._log(f"🔊 Озвучивание: {text[:50]}...")
        self._emit('on_tts_start', text)

    def _on_tts_finish(self):
        """TTS завершил озвучивание"""
        self._log("✅ Озвучивание завершено")
        self._emit('on_tts_finish')