"""
Сервис распознавания речи с Faster-Whisper (GPU)
С улучшенной обработкой ошибок
"""

import logging
import numpy as np
import torch
from faster_whisper import WhisperModel
from utils.config_loader import config
import threading

logger = logging.getLogger(__name__)

class SpeechRecognitionService:
    """Распознавание речи с Faster-Whisper"""

    def __init__(self):
        self.model = None
        self.audio_buffer = []
        self.is_recording = False

        self.sample_rate = config.get('audio.sample_rate', 16000)
        self.sensitivity = config.get('speech_recognition.sensitivity', 5)
        self.noise_floor = 0.01
        self.sensitivity_multiplier = 2.0

        self.silent_frames_count = 0
        self.sound_frames_count = 0
        self.calibration_samples = []
        self.is_calibrating = False

        # Пороги
        self.SILENT_FRAMES_TO_STOP = config.get('speech_recognition.silent_frames_to_stop', 15)
        self.MIN_SOUND_FRAMES = config.get('speech_recognition.min_sound_frames', 3)
        self.CALIBRATION_FRAMES = config.get('speech_recognition.calibration_frames', 30)

        # Callbacks
        self.speech_recognized_callback = None
        self.audio_level_callback = None
        self.noise_floor_callback = None

        # Защитный таймер
        self.recording_timer = None

        logger.debug("SpeechRecognitionService инициализирован")

    def initialize(self):
        """Инициализация модели Faster-Whisper"""
        try:
            # Если уже загружена - переиспользуем
            if self.model:
                logger.info("Whisper уже загружен, переиспользуем")
                self.stop()  # Сбрасываем состояние
                return

            model_name = config.get('speech_recognition.model_name', 'large-v3')
            device = config.get('speech_recognition.device', 'cuda')
            compute_type = config.get('speech_recognition.compute_type', 'float16')

            # Проверка доступности GPU
            if device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA недоступна, переключаюсь на CPU")
                device = 'cpu'
                compute_type = 'int8'

            # Размеры моделей для информации
            model_sizes = {
                'tiny': '75MB',
                'base': '145MB',
                'small': '466MB',
                'medium': '1.5GB',
                'large-v3': '3GB',
                'large-v3-turbo': '1.6GB'
            }

            size = model_sizes.get(model_name, '~1GB')
            logger.info(f"Загрузка Faster-Whisper: {model_name} ({size}) на {device} ({compute_type})")
            logger.info(f"⏳ Первая загрузка может занять 1-10 минут...")
            logger.info(f"💡 Модель кэшируется, следующий запуск будет мгновенным")

            self.model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
                download_root="models",
                num_workers=4
            )

            logger.info("✅ Faster-Whisper готов!")

            # Warm-up (первый пустой прогон для инициализации CUDA)
            if device == 'cuda':
                try:
                    logger.info("Прогрев GPU...")
                    dummy_audio = np.zeros(16000, dtype=np.float32)
                    list(self.model.transcribe(dummy_audio, language='ru'))
                    logger.info("✅ GPU готов!")
                except Exception as e:
                    logger.warning(f"Ошибка прогрева GPU (не критично): {e}")

        except Exception as e:
            logger.error(f"Ошибка инициализации Whisper: {e}", exc_info=True)
            raise

    def update_sensitivity(self, level: int):
        """
        Обновить чувствительность (1-10)

        Args:
            level: 1 = менее чувствительно, 10 = более чувствительно
        """
        try:
            self.sensitivity = level
            # Инвертируем: чем выше level, тем ниже множитель порога
            self.sensitivity_multiplier = max((11 - level) * 0.3, 0.5)
            logger.info(f"✓ Чувствительность: {level}/10 (множитель {self.sensitivity_multiplier:.1f})")
        except Exception as e:
            logger.error(f"Ошибка обновления чувствительности: {e}")

    def calibrate_noise_floor(self, duration: float = 2.0):
        """
        Калибровка фонового шума - ждет накопления сэмплов

        Args:
            duration: Длительность калибровки в секундах
        """
        try:
            import time
            logger.info(f"🔧 Калибровка фона ({duration}с)...")
            self.is_calibrating = True
            self.calibration_samples.clear()

            # Вычисляем сколько сэмплов должно быть за duration
            # При chunk_size=1024 и sample_rate=16000 -> ~15.6 chunks в секунду
            expected_samples = int(duration * (self.sample_rate / 1024))
            max_wait_time = duration * 2  # Максимум ждем в 2 раза дольше

            start_time = time.time()
            logger.info(f"Ожидаем {expected_samples} сэмплов...")

            # Ждем пока не наберется достаточно сэмплов или не истечет время
            while len(self.calibration_samples) < expected_samples:
                if time.time() - start_time > max_wait_time:
                    logger.warning(f"Таймаут калибровки, набрано только {len(self.calibration_samples)} сэмплов")
                    break
                time.sleep(0.05)  # Маленькая пауза

            if self.calibration_samples:
                self.noise_floor = np.mean(self.calibration_samples)
                new_threshold = self.get_voice_threshold()

                logger.info(f"✅ Калибровка завершена:")
                logger.info(f"  - Сэмплов: {len(self.calibration_samples)}")
                logger.info(f"  - Фон: {self.noise_floor:.4f}")
                logger.info(f"  - Порог: {new_threshold:.4f}")

                if self.noise_floor_callback:
                    try:
                        self.noise_floor_callback(self.noise_floor)
                    except Exception as e:
                        logger.error(f"Ошибка в noise_floor_callback: {e}")
            else:
                logger.error("Калибровка не удалась - не набрано сэмплов!")
                logger.error("Проверьте что аудио захват работает")

            self.is_calibrating = False
            self.calibration_samples.clear()

        except Exception as e:
            logger.error(f"Ошибка калибровки: {e}", exc_info=True)
            self.is_calibrating = False
            self.calibration_samples.clear()

    def get_voice_threshold(self) -> float:
        """Вычислить порог голоса на основе фона и чувствительности"""
        try:
            return self.noise_floor * self.sensitivity_multiplier
        except Exception as e:
            logger.error(f"Ошибка вычисления порога: {e}")
            return 0.02  # Дефолтное значение

    def start_recording(self):
        """Начать запись команды"""
        try:
            self.is_recording = True
            self.audio_buffer.clear()
            self.silent_frames_count = 0
            self.sound_frames_count = 0

            logger.info(f"🎙️ Запись команды (порог: {self.get_voice_threshold():.4f})")

            # Защитный таймер на 60 секунд
            if self.recording_timer:
                try:
                    self.recording_timer.cancel()
                except Exception as e:
                    logger.warning(f"Ошибка отмены таймера: {e}")

            self.recording_timer = threading.Timer(60.0, self._timeout_handler)
            self.recording_timer.start()

        except Exception as e:
            logger.error(f"Ошибка start_recording: {e}", exc_info=True)

    def _timeout_handler(self):
        """Обработчик таймаута записи"""
        try:
            if self.is_recording:
                logger.warning("⏱️ Таймаут записи (60с)")
                self._recognize_speech()
        except Exception as e:
            logger.error(f"Ошибка в timeout_handler: {e}")

    def process_audio(self, audio_data: bytes):
        """Обработка аудио данных"""
        try:
            # Вычисление RMS
            try:
                audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2)) / 32768.0
            except Exception as e:
                logger.warning(f"Ошибка вычисления RMS: {e}")
                return

            # Калибровка
            if self.is_calibrating:
                self.calibration_samples.append(rms)
                count = len(self.calibration_samples)

                if self.audio_level_callback:
                    try:
                        self.audio_level_callback(rms, f"Калибровка {count}")
                    except Exception as e:
                        logger.error(f"Ошибка в audio_level_callback: {e}")

                # Логируем каждые 5 сэмплов
                if count % 5 == 0:
                    logger.debug(f"Калибровка: набрано {count} сэмплов, RMS={rms:.4f}")

                return

            if not self.is_recording:
                return

            # Определение голоса
            threshold = self.get_voice_threshold()
            is_voice = rms > threshold

            status = "Тишина" if rms < self.noise_floor else "Шум" if rms < threshold else "ГОЛОС"

            # Отправка уровня звука
            if self.audio_level_callback:
                try:
                    self.audio_level_callback(rms, status)
                except Exception as e:
                    logger.error(f"Ошибка в audio_level_callback: {e}")

            logger.debug(
                f"RMS: {rms:.4f} | Порог: {threshold:.4f} | Фон: {self.noise_floor:.4f} | Статус: {status}")

            # Логика записи
            if is_voice:
                self.sound_frames_count += 1
                self.silent_frames_count = 0
                self.audio_buffer.extend(audio_int16)

                # Перезапуск таймера при голосе
                if self.recording_timer:
                    try:
                        self.recording_timer.cancel()
                        self.recording_timer = threading.Timer(60.0, self._timeout_handler)
                        self.recording_timer.start()
                    except Exception as e:
                        logger.warning(f"Ошибка перезапуска таймера: {e}")

                logger.debug(f"📝 Записываю: буфер {len(self.audio_buffer) // 1000}K сэмплов")
            else:
                if self.sound_frames_count >= self.MIN_SOUND_FRAMES:
                    self.silent_frames_count += 1
                    self.audio_buffer.extend(audio_int16)

                    logger.debug(f"⏸️ Пауза ({self.silent_frames_count}/{self.SILENT_FRAMES_TO_STOP})")

                    if self.silent_frames_count >= self.SILENT_FRAMES_TO_STOP:
                        logger.debug("✅ Достаточно тишины → распознаю")
                        self._recognize_speech()
                else:
                    logger.debug(f"⏳ Жду голос... ({self.sound_frames_count}/{self.MIN_SOUND_FRAMES})")

        except Exception as e:
            logger.error(f"Критическая ошибка в process_audio: {e}", exc_info=True)

    def _recognize_speech(self):
        """Распознавание записанной речи"""
        try:
            self.is_recording = False

            if self.recording_timer:
                try:
                    self.recording_timer.cancel()
                except Exception as e:
                    logger.warning(f"Ошибка отмены таймера: {e}")

            if len(self.audio_buffer) < self.sample_rate:  # Меньше 1 секунды
                logger.warning(f"Мало данных ({len(self.audio_buffer)} сэмплов)")
                if self.speech_recognized_callback:
                    try:
                        self.speech_recognized_callback("[тишина]")
                    except Exception as e:
                        logger.error(f"Ошибка в callback: {e}")
                return

            # Конвертация в float32
            try:
                audio_float = np.array(self.audio_buffer, dtype=np.float32) / 32768.0
            except Exception as e:
                logger.error(f"Ошибка конвертации аудио: {e}")
                return
            finally:
                self.audio_buffer.clear()

            logger.info(f"Обработка {len(audio_float)//1000}K сэмплов...")

            # Запуск в отдельном потоке
            threading.Thread(target=self._transcribe, args=(audio_float,), daemon=True).start()

        except Exception as e:
            logger.error(f"Ошибка в _recognize_speech: {e}", exc_info=True)

    def _transcribe(self, audio: np.ndarray):
        """Транскрипция аудио"""
        try:
            beam_size = config.get('speech_recognition.beam_size', 1)
            language = config.get('speech_recognition.language', 'ru')
            vad_filter = config.get('speech_recognition.vad_filter', True)

            segments, info = self.model.transcribe(
                audio,
                language=language,
                beam_size=beam_size,
                vad_filter=vad_filter,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Сбор текста
            text_parts = []
            for segment in segments:
                try:
                    text_parts.append(segment.text.strip())
                    logger.debug(f"Сегмент: {segment.text.strip()}")
                except Exception as e:
                    logger.warning(f"Ошибка обработки сегмента: {e}")

            result_text = " ".join(text_parts).strip()

            if result_text:
                logger.info(f"✅ Результат: {result_text}")
                if self.speech_recognized_callback:
                    try:
                        self.speech_recognized_callback(result_text)
                    except Exception as e:
                        logger.error(f"Ошибка в callback: {e}")
            else:
                logger.info("Пустой результат")
                if self.speech_recognized_callback:
                    try:
                        self.speech_recognized_callback("[тишина]")
                    except Exception as e:
                        logger.error(f"Ошибка в callback: {e}")

        except Exception as e:
            logger.error(f"Ошибка распознавания: {e}", exc_info=True)
            if self.speech_recognized_callback:
                try:
                    self.speech_recognized_callback("[ошибка]")
                except Exception as e:
                    logger.error(f"Ошибка в callback: {e}")

    def stop(self):
        """Остановка записи и очистка"""
        try:
            self.is_recording = False
            self.is_calibrating = False
            self.audio_buffer.clear()
            self.calibration_samples.clear()
            self.silent_frames_count = 0
            self.sound_frames_count = 0

            # Отмена таймеров
            if self.recording_timer:
                try:
                    self.recording_timer.cancel()
                    self.recording_timer = None
                except Exception as e:
                    logger.warning(f"Ошибка отмены таймера: {e}")

            logger.info("✓ Speech recognition остановлен")

        except Exception as e:
            logger.error(f"Ошибка остановки: {e}")

    def set_speech_recognized_callback(self, callback):
        """Установить callback для распознанной речи"""
        self.speech_recognized_callback = callback

    def set_audio_level_callback(self, callback):
        """Установить callback для уровня звука"""
        self.audio_level_callback = callback

    def set_noise_floor_callback(self, callback):
        """Установить callback для калибровки"""
        self.noise_floor_callback = callback