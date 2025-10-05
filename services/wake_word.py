"""
Сервис детекции ключевого слова (Wake Word) с Vosk
С улучшенной обработкой ошибок
"""

import os
import json
import logging
import numpy as np
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional
from vosk import Model, KaldiRecognizer
from utils.config_loader import config

logger = logging.getLogger(__name__)

class WakeWordService:
    """Детекция ключевого слова"""

    def __init__(self):
        self.model = None
        self.recognizer = None
        self.wake_word = config.get('wake_word.keyword', 'привет ассистент').lower()
        self.model_path = config.get('wake_word.model_path')
        self.sample_rate = config.get('audio.sample_rate', 16000)

        self.is_recording = False
        self.wake_word_callback = None
        self.audio_level_callback = None

        logger.debug("WakeWordService инициализирован")

    def initialize(self, wake_word: Optional[str] = None):
        """
        Инициализация модели Vosk (с автозагрузкой)

        Args:
            wake_word: Ключевое слово для детекции
        """
        try:
            if wake_word:
                self.wake_word = wake_word.lower()

            # Если уже инициализирована - просто сбрасываем
            if self.model and self.recognizer:
                logger.info("Vosk уже загружен, переиспользуем")
                try:
                    self.recognizer.Reset()
                except Exception as e:
                    logger.warning(f"Ошибка сброса recognizer: {e}")
                self.is_recording = False
                return

            logger.info(f"Инициализация Vosk для '{self.wake_word}'...")

            # Автозагрузка модели если не найдена
            if not os.path.exists(self.model_path):
                logger.warning(f"Модель Vosk не найдена, загружаю...")
                self._download_model()

            # Загрузка модели
            self.model = Model(self.model_path)
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)

            try:
                self.recognizer.SetMaxAlternatives(0)
                self.recognizer.SetWords(False)
            except Exception as e:
                logger.warning(f"Не удалось установить параметры recognizer: {e}")

            logger.info(f"✓ Vosk готов. Ключевое слово: '{self.wake_word}'")

        except Exception as e:
            logger.error(f"Ошибка инициализации Vosk: {e}", exc_info=True)
            raise

    def _download_model(self):
        """Автозагрузка модели Vosk"""
        model_url = config.get('wake_word.model_url')
        model_name = os.path.basename(self.model_path)
        zip_path = f"{self.model_path}.zip"

        try:
            # Создание папки
            os.makedirs("models", exist_ok=True)

            # Загрузка
            logger.info(f"Загрузка {model_name} (~50MB)...")
            logger.info("Это займет 1-3 минуты в зависимости от скорости интернета...")

            def progress_hook(block_num, block_size, total_size):
                try:
                    downloaded = block_num * block_size
                    if total_size > 0:
                        percent = min(downloaded * 100 / total_size, 100)
                        if block_num % 100 == 0:  # Логируем каждые 100 блоков
                            logger.info(f"Загрузка: {percent:.1f}%")
                except Exception as e:
                    logger.warning(f"Ошибка progress_hook: {e}")

            urllib.request.urlretrieve(model_url, zip_path, progress_hook)
            logger.info("Загрузка завершена, распаковка...")

            # Распаковка
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("models")

            # Удаление архива
            try:
                os.remove(zip_path)
            except Exception as e:
                logger.warning(f"Не удалось удалить архив: {e}")

            logger.info(f"✅ Модель Vosk установлена: {self.model_path}")

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}", exc_info=True)
            logger.info(f"Скачайте модель вручную: {model_url}")
            raise

    def process_audio(self, audio_data: bytes):
        """
        Обработка аудио данных

        Args:
            audio_data: Аудио в формате bytes (int16)
        """
        try:
            if not self.recognizer or self.is_recording:
                return

            # Вычисление RMS для индикатора
            try:
                audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2)) / 32768.0
            except Exception as e:
                logger.warning(f"Ошибка вычисления RMS: {e}")
                rms = 0.0

            # Отправка уровня звука
            if self.audio_level_callback:
                try:
                    status = self._get_audio_status(rms)
                    self.audio_level_callback(rms, status)
                except Exception as e:
                    logger.error(f"Ошибка в audio_level_callback: {e}")

            # Распознавание
            try:
                if self.recognizer.AcceptWaveform(audio_data):
                    result = self.recognizer.Result()
                    self._check_wake_word(result, is_final=True)
                else:
                    partial = self.recognizer.PartialResult()
                    self._check_wake_word(partial, is_final=False)
            except Exception as e:
                logger.error(f"Ошибка распознавания Vosk: {e}")

        except Exception as e:
            logger.error(f"Критическая ошибка в process_audio: {e}", exc_info=True)

    def _check_wake_word(self, json_result: str, is_final: bool):
        """Проверка наличия ключевого слова в результате"""
        try:
            result = json.loads(json_result)
            text = result.get('text') or result.get('partial', '')

            if text and self.wake_word in text.lower():
                logger.info(f"✅ Wake Word обнаружено: '{text}'")
                self.is_recording = True

                if self.wake_word_callback:
                    try:
                        self.wake_word_callback()
                    except Exception as e:
                        logger.error(f"Ошибка в wake_word_callback: {e}")

                # Очистка буфера
                try:
                    self.recognizer.Reset()
                except Exception as e:
                    logger.warning(f"Ошибка сброса recognizer: {e}")

        except json.JSONDecodeError as e:
            logger.debug(f"JSON decode error (это норма): {e}")
        except Exception as e:
            logger.error(f"Ошибка проверки wake word: {e}")

    def _get_audio_status(self, rms: float) -> str:
        """Определение статуса аудио по RMS"""
        try:
            if rms < 0.02:
                return "Тишина"
            elif rms < 0.05:
                return "Шум"
            else:
                return "Голос"
        except Exception as e:
            logger.error(f"Ошибка определения статуса: {e}")
            return "Неизвестно"

    def stop(self):
        """Остановка записи (возврат к детекции wake word)"""
        try:
            self.is_recording = False
            if self.recognizer:
                try:
                    self.recognizer.Reset()
                except Exception as e:
                    logger.warning(f"Ошибка сброса при остановке: {e}")
            logger.info("✓ Возврат к Wake Word Detection")
        except Exception as e:
            logger.error(f"Ошибка остановки: {e}")

    def set_wake_word_callback(self, callback):
        """Установить callback для обнаружения wake word"""
        self.wake_word_callback = callback

    def set_audio_level_callback(self, callback):
        """Установить callback для уровня звука"""
        self.audio_level_callback = callback

    def __del__(self):
        """Очистка ресурсов"""
        try:
            self.model = None
            self.recognizer = None
        except Exception as e:
            logger.error(f"Ошибка в __del__: {e}")