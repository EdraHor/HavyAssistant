"""
Сервис детекции ключевого слова (Wake Word) с Vosk
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

    def initialize(self, wake_word: Optional[str] = None):
        """
        Инициализация модели Vosk (с автозагрузкой)

        Args:
            wake_word: Ключевое слово для детекции
        """
        if wake_word:
            self.wake_word = wake_word.lower()

        # Если уже инициализирована - просто сбрасываем
        if self.model and self.recognizer:
            logger.info("Vosk уже загружен, переиспользуем")
            self.recognizer.Reset()
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
        self.recognizer.SetMaxAlternatives(0)
        self.recognizer.SetWords(False)

        logger.info(f"Vosk готов. Ключевое слово: '{self.wake_word}'")

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
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(downloaded * 100 / total_size, 100)
                    if block_num % 100 == 0:  # Логируем каждые 100 блоков
                        logger.info(f"Загрузка: {percent:.1f}%")

            urllib.request.urlretrieve(model_url, zip_path, progress_hook)
            logger.info("Загрузка завершена, распаковка...")

            # Распаковка
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("models")

            # Удаление архива
            os.remove(zip_path)

            logger.info(f"✅ Модель Vosk установлена: {self.model_path}")

        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            logger.info(f"Скачайте модель вручную: {model_url}")
            raise

    def process_audio(self, audio_data: bytes):
        """
        Обработка аудио данных

        Args:
            audio_data: Аудио в формате bytes (int16)
        """
        if not self.recognizer or self.is_recording:
            return

        # Вычисление RMS для индикатора
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_int16.astype(np.float32) ** 2)) / 32768.0

        # Отправка уровня звука
        if self.audio_level_callback:
            status = self._get_audio_status(rms)
            self.audio_level_callback(rms, status)

        # Распознавание
        if self.recognizer.AcceptWaveform(audio_data):
            result = self.recognizer.Result()
            self._check_wake_word(result, is_final=True)
        else:
            partial = self.recognizer.PartialResult()
            self._check_wake_word(partial, is_final=False)

    def _check_wake_word(self, json_result: str, is_final: bool):
        """Проверка наличия ключевого слова в результате"""
        try:
            result = json.loads(json_result)
            text = result.get('text') or result.get('partial', '')

            if text and self.wake_word in text.lower():
                logger.info(f"✅ Wake Word обнаружено: '{text}'")
                self.is_recording = True

                if self.wake_word_callback:
                    self.wake_word_callback()

                # Очистка буфера
                self.recognizer.Reset()

        except json.JSONDecodeError:
            pass

    def _get_audio_status(self, rms: float) -> str:
        """Определение статуса аудио по RMS"""
        if rms < 0.02:
            return "Тишина"
        elif rms < 0.05:
            return "Шум"
        else:
            return "Голос"

    def stop(self):
        """Остановка записи (возврат к детекции wake word)"""
        self.is_recording = False
        if self.recognizer:
            self.recognizer.Reset()
        logger.info("Возврат к Wake Word Detection")

    def set_wake_word_callback(self, callback):
        """Установить callback для обнаружения wake word"""
        self.wake_word_callback = callback

    def set_audio_level_callback(self, callback):
        """Установить callback для уровня звука"""
        self.audio_level_callback = callback

    def __del__(self):
        # Очистка ресурсов
        self.model = None
        self.recognizer = None