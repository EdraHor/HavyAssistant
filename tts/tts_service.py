"""
Сервис синтеза речи с поддержкой разных движков
"""

import logging
import sounddevice as sd
import numpy as np
import io
import soundfile as sf
from typing import Optional
from utils.config_loader import config

logger = logging.getLogger(__name__)

class TTSService:
    """Унифицированный сервис TTS"""
    
    def __init__(self):
        self.tts_engine = None
        self.enabled = config.get('tts.enabled', True)
        self.engine_type = config.get('tts.engine', 'silero')
        self.voice = config.get('tts.voice', 'baya')
        self.device = config.get('tts.device', 'cuda')
        
        # Callbacks
        self.on_start_callback = None
        self.on_finish_callback = None
        
    def initialize(self):
        """Инициализация TTS движка"""
        try:
            if not self.enabled:
                logger.info("TTS отключен в конфигурации")
                return
            
            logger.info(f"Инициализация TTS: {self.engine_type}")
            
            # Выбор движка
            if self.engine_type == 'silero':
                from tts.silero_tts import SileroTTS
                self.tts_engine = SileroTTS(device=self.device, voice=self.voice)
            elif self.engine_type == 'piper':
                from tts.piper_tts import PiperTTS
                self.tts_engine = PiperTTS(voice=self.voice)
            else:
                raise ValueError(f"Неизвестный TTS движок: {self.engine_type}")
            
            # Инициализация
            self.tts_engine.initialize()
            logger.info(f"✅ TTS готов ({self.engine_type})")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации TTS: {e}", exc_info=True)
            self.enabled = False
            raise
    
    def speak(self, text: str):
        """
        Озвучить текст
        
        Args:
            text: Текст для озвучки
        """
        try:
            if not self.enabled or not self.tts_engine:
                logger.warning("TTS недоступен")
                return
            
            # Callback начала
            if self.on_start_callback:
                try:
                    self.on_start_callback(text)
                except Exception as e:
                    logger.error(f"Ошибка в on_start_callback: {e}")
            
            logger.info(f"🔊 Озвучивание: '{text[:50]}...'")
            
            # Синтез
            audio_bytes = self.tts_engine.synthesize(text)
            
            # Воспроизведение
            self._play_audio(audio_bytes)
            
            # Callback завершения
            if self.on_finish_callback:
                try:
                    self.on_finish_callback()
                except Exception as e:
                    logger.error(f"Ошибка в on_finish_callback: {e}")
            
            logger.info("✅ Озвучивание завершено")
            
        except Exception as e:
            logger.error(f"Ошибка озвучивания: {e}", exc_info=True)
    
    def _play_audio(self, audio_bytes: bytes):
        """Воспроизвести аудио"""
        try:
            # Чтение WAV из bytes
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Воспроизведение
            sd.play(audio_data, sample_rate)
            sd.wait()  # Ждем завершения
            
        except Exception as e:
            logger.error(f"Ошибка воспроизведения: {e}", exc_info=True)
            raise
    
    def get_available_voices(self) -> list[str]:
        """Получить доступные голоса"""
        if self.tts_engine:
            return self.tts_engine.get_available_voices()
        return []
    
    def set_voice(self, voice_name: str):
        """Изменить голос"""
        try:
            if self.tts_engine:
                self.tts_engine.set_voice(voice_name)
                self.voice = voice_name
                logger.info(f"Голос изменен: {voice_name}")
        except Exception as e:
            logger.error(f"Ошибка смены голоса: {e}")
    
    def set_callbacks(self, on_start=None, on_finish=None):
        """Установить callbacks"""
        self.on_start_callback = on_start
        self.on_finish_callback = on_finish
    
    def cleanup(self):
        """Очистка"""
        try:
            if self.tts_engine:
                self.tts_engine.cleanup()
                self.tts_engine = None
            logger.info("✓ TTS очищен")
        except Exception as e:
            logger.error(f"Ошибка очистки TTS: {e}")
