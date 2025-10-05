"""
Базовый абстрактный класс для TTS
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class BaseTTS(ABC):
    """Абстрактный класс для всех TTS движков"""
    
    def __init__(self):
        self.initialized = False
    
    @abstractmethod
    def initialize(self):
        """Инициализация модели"""
        pass
    
    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """
        Синтез речи из текста
        
        Args:
            text: Текст для озвучки
            
        Returns:
            bytes: Аудио данные в формате WAV
        """
        pass
    
    @abstractmethod
    def get_available_voices(self) -> list[str]:
        """Получить список доступных голосов"""
        pass
    
    @abstractmethod
    def set_voice(self, voice_name: str):
        """Установить голос"""
        pass
    
    def cleanup(self):
        """Очистка ресурсов"""
        pass
