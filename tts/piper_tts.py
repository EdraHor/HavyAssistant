"""
Piper TTS - быстрый легкий TTS
"""

import logging
import subprocess
import urllib.request
import json
from pathlib import Path
from .base_tts import BaseTTS

logger = logging.getLogger(__name__)

class PiperTTS(BaseTTS):
    """Piper TTS - легкий и быстрый"""
    
    # Русские модели Piper
    MODELS = {
        'ru_iryna': {
            'url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/iryna/medium/ru_RU-iryna-medium.onnx',
            'config_url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/iryna/medium/ru_RU-iryna-medium.onnx.json',
            'description': 'Женский, средний'
        },
        'ru_dmitri': {
            'url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx',
            'config_url': 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx.json',
            'description': 'Мужской, средний'
        }
    }
    
    def __init__(self, voice: str = 'ru_iryna'):
        super().__init__()
        self.voice = voice
        self.model_path = None
        self.config_path = None
        self.piper_bin = 'piper'  # Установить: pip install piper-tts
        
    def initialize(self):
        """Инициализация Piper"""
        try:
            if self.initialized:
                return
            
            logger.info("Инициализация Piper TTS...")
            
            # Загрузка модели
            self._download_model(self.voice)
            
            logger.info(f"✅ Piper TTS готов (голос: {self.voice})")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации Piper: {e}", exc_info=True)
            raise
    
    def _download_model(self, voice: str):
        """Скачать модель"""
        if voice not in self.MODELS:
            raise ValueError(f"Голос {voice} не найден")
        
        model_dir = Path('models/piper')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_info = self.MODELS[voice]
        self.model_path = model_dir / f"{voice}.onnx"
        self.config_path = model_dir / f"{voice}.onnx.json"
        
        # Скачивание если нет
        if not self.model_path.exists():
            logger.info(f"Загрузка модели {voice}...")
            urllib.request.urlretrieve(model_info['url'], self.model_path)
            urllib.request.urlretrieve(model_info['config_url'], self.config_path)
            logger.info("✅ Модель загружена")
    
    def synthesize(self, text: str) -> bytes:
        """Синтез речи через CLI"""
        try:
            if not self.initialized:
                self.initialize()
            
            # Вызов Piper CLI
            cmd = [
                self.piper_bin,
                '--model', str(self.model_path),
                '--output-raw'
            ]
            
            result = subprocess.run(
                cmd,
                input=text.encode('utf-8'),
                capture_output=True,
                check=True
            )
            
            return result.stdout
            
        except Exception as e:
            logger.error(f"Ошибка синтеза Piper: {e}", exc_info=True)
            raise
    
    def get_available_voices(self) -> list[str]:
        return list(self.MODELS.keys())
    
    def set_voice(self, voice_name: str):
        if voice_name in self.MODELS:
            self.voice = voice_name
            self.initialized = False  # Переинициализация
            logger.info(f"Голос изменен: {voice_name}")
