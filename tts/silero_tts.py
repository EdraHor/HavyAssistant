"""
Silero TTS - высококачественный русский TTS
"""

import torch
import logging
import io
import soundfile as sf
from pathlib import Path
from .base_tts import BaseTTS

logger = logging.getLogger(__name__)

class SileroTTS(BaseTTS):
    """Silero TTS для русского языка"""
    
    # Женские голоса (мягкие и спокойные)
    FEMALE_VOICES = {
        'baya': 'Женский, спокойный, мягкий',
        'kseniya': 'Женский, нейтральный',
        'xenia': 'Женский, выразительный',
        'aidar': 'Мужской (для сравнения)'
    }
    
    def __init__(self, device: str = 'cuda', voice: str = 'baya'):
        super().__init__()
        self.device = device
        self.voice = voice
        self.model = None
        self.sample_rate = 48000
        
    def initialize(self):
        """Инициализация Silero"""
        try:
            if self.initialized:
                logger.info("Silero TTS уже загружен")
                return
            
            logger.info("Загрузка Silero TTS...")
            
            # Проверка CUDA
            if self.device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA недоступна, переключаюсь на CPU")
                self.device = 'cpu'
            
            # Загрузка модели (автоматически скачается при первом запуске)
            model_path = Path('models/silero_tts')
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Silero v3 для русского
            self.model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker='v3_1_ru'
            )
            
            self.model.to(self.device)
            
            logger.info(f"✅ Silero TTS готов (устройство: {self.device}, голос: {self.voice})")
            self.initialized = True
            
        except Exception as e:
            logger.error(f"Ошибка инициализации Silero: {e}", exc_info=True)
            raise
    
    def synthesize(self, text: str) -> bytes:
        """Синтез речи"""
        try:
            if not self.initialized:
                self.initialize()
            
            logger.info(f"Синтез: '{text[:50]}...' (голос: {self.voice})")
            
            # Генерация аудио
            audio = self.model.apply_tts(
                text=text,
                speaker=self.voice,
                sample_rate=self.sample_rate,
                put_accent=True,
                put_yo=True
            )
            
            # Конвертация в WAV bytes
            audio_np = audio.cpu().numpy()
            
            # Используем io.BytesIO для создания WAV в памяти
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_np, self.sample_rate, format='WAV')
            wav_bytes = wav_buffer.getvalue()
            
            logger.info(f"✅ Синтез завершен ({len(wav_bytes)//1000}KB)")
            return wav_bytes
            
        except Exception as e:
            logger.error(f"Ошибка синтеза: {e}", exc_info=True)
            raise
    
    def get_available_voices(self) -> list[str]:
        """Список голосов"""
        return list(self.FEMALE_VOICES.keys())
    
    def set_voice(self, voice_name: str):
        """Установить голос"""
        if voice_name in self.FEMALE_VOICES:
            self.voice = voice_name
            logger.info(f"Голос изменен на: {voice_name} ({self.FEMALE_VOICES[voice_name]})")
        else:
            logger.warning(f"Голос {voice_name} не найден")
    
    def cleanup(self):
        """Очистка"""
        try:
            if self.model:
                self.model = None
                torch.cuda.empty_cache()
            logger.info("✓ Silero TTS очищен")
        except Exception as e:
            logger.error(f"Ошибка очистки: {e}")
