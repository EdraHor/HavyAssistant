"""
Сервис захвата аудио с микрофона
"""

import sounddevice as sd
import numpy as np
import logging
from typing import Callable, List, Dict
from utils.config_loader import config

logger = logging.getLogger(__name__)

class AudioCaptureService:
    """Захват аудио с микрофона"""
    
    def __init__(self):
        self.sample_rate = config.get('audio.sample_rate', 16000)
        self.channels = config.get('audio.channels', 1)
        self.chunk_size = config.get('audio.chunk_size', 1024)
        self.device_index = config.get('audio.device_index')
        
        self.stream = None
        self.callback = None
        self.is_recording = False
    
    def get_audio_devices(self) -> List[Dict]:
        """Получить список доступных микрофонов"""
        devices = []
        device_list = sd.query_devices()
        
        for i, device in enumerate(device_list):
            if device['max_input_channels'] > 0:  # Только входные устройства
                devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': int(device['default_samplerate'])
                })
        
        logger.info(f"Найдено микрофонов: {len(devices)}")
        return devices
    
    def start_capture(self, device_index: int, callback: Callable[[bytes], None]):
        """
        Начать захват аудио
        
        Args:
            device_index: Индекс устройства
            callback: Функция обработки аудио данных
        """
        if self.is_recording:
            logger.warning("Захват уже запущен")
            return
        
        self.callback = callback
        self.device_index = device_index
        
        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            
            # Конвертируем в байты (int16)
            audio_data = (indata * 32767).astype(np.int16).tobytes()
            
            if self.callback:
                self.callback(audio_data)
        
        try:
            self.stream = sd.InputStream(
                device=device_index,
                channels=self.channels,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=audio_callback,
                dtype=np.float32
            )
            
            self.stream.start()
            self.is_recording = True
            logger.info(f"Захват аудио запущен: устройство {device_index}, {self.sample_rate}Hz")
            
        except Exception as e:
            logger.error(f"Ошибка запуска захвата: {e}")
            raise

    def stop_capture(self):
        """Остановить захват аудио"""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                logger.error(f"Ошибка остановки stream: {e}")
            finally:
                self.stream = None
                self.is_recording = False
                self.callback = None

        logger.info("Захват аудио остановлен")
    
    def __del__(self):
        self.stop_capture()
