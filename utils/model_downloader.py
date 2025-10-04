"""
Утилита для проверки и загрузки необходимых моделей
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def check_models() -> dict:
    """
    Проверка наличия всех необходимых моделей
    
    Returns:
        dict: Статус каждой модели
    """
    status = {
        'vosk': False,
        'whisper': True,  # Whisper скачивается автоматически faster-whisper
    }
    
    # Проверка Vosk
    vosk_path = Path("models/vosk-model-small-ru-0.22")
    vosk_marker = vosk_path / "am" / "final.mdl"
    status['vosk'] = vosk_marker.exists()
    
    return status

def are_all_models_ready() -> bool:
    """
    Проверка готовности всех моделей
    
    Returns:
        bool: True если все модели готовы
    """
    status = check_models()
    return all(status.values())

def get_missing_models() -> list:
    """
    Получить список отсутствующих моделей
    
    Returns:
        list: Названия отсутствующих моделей
    """
    status = check_models()
    return [name for name, ready in status.items() if not ready]

def estimate_download_time() -> str:
    """
    Оценка времени загрузки
    
    Returns:
        str: Примерное время
    """
    missing = get_missing_models()
    
    if not missing:
        return "0 минут"
    
    # Vosk ~50MB
    if 'vosk' in missing:
        return "1-3 минуты (зависит от скорости интернета)"
    
    return "неизвестно"

def get_models_info() -> str:
    """
    Информация о моделях для пользователя
    
    Returns:
        str: Текст с информацией
    """
    status = check_models()
    missing = get_missing_models()
    
    if not missing:
        return "✅ Все модели установлены"
    
    info = "📥 Требуется загрузка моделей:\n\n"
    
    if 'vosk' in missing:
        info += "• Vosk (Wake Word Detection) - ~50MB\n"
    
    info += f"\n⏱️ Примерное время: {estimate_download_time()}\n"
    info += "\nМодели загрузятся автоматически при первом запуске."
    
    return info
