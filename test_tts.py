#!/usr/bin/env python3
"""
Тест TTS модуля
"""

import sys
import logging
from utils.logger import setup_logger
from tts import TTSService

def main():
    # Настройка логирования
    setup_logger()
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("    Тест TTS модуля")
    print("=" * 60)
    print()
    
    try:
        # Создаем TTS сервис
        tts = TTSService()
        
        # Инициализация
        print("Инициализация TTS...")
        tts.initialize()
        print("✅ TTS готов!")
        print()
        
        # Показываем доступные голоса
        voices = tts.get_available_voices()
        print(f"Доступные голоса: {', '.join(voices)}")
        print()
        
        # Тестовые фразы
        test_phrases = [
            "Привет! Это тестовое озвучивание.",
            "Меня зовут Бая, я голосовой ассистент.",
            "Погода сегодня замечательная!",
            "Как дела? Чем могу помочь?",
        ]
        
        # Озвучиваем каждую фразу
        for i, phrase in enumerate(test_phrases, 1):
            print(f"[{i}/{len(test_phrases)}] Озвучивание: {phrase}")
            tts.speak(phrase)
            print("✅ Завершено")
            print()
        
        # Тест смены голоса (если доступно)
        if len(voices) > 1:
            print(f"Переключение на голос: {voices[1]}")
            tts.set_voice(voices[1])
            tts.speak("Теперь я говорю другим голосом!")
            print("✅ Завершено")
            print()
        
        print("=" * 60)
        print("✅ Все тесты пройдены успешно!")
        print("=" * 60)
        
        # Очистка
        tts.cleanup()
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nПрервано пользователем")
        return 0
        
    except Exception as e:
        logger.error(f"Ошибка теста: {e}", exc_info=True)
        print(f"\n❌ Ошибка: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
