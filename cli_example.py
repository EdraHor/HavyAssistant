#!/usr/bin/env python3
"""
CLI версия голосового ассистента
Демонстрация работы VoiceAssistantController без GUI
"""

import sys
import time
import logging
from services.assistant_controller import VoiceAssistantController, AssistantState
from utils.logger import setup_logger

def main():
    # Настройка логирования
    setup_logger()
    logger = logging.getLogger(__name__)
    
    print("=" * 60)
    print("    Голосовой Ассистент - CLI версия")
    print("=" * 60)
    print()
    
    try:
        # Создаем контроллер
        controller = VoiceAssistantController()
        
        # Подключаем обработчики событий
        def on_status_update(text, color):
            print(f"[СТАТУС] {text}")
        
        def on_log(message):
            print(f"[ЛОГ] {message}")
        
        def on_wake_word_detected():
            print("[WAKE WORD] Обнаружено ключевое слово!")
        
        def on_speech_recognized(text):
            print(f"[РАСПОЗНАНО] {text}")
        
        def on_llm_response(response):
            print(f"[ОТВЕТ] {response}")
            print()
        
        def on_llm_error(error):
            print(f"[ОШИБКА LLM] {error}")
        
        def on_error(error):
            print(f"[ОШИБКА] {error}")
        
        def on_history_updated(count):
            print(f"[ИСТОРИЯ] {count} сообщений")
        
        # Подключаем callbacks
        controller.set_callback('on_status_update', on_status_update)
        controller.set_callback('on_log', on_log)
        controller.set_callback('on_wake_word_detected', on_wake_word_detected)
        controller.set_callback('on_speech_recognized', on_speech_recognized)
        controller.set_callback('on_llm_response', on_llm_response)
        controller.set_callback('on_llm_error', on_llm_error)
        controller.set_callback('on_error', on_error)
        controller.set_callback('on_history_updated', on_history_updated)
        
        # Получаем список микрофонов
        devices = controller.get_audio_devices()
        
        if not devices:
            print("❌ Не найдено аудио устройств!")
            return 1
        
        print("Доступные микрофоны:")
        for i, device in enumerate(devices):
            print(f"  {i}: {device['name']}")
        print()
        
        # Выбираем первое устройство по умолчанию
        device_index = devices[0]['index']
        print(f"Используется: {devices[0]['name']}")
        print()
        
        # Запускаем ассистент
        wake_word = "привет ассистент"
        print(f"Запуск ассистента с ключевым словом: '{wake_word}'")
        print("Нажмите Ctrl+C для остановки")
        print("-" * 60)
        print()
        
        success = controller.start(device_index, wake_word)
        
        if not success:
            print("❌ Не удалось запустить ассистент!")
            return 1
        
        # Ждем завершения (Ctrl+C)
        try:
            while True:
                time.sleep(1)
                
                # Проверяем состояние
                if controller.state == AssistantState.ERROR:
                    print("❌ Ассистент в состоянии ошибки, перезапуск...")
                    controller.stop()
                    time.sleep(2)
                    controller.start(device_index, wake_word)
                    
        except KeyboardInterrupt:
            print("\n\nОстановка ассистента...")
            controller.stop()
            print("✓ Остановлено")
        
        return 0
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        print(f"\n❌ Критическая ошибка: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
