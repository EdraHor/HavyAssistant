#!/usr/bin/env python3
"""
Тест Chatterbox TTS - изолированный от основного проекта
"""

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("=" * 60)
print("🎤 Тест Chatterbox Multilingual TTS")
print("=" * 60)

# Проверка CUDA
print(f"\n✓ PyTorch: {torch.__version__}")
print(f"✓ CUDA доступна: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Загрузка модели
print("\n⏳ Загрузка модели Chatterbox...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxMultilingualTTS.from_pretrained(device=device)
print("✅ Модель загружена!")

# Тест 1: Дефолтный голос
print("\n📝 Тест 1: Дефолтный голос (русский)")
text_ru = "Привет! Это тестовое сообщение голосового ассистента на русском языке."
wav_ru = model.generate(text_ru, language_id="ru")
ta.save("test_russian_default.wav", wav_ru, model.sr)
print(f"✅ Сохранено: test_russian_default.wav ({len(wav_ru)//1000}K сэмплов)")

# Тест 2: Английский
print("\n📝 Тест 2: Английский")
text_en = "Hello! This is a test message from the voice assistant."
wav_en = model.generate(text_en, language_id="en")
ta.save("test_english_default.wav", wav_en, model.sr)
print(f"✅ Сохранено: test_english_default.wav")

# Тест 3: С эмоциями (если есть параметр)
print("\n📝 Тест 3: С эмоциональностью")
text_emotion = "Какая замечательная погода сегодня! Я очень рад!"
wav_emotion = model.generate(
    text_emotion,
    language_id="ru",
    exaggeration=0.8  # Больше эмоций
)
ta.save("test_emotion.wav", wav_emotion, model.sr)
print(f"✅ Сохранено: test_emotion.wav")

print("\n" + "=" * 60)
print("✅ Все тесты завершены!")
print("=" * 60)
print("\n📁 Результаты в папке: tts_test/")
print("🎧 Прослушайте .wav файлы и оцените качество")