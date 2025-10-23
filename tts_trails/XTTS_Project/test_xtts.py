from TTS.api import TTS
import torch

# Проверка CUDA
print(f"CUDA доступна: {torch.cuda.is_available()}")
print(f"Устройство: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Загрузка модели
print("Загрузка модели XTTS-v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

print("Модель загружена успешно!")
print(f"Поддерживаемые языки: {tts.languages}")