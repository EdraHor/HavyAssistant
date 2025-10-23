from pathlib import Path
import soundfile as sf
import librosa
import numpy as np
from TTS.api import TTS
import torch


def load_and_normalize_audio(audio_path, target_sr=22050):
    """Загрузка и нормализация аудио"""
    print(f"    Загрузка: {audio_path.name}")
    audio_data, sr = librosa.load(str(audio_path), sr=target_sr)

    # Нормализация громкости
    audio_data = audio_data / np.max(np.abs(audio_data))

    return audio_data, target_sr


def combine_audio_files(audio_files, output_path="combined_reference.wav", target_sr=22050):
    """Объединение нескольких аудио файлов в один референс"""
    print(f"\n🔗 Объединение {len(audio_files)} файлов в один референс...")

    combined_audio = []
    total_duration = 0

    for audio_file in audio_files:
        audio_data, sr = load_and_normalize_audio(audio_file, target_sr)
        combined_audio.append(audio_data)
        duration = len(audio_data) / sr
        total_duration += duration
        print(f"      ✓ {audio_file.name} ({duration:.2f}s)")

    # Объединяем все файлы
    final_audio = np.concatenate(combined_audio)

    # Сохраняем
    sf.write(output_path, final_audio, target_sr)

    print(f"\n✓ Объединенный референс: {output_path}")
    print(f"  Общая длительность: {total_duration:.2f}s")
    print(f"  Оптимально: 10-30 секунд")

    if total_duration < 5:
        print("  ⚠ Слишком короткий! Добавьте больше файлов для лучшего качества.")
    elif total_duration > 60:
        print("  ⚠ Слишком длинный! Может замедлить генерацию.")
    else:
        print("  ✓ Отличная длина!")

    return output_path, total_duration


def reduce_noise_simple(audio_data, sr):
    """Простое шумоподавление"""
    # Удаление тишины в начале и конце
    audio_data, _ = librosa.effects.trim(audio_data, top_db=20)
    return audio_data


def prepare_reference(ref_dir, use_all=True, specific_files=None):
    """Подготовка референса из файлов в Reference/"""
    ref_dir = Path(ref_dir)

    if specific_files:
        # Используем конкретные файлы
        audio_files = [ref_dir / f for f in specific_files]
    elif use_all:
        # Используем все файлы
        audio_files = list(ref_dir.glob("*.opus")) + list(ref_dir.glob("*.wav"))
    else:
        return None

    audio_files = [f for f in audio_files if f.exists()]

    if not audio_files:
        return None

    # Сортируем для предсказуемости
    audio_files = sorted(audio_files)

    # Объединяем
    combined_path, duration = combine_audio_files(
        audio_files,
        output_path="temp_wav/combined_reference.wav"
    )

    return combined_path


# Инициализация
print("=" * 70)
print(" " * 15 + "XTTS-v2 Quality Generator")
print("=" * 70)

# Создание папок
Path("Reference").mkdir(exist_ok=True)
Path("Output").mkdir(exist_ok=True)
Path("temp_wav").mkdir(exist_ok=True)

# Проверка CUDA
print(f"\n🔧 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

print("\n📦 Загрузка модели XTTS-v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
print("✓ Модель загружена!\n")

# Поиск референсов
ref_dir = Path("Reference")
ref_files = list(ref_dir.glob("*.opus")) + list(ref_dir.glob("*.wav"))

if not ref_files:
    print("⚠ Нет файлов в папке Reference/")
    print("  Поместите несколько .opus или .wav файлов и запустите снова.")
    exit()

print(f"📁 Найдено файлов в Reference/: {len(ref_files)}")
for i, f in enumerate(ref_files, 1):
    # Получаем длительность
    try:
        audio, sr = librosa.load(str(f), sr=None)
        duration = len(audio) / sr
        print(f"  {i}. {f.name} ({duration:.2f}s)")
    except:
        print(f"  {i}. {f.name}")

# РЕЖИМЫ РАБОТЫ
print("\n" + "=" * 70)
print("ВЫБЕРИТЕ РЕЖИМ:")
print("=" * 70)
print("1. Использовать ВСЕ файлы как объединенный референс (рекомендуется)")
print("2. Выбрать конкретные файлы")
print("3. Использовать один файл")
print("=" * 70)

mode = input("\nВыберите режим (1/2/3): ").strip()

reference_wav = None

if mode == "1":
    # Объединяем все файлы
    reference_wav = prepare_reference(ref_dir, use_all=True)

elif mode == "2":
    # Выбор конкретных файлов
    print("\nВведите номера файлов через запятую (например: 1,3,5):")
    indices = input().strip().split(",")
    try:
        selected_files = [ref_files[int(i.strip()) - 1].name for i in indices]
        reference_wav = prepare_reference(ref_dir, use_all=False, specific_files=selected_files)
    except (ValueError, IndexError):
        print("⚠ Неверные номера!")
        exit()

elif mode == "3":
    # Один файл
    print("\nВведите номер файла:")
    try:
        idx = int(input().strip()) - 1
        single_file = ref_files[idx]

        # Конвертируем если нужно
        if single_file.suffix.lower() != '.wav':
            audio, sr = librosa.load(str(single_file), sr=22050)
            reference_wav = "temp_wav/single_reference.wav"
            sf.write(reference_wav, audio, sr)
        else:
            reference_wav = str(single_file)

        print(f"✓ Выбран: {single_file.name}")

    except (ValueError, IndexError):
        print("⚠ Неверный номер!")
        exit()
else:
    print("⚠ Неверный режим!")
    exit()

if not reference_wav:
    print("⚠ Не удалось подготовить референс!")
    exit()

# Генерация
print("\n" + "=" * 70)
print("ГЕНЕРАЦИЯ РЕЧИ")
print("=" * 70)

while True:
    print("\nВведите текст для озвучки (или 'q' для выхода):")
    text = input().strip()

    if text.lower() == 'q':
        break

    if not text:
        print("⚠ Текст не может быть пустым!")
        continue

    # Параметры генерации
    print("\n🎛️  Настройки качества:")
    print("  1. Стандартное (быстро)")
    print("  2. Высокое (медленнее, лучше просодия)")

    quality = input("Выберите (1/2, Enter=1): ").strip() or "1"

    output_name = f"output_{len(list(Path('Output').glob('*.wav')))}.wav"
    output_path = Path("Output") / output_name

    print(f"\n🎙️  Генерация: {text[:50]}{'...' if len(text) > 50 else ''}")

    try:
        if quality == "2":
            # Высокое качество - разбиваем на предложения
            tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=reference_wav,
                language="ru",
                split_sentences=True  # Лучшая просодия
            )
        else:
            # Стандартное
            tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=reference_wav,
                language="ru"
            )

        print(f"✓ Сохранено: {output_path}")

    except Exception as e:
        print(f"✗ Ошибка: {e}")

print("\n" + "=" * 70)
print("Завершено!")
print("=" * 70)