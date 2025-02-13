"""
Утилиты для работы с аудио.
"""

import wave

def save_audio_to_file(audio_data, file_path, sampling_rate=16000, channels=1, sample_width=2):
    """
    Сохраняет бинарные аудиоданные в WAV файл.

    :param audio_data: Бинарные аудиоданные.
    :param file_path: Путь для сохранения файла.
    :param sampling_rate: Частота дискретизации (по умолчанию 16000 Гц).
    :param channels: Количество аудиоканалов (по умолчанию 1 — моно).
    :param sample_width: Ширина сэмпла в байтах (по умолчанию 2).
    :return: Путь к созданному файлу.
    """
    with wave.open(file_path, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sampling_rate)
        wav_file.writeframes(audio_data)
    return file_path
