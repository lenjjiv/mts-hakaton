"""
Модуль для работы с моделью Faster Whisper.
"""

import os
from faster_whisper import WhisperModel

class FasterWhisperASR:
    """
    Класс для распознавания речи с помощью Faster Whisper.
    """

    def __init__(self, model_size="large-v3"):
        # Инициализация модели: пытаемся использовать GPU с FP16
        self.model = WhisperModel(model_size, device="cuda", compute_type="float16")

    def transcribe(self, audio_file_path, language=None):
        """
        Распознаёт речь из указанного WAV файла.

        :param audio_file_path: Путь к аудиофайлу.
        :param language: Код языка (например, "en" для английского). Если None — автоопределение.
        :return: Словарь с результатом распознавания.
        """
        segments, info = self.model.transcribe(audio_file_path, word_timestamps=True, language=language)
        segments = list(segments)
        text = " ".join([seg.text.strip() for seg in segments])
        result = {
            "language": info.language,
            "language_probability": info.language_probability,
            "text": text,
        }
        return result
