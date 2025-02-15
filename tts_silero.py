import torch
import torchaudio
from pathlib import Path
import logging


logging.basicConfig(
    level=logging.INFO,  # Установим уровень логирования
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # Вывод в консоль
    ]
)

logger = logging.getLogger(__name__)
logger.info("Вызван tts_silero.py")


def text_to_speech_silero(text, model, file_to_save="output.mp3", sample_rate=24000, speaker='eugene', put_accent=True, put_yo=True):
    """
    Преобразует текст в аудио, используя модель Silero TTS.

    Args:
        text (str): Текст для озвучки (может быть обычным текстом или SSML)
        model: Инициализированная модель TTS Silero
        file_to_save (str): Путь для сохранения файла с TTS (по умолчанию: "output.mp3")
        sample_rate (int): Частота дискретизации аудио (по умолчанию: 24000)
        speaker (str): Имя диктора (по умолчанию: 'eugene')
        put_accent (bool): Расставлять ударения (по умолчанию: True)
        put_yo (bool): Использовать букву "ё" (по умолчанию: True)

    Returns:
        None
    """
    try:
        # Проверяем, является ли текст SSML
        if "<speak>" in text:
            audio = model.apply_tts(
                ssml_text=text,
                speaker=speaker,
                sample_rate=sample_rate,
                put_accent=put_accent,
                put_yo=put_yo
            )
        else:
            audio = model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=sample_rate,
                put_accent=put_accent,
                put_yo=put_yo
            )

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        torchaudio.save(file_to_save, audio, sample_rate)
        logging.info(f"TTS сохранён в {file_to_save}")

    except Exception as e:
        logging.error(f"Ошибка во время генерации TTS: {e}")


if __name__ == "__main__":

    # Параметры модели
    language = 'ru'
    model_id = 'v4_ru'
    device = 'cuda'
    
    # Проверка доступности CUDA и переключение на CPU при необходимости
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, переключение на CPU.")
        device = "cpu"
    
    # Загрузка модели Silero TTS
    model, example_text  = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language=language,
        speaker=model_id
    )
    
    model.to(device)

    text = "<speak>Привет, это пример текста, который будет преобразован в аудио.</speak>"
    
    file_to_save = "output_silero_2.mp3"
    
    text_to_speech_silero(text=text, model=model, file_to_save=file_to_save)
