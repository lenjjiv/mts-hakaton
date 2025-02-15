from TTS.api import TTS
from pathlib import Path
import torch
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Функция для синтеза речи
def text_to_speech_coqui(text, model, file_to_save="output.wav", speed=1.0):
    """
    Преобразует текст в аудио.

    Args:
        text (str): Текст для озвучки
        model: Инициализированная модель TTS
        file_to_save (str, Path): Путь для сохранения файла с TTS (по умолчанию: "output.wav")
        speed (float): Скорость речи (по умолчанию: 1.0)

    Returns:
        None
    """
    try:
        # Генерация речи и сохранение в файл
        model.tts_to_file(
            text=text,
            file_path=file_to_save,
            speed=speed
        )

        logging.info(f"TTS сохранён в {file_to_save}")

    except Exception as e:
        logging.error(f"Ошибка во время генерации TTS: {e}")


if __name__ == "main":

    model_name = "tts_models/en/ljspeech/vits"
    device = "cuda"

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA недоступна, переключение на CPU.")
        device = "cpu"

    model = TTS(model_name=model_name)
    model = model.to(device)

    # Текст для синтеза
    text = "Artificial intelligence is revolutionizing the way we interact with technology. It has the potential to enhance our daily lives by providing personalized experiences and improving efficiency in various industries. As we continue to innovate, it is essential to consider the ethical implications of AI and ensure that it benefits everyone. Together, we can shape a future where technology and humanity coexist harmoniously."
    file_to_save = "output_1.mp3"

    # Вызов функции синтеза речи
    text_to_speech_coqui(text, model, file_to_save=file_to_save)