import asyncio
from typing import Tuple
import numpy as np

import torch
from torch.amp import autocast

import uvicorn
from fastapi import Depends, FastAPI, Request
import whisper

app = FastAPI()

# Параметры модели
MODEL_TYPE = "large-v3-turbo"
# RUN_TYPE и GPU_DEVICE_INDICES больше не используются, т.к. определяем устройство автоматически

def create_whisper_model():
    """
    Создает и инициализирует модель Whisper.
    
    Returns:
        tuple: (модель Whisper, устройство ('cuda:0' или 'cpu'))
    
    Raises:
        Exception: Если возникает ошибка при загрузке модели
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        model = whisper.load_model(MODEL_TYPE, device=device)
        if device.startswith("cuda"):
            model = model.half()  # Приводим модель к half precision для ускорения работы на GPU
    except Exception as e:
        print(f"Ошибка инициализации модели Whisper: {e}")
        raise
    return model, device

# Инициализация модели при старте приложения
try:
    model, device = create_whisper_model()
    print("Модель успешно загружена")
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    raise

async def parse_body(request: Request) -> bytes:
    """
    Асинхронно читает и возвращает тело запроса.
    
    Args:
        request (Request): Объект запроса FastAPI
    
    Returns:
        bytes: Тело запроса в виде байтов
    """
    try:
        data: bytes = await request.body()
        return data
    except Exception as e:
        print(f"Ошибка при чтении тела запроса: {e}")
        raise

async def transcribe_audio(
    model,
    device: str,
    audio_data_array: np.ndarray,
    language_code: str = ""
) -> Tuple[str, str, float]:
    """
    Асинхронно выполняет транскрипцию аудио данных с использованием модели Whisper.
    
    Args:
        model: Загруженная модель Whisper.
        device (str): Устройство, на котором запущена модель ('cuda:0' или 'cpu').
        audio_data_array (np.ndarray): Аудио данные.
        language_code (str, optional): Код языка. По умолчанию "".
    
    Returns:
        Tuple[str, str, float]: Кортеж (транскрипция, язык, вероятность определения языка).
                                Так как библиотека whisper не возвращает вероятность языка,
                                возвращается значение 1.0 по умолчанию.
    """
    try:
        loop = asyncio.get_event_loop()

        def run_transcription():
            if device.startswith("cuda"):
                with autocast(device_type='cuda', dtype=torch.float16):
                    return model.transcribe(
                        audio_data_array,
                        language=language_code if language_code else None
                    )
            else:
                return model.transcribe(
                    audio_data_array,
                    language=language_code if language_code else None
                )

        result = await loop.run_in_executor(None, run_transcription)
        transcription = result.get("text", "").strip()
        language = result.get("language", "unknown")
        language_probability = 1.0  # Whisper не предоставляет вероятность определения языка
        return transcription, language, language_probability

    except Exception as e:
        print(f"Ошибка при транскрипции: {e}")
        return "Error during transcription", "unknown", 0.0

@app.post("/predict")
async def predict(
    audio_data: bytes = Depends(parse_body), 
    language_code: str = ""
):
    """
    Эндпоинт для получения транскрипции аудио.
    
    Args:
        audio_data (bytes): Аудио данные в формате bytes.
        language_code (str, optional): Код языка. По умолчанию "".
    
    Returns:
        dict: Результат транскрипции с полями prediction, language и language_probability.
    """
    try:
        # Преобразуем байты в numpy массив.
        # Предполагается, что аудио данные закодированы в формате int16.
        audio_data_array = np.frombuffer(audio_data, np.int16).astype(np.float32) / 32768.0
        
        # Выполняем транскрипцию
        transcription, language, probability = await transcribe_audio(
            model, 
            device,
            audio_data_array,
            language_code
        )
        
        return {
            "prediction": transcription,
            "language": language,
            "language_probability": probability
        }
    
    except Exception as e:
        print(f"Ошибка в эндпоинте predict: {e}")
        return {
            "prediction": "Error processing request",
            "language": "unknown",
            "language_probability": 0.0
        }

if __name__ == "__main__":
    # Запускаем приложение FastAPI
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="asyncio",
        reload=True
    )