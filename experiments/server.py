import asyncio
from typing import Tuple
import numpy as np

import torch
torch.backends.cudnn.enabled = False

import uvicorn
from fastapi import Depends, FastAPI, Request
from faster_whisper import WhisperModel

app = FastAPI()

MODEL_TYPE = "large-v2"
RUN_TYPE = "gpu"
GPU_DEVICE_INDICES = [0]
VAD_FILTER = False

def create_whisper_model() -> WhisperModel:
    """
    Создает и инициализирует модель Whisper.
    
    Returns:
        WhisperModel: Инициализированная модель для транскрипции
    
    Raises:
        ValueError: Если указан неверный тип запуска (не cpu/gpu)
    """
    if RUN_TYPE.lower() == "gpu":
        # Для GPU важно явно указать compute_type и убедиться, что CUDA доступна
        try:
            whisper = WhisperModel(
                MODEL_TYPE,
                device="cuda",
                compute_type="float16",
                device_index=GPU_DEVICE_INDICES,
                download_root="./models",
                local_files_only=False  # Предотвращает попытки загрузки во время инференса
            )
        except Exception as e:
            print(f"Ошибка инициализации GPU модели: {e}")
            raise
    else:
        raise ValueError(f"Неподдерживаемый тип запуска: {RUN_TYPE}")

    return whisper

# Инициализация модели при старте приложения
try:
    model = create_whisper_model()
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
    model: WhisperModel,
    audio_data_array: np.ndarray,
    language_code: str = ""
) -> Tuple[str, str, float]:
    """
    Асинхронно выполняет транскрипцию аудио данных.
    
    Args:
        model (WhisperModel): Модель Whisper
        audio_data_array (np.ndarray): Аудио данные
        language_code (str, optional): Код языка. По умолчанию ""
    
    Returns:
        Tuple[str, str, float]: Кортеж (транскрипция, язык, вероятность определения языка)
    """
    try:
        # Запускаем тяжелую операцию транскрипции в отдельном потоке
        segments, info = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: model.transcribe(
                audio_data_array,
                language=language_code if language_code != "" else None,
                beam_size=5,
                vad_filter=VAD_FILTER,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
        )
        
        segments_text = [s.text for s in segments]
        transcription = " ".join(segments_text).strip()
        return transcription, info.language, info.language_probability
    
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
        audio_data (bytes): Аудио данные в формате bytes
        language_code (str, optional): Код языка. По умолчанию ""
    
    Returns:
        dict: Результат транскрипции с полями prediction, language и language_probability
    """
    try:
        # Конвертируем байты в numpy массив
        audio_data_array = np.frombuffer(audio_data, np.int16).astype(np.float32) / 255.0
        
        # Выполняем транскрипцию
        transcription, language, probability = await transcribe_audio(
            model, 
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
    # Запускаем с правильными настройками для FastAPI
    uvicorn.run(
        "server:app",  # Важно указать путь к приложению в формате "файл:app"
        host="0.0.0.0",
        port=8000,
        workers=1,  # Для GPU лучше использовать один воркер
        loop="asyncio",
        reload=True  # Отключаем автоперезагрузку для продакшена
    )