import asyncio
from typing import Tuple

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Request
from faster_whisper import WhisperModel

app = FastAPI()

MODEL_TYPE = "large-v2"
RUN_TYPE = "gpu"  # "cpu" or "gpu"

# For CPU usage (https://github.com/SYSTRAN/faster-whisper/issues/100#issuecomment-1492141352)
NUM_WORKERS = 10
CPU_THREADS = 4

# For GPU usage
GPU_DEVICE_INDICES = [0]

VAD_FILTER = True

def create_whisper_model() -> WhisperModel:
    if RUN_TYPE.lower() == "gpu":
        whisper = WhisperModel(
            MODEL_TYPE,
            device="cuda",
            compute_type="float16",
            device_index=GPU_DEVICE_INDICES,
            download_root="./models"
        )

    elif RUN_TYPE.lower() == "cpu":
        whisper = WhisperModel(MODEL_TYPE,
            device="cpu",
            compute_type="int8",
            num_workers=NUM_WORKERS,
            cpu_threads=CPU_THREADS,
            download_root="./models"
        )
    else:
        raise ValueError(f"Invalid model type: {RUN_TYPE}")

    print("Loaded model")

    return whisper


model = create_whisper_model()
print("Loaded model")


async def parse_body(request: Request):
    data: bytes = await request.body()
    return data


def transcribe_audio(
    model: WhisperModel,
    audio_data_array: np.ndarray,
    language_code: str = ""
) -> Tuple[str, str, float]:
    """
    Выполняет синхронную транскрипцию аудиоданных с помощью модели Whisper.
    
    Аргументы:
        model (WhisperModel): Экземпляр модели Whisper.
        audio_data_array (np.ndarray): Аудиоданные в виде массива numpy.float32.
        language_code (str): Код языка (если пустой, автоопределение).

    Возвращает:
        tuple: Кортеж, содержащий транскрипцию, определённый язык и вероятность определения языка.
        
    Исключения:
        RuntimeError: Если возникла ошибка при работе с GPU
        ValueError: Если входные данные некорректны
    """
    try:
        language_code = language_code.lower().strip()
        
        # Проверка валидности входных данных
        if len(audio_data_array) == 0:
            raise ValueError("Пустой массив аудиоданных")
            
        # Проверка на NaN и бесконечности
        if np.isnan(audio_data_array).any() or np.isinf(audio_data_array).any():
            raise ValueError("Некорректные значения в аудиоданных")

        segments, info = model.transcribe(
            audio_data_array,
            language=language_code if language_code != "" else None,
            beam_size=5,
            vad_filter=VAD_FILTER,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Проверка результатов
        if not hasattr(info, 'language') or not hasattr(info, 'language_probability'):
            raise RuntimeError("Некорректный результат от модели")
            
        segments = [s.text for s in segments]
        transcription = " ".join(segments).strip()
        
        return transcription, info.language, info.language_probability
        
    except Exception as e:
        print(f"[*] Ошибка в transcribe_audio: {str(e)}")
        # Возвращаем безопасные значения по умолчанию
        return "", "unknown", 0.0


@app.post("/predict")
async def predict(
        audio_data: bytes = Depends(parse_body), 
        language_code: str = ""
    ) -> dict:
    """
    Асинхронный обработчик POST-запросов для транскрипции аудио.
    
    Аргументы:
        audio_data (bytes): Аудиоданные в бинарном формате
        language_code (str): Код языка для транскрипции
        
    Возвращает:
        dict: Словарь с результатами транскрипции:
            - prediction: текст транскрипции
            - language: определённый язык
            - language_probability: вероятность определения языка
    """
    try:
        # Проверка входных данных
        if not audio_data:
            raise ValueError("Пустые аудиоданные")
            
        # Конвертация аудио в numpy массив с проверкой
        audio_data_array = np.frombuffer(audio_data, np.int16)
        if len(audio_data_array) == 0:
            raise ValueError("Ошибка при конвертации аудиоданных")
            
        # Нормализация данных
        audio_data_array = audio_data_array.astype(np.float32) / 255.0

        # Запуск транскрипции в отдельном потоке
        try:
            result = await asyncio.get_running_loop().run_in_executor(
                None, 
                transcribe_audio, 
                model, 
                audio_data_array,
                language_code
            )
        except Exception as e:
            print(f"[*] Ошибка при выполнении транскрипции: {str(e)}")
            result = ("", "error", 0.0)

        # Формирование ответа с проверкой
        response = {
            "prediction": result[0] if result[0] else "",
            "language": result[1] if result[1] else "unknown",
            "language_probability": float(result[2]) if result[2] is not None else 0.0
        }
        
        return response

    except Exception as e:
        print(f"[*] Общая ошибка в predict: {str(e)}")
        # Возвращаем безопасный ответ в случае ошибки
        return {
            "prediction": "",
            "language": "error",
            "language_probability": 0.0
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
