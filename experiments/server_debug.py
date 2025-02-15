import os
import asyncio
from datetime import datetime

import uvicorn
from fastapi import Depends, FastAPI, Request

app = FastAPI()

UPLOAD_DIR = "./uploads"
FIXED_TRANSCRIPTION = "пример транскрипта на русском"

# Убедимся, что папка для загрузок существует
os.makedirs(UPLOAD_DIR, exist_ok=True)

async def parse_body(request: Request) -> bytes:
    """
    Асинхронно читает и возвращает тело запроса.
    """
    try:
        data: bytes = await request.body()
        return data
    except Exception as e:
        print(f"Ошибка при чтении тела запроса: {e}")
        raise

@app.post("/predict")
async def predict(
    audio_data: bytes = Depends(parse_body), 
    language_code: str = ""
):
    """
    Эндпоинт для сохранения файла и возврата фиксированного ответа.
    """
    try:
        # Сохраняем полученные данные на диск с уникальным именем
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        file_path = os.path.join(UPLOAD_DIR, f"audio_{timestamp}.bin")
        with open(file_path, "wb") as f:
            f.write(audio_data)
        print(f"Файл сохранён: {file_path}")

        # Возвращаем фиксированный результат
        return {
            "prediction": FIXED_TRANSCRIPTION,
            "language": "ru",
            "language_probability": 1.0
        }
    
    except Exception as e:
        print(f"Ошибка в эндпоинте predict: {e}")
        return {
            "prediction": "Ошибка при обработке запроса",
            "language": "unknown",
            "language_probability": 0.0
        }

if __name__ == "__main__":
    # Запускаем приложение FastAPI
    uvicorn.run(
        "server:app",  # формат "файл:app"
        host="0.0.0.0",
        port=8000,
        workers=1,   # Один воркер для последовательной работы
        loop="asyncio",
        reload=True  # Автоперезагрузка для разработки
    )
