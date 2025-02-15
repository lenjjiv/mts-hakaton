from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/predict")
async def predict(request: Request):
    """
    Обрабатывает POST-запрос для транскрипции аудио.

    Принимает аудиоданные в виде байтов (формат int16) и возвращает JSON-ответ с результатом транскрипции.

    Возвращает:
        dict: JSON-объект с ключами:
            - prediction (str): Результат транскрипции.
            - language (str): Определённый язык.
            - language_probability (float): Вероятность определения языка.
    """
    try:
        # Получение тела запроса (аудиоданных)
        audio_bytes = await request.body()
        # Здесь должна быть логика обработки аудио (например, с использованием модели Whisper)
        # Для демонстрации возвращается фиктивный результат:
        transcription = "Пример транскрипции"
        language = "ru"
        language_probability = 1.0
        return {
            "prediction": transcription,
            "language": language,
            "language_probability": language_probability
        }
    except Exception as e:
        print(f"[*] Ошибка при обработке запроса транскрипции: {e}")
        return {
            "prediction": "ERROR",
            "language": "ERROR",
            "language_probability": 0.0
        }

if __name__ == "__main__":
    # Запуск сервера на порту 8000. Используйте команду:
    # uvicorn transcription_api:app --host 0.0.0.0 --port 8000
    uvicorn.run("transcription_api:app", host="0.0.0.0", port=8000, reload=True)