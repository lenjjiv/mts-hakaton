# server.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import os

app = FastAPI()

# Размер чанка для стриминга (8KB)
CHUNK_SIZE = 8192

async def file_streamer(file_path: str):
    with open(file_path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            yield chunk

@app.post("/stream-audio")
async def stream_audio(request: Request):
    # Получаем JSON из тела запроса
    data = await request.json()
    text = data.get("text", "")
    
    # Путь к MP3 файлу (замените на свой)
    mp3_path = "response.mp3"
    
    # Проверяем существование файла
    if not os.path.exists(mp3_path):
        return {"error": "Audio file not found"}
    
    # Возвращаем стрим MP3 файла
    return StreamingResponse(
        file_streamer(mp3_path),
        media_type="audio/mpeg",
        headers={
            "Content-Disposition": f'attachment; filename="response.mp3"'
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# client.py
import requests
import json

def stream_audio(text: str, save_path: str):
    # URL сервера
    url = "http://localhost:8000/stream-audio"
    
    # Подготавливаем данные для отправки
    data = {"text": text}
    
    try:
        # Отправляем POST запрос
        response = requests.post(
            url,
            json=data,
            stream=True  # Включаем стриминг
        )
        
        # Проверяем статус ответа
        response.raise_for_status()
        
        # Открываем файл для записи в бинарном режиме
        with open(save_path, 'wb') as f:
            # Записываем данные по чанкам
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
        print(f"Аудио файл успешно сохранен в {save_path}")
        
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении аудио: {e}")

# Пример использования клиента
if __name__ == "__main__":
    text = "Привет, это тестовый текст!"
    save_path = "downloaded_audio.mp3"
    stream_audio(text, save_path)
