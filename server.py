from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from TTS.api import TTS
import os
import torch
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI()

# Определение модели Coqui TTS
model_name = "tts_models/en/ljspeech/vits"
device = "cuda" if torch.cuda.is_available() else "cpu"

logger.info(f"Используем устройство: {device}")
model = TTS(model_name=model_name)
model = model.to(device)

# Функция для генерации аудиофайла
def text_to_speech_coqui(text: str, file_to_save="output.mp3", speed=1.0):
    try:
        model.tts_to_file(text=text, file_path=file_to_save, speed=speed)
        logger.info(f"TTS сохранён в {file_to_save}")
        return file_to_save
    except Exception as e:
        logger.error(f"Ошибка во время генерации TTS: {e}")
        return None

# Функция для стриминга аудиофайла
async def file_streamer(file_path: str):
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            yield chunk

@app.post("/stream-audio")
async def stream_audio(request: Request):
    data = await request.json()
    text = data.get("text", "").strip()

    if not text:
        return {"error": "Текст не должен быть пустым"}

    audio_path: str = "generated_audio.mp3"
    text_to_speech_coqui(text, file_to_save=audio_path)

    if not generated_file or not os.path.exists(audio_path):
        return {"error": "Ошибка генерации аудио"}

    return StreamingResponse(
        file_streamer(audio_path),
        media_type="audio/mpeg",
        headers={"Content-Disposition": f'attachment; filename="audio.mp3"'}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
