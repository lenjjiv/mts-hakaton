### server_tts.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from TTS.api import TTS
import torch
import torchaudio
import os
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Инициализация FastAPI
app = FastAPI()

# ----------- Инициализация моделей ----------- #

# Определение устройства (GPU или CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Coqui TTS (английский)
model_en = TTS(model_name="tts_models/en/ljspeech/vits")
model_en = model_en.to(device)

# Silero TTS (русский)
language = 'ru'
model_id = 'v4_ru'
model_speaker = 'aidar'

# Загрузка модели Silero TTS согласно требуемому примеру
model_ru, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language=language,
    speaker=model_id
)
model_ru.to(device)

# ----------- Функции синтеза речи ----------- #

def text_to_speech_coqui(text: str, file_to_save="output_en.mp3", speed=1.0):
    """Генерирует аудио из текста (Coqui TTS)."""
    try:
        model_en.tts_to_file(text=text, file_path=file_to_save, speed=speed)
        logger.info(f"TTS сохранён в {file_to_save}")
        return file_to_save
    except Exception as e:
        logger.error(f"Ошибка в Coqui TTS: {e}")
        return None

def text_to_speech_silero(
        text: str, 
        file_to_save="output_ru.mp3", 
        sample_rate=24000,
        speaker=model_speaker, 
        max_string_length: int = 999,
        put_accent=True, 
        put_yo=True
    ):
    """Генерирует аудио из текста (Silero TTS) с использованием загруженной модели."""
    try:
        audio = model_ru.apply_tts(
            text=text[:max_string_length],
            speaker=speaker,
            sample_rate=sample_rate,
            put_accent=put_accent,
            put_yo=put_yo
        )
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        torchaudio.save(file_to_save, audio, sample_rate)
        logger.info(f"TTS сохранён в {file_to_save}")
        return file_to_save
    except Exception as e:
        logger.error(f"Ошибка в Silero TTS: {e}")
        return None

# ----------- Функция для стриминга аудиофайла ----------- #
async def file_streamer(file_path: str):
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            yield chunk

@app.post("/stream-audio")
async def stream_audio(request: Request):
    """Эндпоинт для генерации речи."""
    data = await request.json()
    text = data.get("text", "").strip()
    lang = data.get("lang", "en").lower()

    if not text:
        raise HTTPException(status_code=400, detail="Текст не должен быть пустым")

    os.makedirs("tmp_server", exist_ok=True)
    if lang == "ru":
        audio_path = "tmp_server/generated_audio_ru.mp3"
        generated_file = text_to_speech_silero(text, file_to_save=audio_path)
    else:
        audio_path = "tmp_server/generated_audio_en.mp3"
        generated_file = text_to_speech_coqui(text, file_to_save=audio_path)

    if not generated_file or not os.path.exists(audio_path):
        raise HTTPException(status_code=500, detail="Ошибка генерации аудио")

    return StreamingResponse(
        file_streamer(audio_path),
        media_type="audio/mpeg",
        headers={"Content-Disposition": f'attachment; filename="{os.path.basename(audio_path)}"'}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
