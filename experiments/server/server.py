"""
Модуль WebSocket сервера, который принимает аудиоданные от клиента,
сохраняет их во временный WAV файл и запускает распознавание речи с помощью Faster Whisper.
"""

import asyncio
import os
import tempfile
import json
import websockets

from server.asr import FasterWhisperASR
from server.audio_utils import save_audio_to_file

# Инициализируем модель ASR (Faster Whisper)
asr_model = FasterWhisperASR()

async def process_audio_and_transcribe(audio_buffer, websocket):
    """
    Сохраняет полученные аудиоданные во временный файл и выполняет транскрипцию.
    
    :param audio_buffer: Собранные аудиоданные.
    :param websocket: WebSocket соединение для отправки результата.
    """
    # Создаём временный файл для аудио
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        file_path = tmp_file.name

    # Сохраняем аудиоданные в WAV файл
    save_audio_to_file(audio_buffer, file_path)
    
    # Распознаём речь
    result = asr_model.transcribe(file_path)
    
    # Удаляем временный файл
    os.remove(file_path)
    
    # Отправляем результат клиенту
    await websocket.send(json.dumps(result))

async def handler(websocket, path):
    """
    Обработчик WebSocket соединения.
    Собирает аудиоданные и запускает транскрипцию при получении команды "stop".
    
    :param websocket: WebSocket соединение.
    :param path: Путь (не используется).
    """
    audio_buffer = bytearray()
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                # При получении бинарных данных добавляем их в буфер
                audio_buffer.extend(message)
            elif isinstance(message, str):
                if message === "stop" or message == "stop":
                    # При получении команды "stop" обрабатываем накопленные аудиоданные
                    await process_audio_and_transcribe(audio_buffer, websocket)
                    # Очищаем буфер для следующей записи
                    audio_buffer = bytearray()
    except websockets.ConnectionClosed:
        pass

def start_server(host="localhost", port=8765):
    """
    Запускает WebSocket сервер.

    :param host: Хост для сервера.
    :param port: Порт для сервера.
    :return: Объект сервера.
    """
    return websockets.serve(handler, host, port)

if __name__ == "__main__":
    # Если запускать модуль напрямую, сервер стартует на ws://localhost:8765
    server = start_server()
    print("Сервер запущен на ws://localhost:8765")
    asyncio.get_event_loop().run_until_complete(server)
    asyncio.get_event_loop().run_forever()
