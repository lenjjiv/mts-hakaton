# server.py
import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
import wave

import websockets
from faster_whisper import WhisperModel

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class AudioTranscriptionServer:
    """
    Сервер для транскрибации аудио через WebSocket соединение.
    Использует faster-whisper для распознавания речи в реальном времени.
    """
    
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.connected_clients = {}
        
        # Инициализация модели Whisper
        logger.info("Инициализация модели Whisper...")
        self.model = WhisperModel("large-v3", device="cuda", compute_type="float16")
        logger.info("Модель Whisper загружена успешно")

    async def save_audio(self, audio_data, client_id):
        """Сохраняет аудио во временный WAV файл"""
        filename = f"temp_{client_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Моно
            wav_file.setsampwidth(2)  # 16 бит
            wav_file.setframerate(16000)  # 16кГц
            wav_file.writeframes(audio_data)
        
        return filename

    async def transcribe_audio(self, filename):
        """Транскрибация аудио с помощью Whisper"""
        logger.info(f"Начало транскрибации файла {filename}")
        segments, info = self.model.transcribe(filename, beam_size=5)
        segments = list(segments)  # Запуск транскрибации
        
        text = " ".join([s.text for s in segments])
        logger.info(f"Транскрибация завершена: {text[:50]}...")
        
        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability
        }

    async def handle_client(self, websocket, client_id):
        """Обработка подключения клиента"""
        try:
            while True:
                message = await websocket.recv()
                
                if isinstance(message, bytes):
                    # Сохраняем аудио
                    filename = await self.save_audio(message, client_id)
                    logger.info(f"Сохранено аудио: {filename}")
                    
                    # Транскрибируем
                    result = await self.transcribe_audio(filename)
                    
                    # Отправляем результат
                    await websocket.send(json.dumps(result))
                    
                    # Удаляем временный файл
                    os.remove(filename)
                    logger.info(f"Удален временный файл: {filename}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Клиент отключился: {client_id}")
        except Exception as e:
            logger.error(f"Ошибка при обработке клиента {client_id}: {str(e)}")

    async def handle_connection(self, websocket, path):
        """Обработка нового подключения"""
        client_id = str(uuid.uuid4())
        logger.info(f"Новое подключение: {client_id}")
        
        self.connected_clients[client_id] = websocket
        try:
            await self.handle_client(websocket, client_id)
        finally:
            del self.connected_clients[client_id]

    async def start(self):
        """Запуск сервера"""
        server = await websockets.serve(self.handle_connection, self.host, self.port)
        logger.info(f"Сервер запущен на ws://{self.host}:{self.port}")
        return server

# main.py
if __name__ == "__main__":
    server = AudioTranscriptionServer()
    asyncio.get_event_loop().run_until_complete(server.start())
    asyncio.get_event_loop().run_forever()