import queue
import re
import sys
import threading
import time
from typing import Dict, List

import numpy as np
import pyaudio
import requests

# Настройки аудио
STEP_IN_SEC: int = 1    # Интервал, через который добавляем аудио для обработки
LENGHT_IN_SEC: int = 6  # Максимальная длина аудио (в секундах) для обработки за раз
NB_CHANNELS = 1
RATE = 16000
CHUNK = RATE

# Для визуализации (максимальное число символов для фрагмента)
MAX_SENTENCE_CHARACTERS = 80

# ВНИМАНИЕ: Укажите корректный URL работающего сервера транскрипции!
TRANSCRIPTION_API_ENDPOINT = "http://localhost:7860/predict"

# This queue holds all the 1-second audio chunks
audio_queue = queue.Queue()

# This queue holds all the chunks that will be processed together
# If the chunk is filled to the max, it will be emptied
length_queue = queue.Queue(maxsize=LENGHT_IN_SEC)


def send_audio_to_server(audio_data: bytes) -> str:
    """
    Отправляет аудиоданные на сервер транскрипции и возвращает транскрипцию.

    Аргументы:
        audio_data (bytes): Аудиоданные в формате байтов.

    Возвращает:
        str: Транскрибированный текст.

    Исключения:
        Exception: Если сервер вернул статус, отличный от 200, или произошла ошибка декодирования JSON.
    """
    response = requests.post(
        TRANSCRIPTION_API_ENDPOINT,
        data=audio_data,
        headers={'Content-Type': 'application/octet-stream'}
    )
    
    if response.status_code != 200:
        raise Exception(
            f"Ошибка запроса: получен статус код {response.status_code}. "
            "Проверьте, что сервер транскрипции запущен и поддерживает POST-запросы."
        )
    
    try:
        result = response.json()
    except Exception as e:
        raise Exception(f"Ошибка декодирования JSON: {e}. Ответ сервера: {response.text}")

    return result["prediction"]


def producer_thread():
    """
    Запускает запись аудио с микрофона и помещает аудиофрагменты в очередь audio_queue.
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=NB_CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,  # 1 секунда аудио
    )

    print("-" * 80)
    print("Микрофон инициализирован, запись началась...")
    print("-" * 80)
    print("ТРАНСКРИПЦИЯ")
    print("-" * 80)

    while True:
        audio_data = b""
        for _ in range(STEP_IN_SEC):
            chunk = stream.read(RATE)
            audio_data += chunk

        audio_queue.put(audio_data)


# Thread which gets items from the queue and prints its length
def consumer_thread(stats):
    """
    Извлекает аудиоданные из очереди, собирает их в пакет для обработки, отправляет на сервер
    транскрипции и выводит результат в консоль. Также собирает статистику задержек.

    Аргументы:
        stats (dict): Словарь для накопления статистики времени.
    """
    while True:
        if length_queue.qsize() >= LENGHT_IN_SEC:
            with length_queue.mutex:
                length_queue.queue.clear()
                print()

        audio_data = audio_queue.get()
        transcription_start_time = time.time()
        length_queue.put(audio_data)

        # Объединяем аудиофрагменты из очереди
        audio_data_to_process = b""
        for i in range(length_queue.qsize()):
            audio_data_to_process += length_queue.queue[i]

        try:
            transcription = send_audio_to_server(audio_data_to_process)
            transcription = re.sub(r"\[.*\]", "", transcription)
            transcription = re.sub(r"\(.*\)", "", transcription)
        except Exception as e:
            transcription = f"Error: {e}"

        transcription_end_time = time.time()

        transcription_to_visualize = transcription.ljust(MAX_SENTENCE_CHARACTERS, " ")

        sys.stdout.write('\033[K' + transcription_to_visualize + '\r')

        audio_queue.task_done()

        overall_elapsed_time = time.time() - transcription_start_time
        transcription_elapsed_time = transcription_end_time - transcription_start_time
        postprocessing_elapsed_time = time.time() - transcription_end_time
        stats["overall"].append(overall_elapsed_time)
        stats["transcription"].append(transcription_elapsed_time)
        stats["postprocessing"].append(postprocessing_elapsed_time)


if __name__ == "__main__":
    stats: Dict[str, List[float]] = {"overall": [], "transcription": [], "postprocessing": []}

    producer = threading.Thread(target=producer_thread)
    producer.start()

    consumer = threading.Thread(target=consumer_thread, args=(stats,))
    consumer.start()

    try:
        producer.join()
        consumer.join()
    except KeyboardInterrupt:
        print("Завершение работы...")
        print("Количество обработанных фрагментов:", len(stats["overall"]))
        print(f"Общее время: avg: {np.mean(stats['overall']):.4f}s, std: {np.std(stats['overall']):.4f}s")
        print(
            f"Время транскрипции: avg: {np.mean(stats['transcription']):.4f}s, std: {np.std(stats['transcription']):.4f}s"
        )
        print(
            f"Время постобработки: avg: {np.mean(stats['postprocessing']):.4f}s, std: {np.std(stats['postprocessing']):.4f}s"
        )
        print(f"Средняя задержка: {np.mean(stats['overall']) + STEP_IN_SEC:.4f}s")