import gradio as gr
import requests
import json
import os
import time
import datetime
import librosa
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def translate_text(text, source_lang, target_lang):
    """
    Переводит текст с указанного исходного языка на целевой язык с использованием внешнего API.

    :param text: Текст для перевода.
    :param source_lang: Код исходного языка (например, "ru").
    :param target_lang: Код целевого языка (например, "en").
    :return: Переведённый текст или сообщение об ошибке.
    """
    api_key = os.getenv("PROXY_API_KEY")
    url = "https://api.proxyapi.ru/google/v1/models/gemini-1.5-pro:generateContent"

    prompt = f"Переведи с {source_lang} на {target_lang}: {text}. В ответе только текст без доп символов."

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 150}
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Ошибка {response.status_code}: {response.text}"


def translate_interface(text, source_lang, target_lang):
    """
    Интерфейс для вызова функции перевода текста.

    :param text: Текст для перевода.
    :param source_lang: Код исходного языка.
    :param target_lang: Код целевого языка.
    :return: Переведённый текст.
    """
    return translate_text(text, source_lang, target_lang)


def tts_from_server(text: str, lang: str, save_path: str):
    """
    Загружает аудио с сервера, преобразующее текст в речь, и сохраняет его в указанный файл.

    :param text: Текст для преобразования в речь.
    :param lang: Код языка для озвучивания.
    :param save_path: Путь для сохранения аудиофайла.
    """
    url = "http://176.114.66.227:8000/stream-audio"
    data = {"text": text, "lang": lang}
    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Ошибка: {e}")


def text_to_speech(text, language='english'):
    """
    Преобразует введённый текст в речь, сохраняя аудиофайл локально.

    :param text: Текст для преобразования.
    :param language: Код языка.
    :return: Путь к загруженному аудиофайлу.
    """
    save_path = "tmp/downloaded_audio.mp3"
    tts_from_server(text, language, save_path)
    return save_path


def send_audio_to_server(audio_data: np.ndarray, language_code: str = "") -> Tuple[str, str, float]:
    """
    Отправляет аудио данные на сервер для получения транскрипции и определения языка.

    :param audio_data: Аудиоданные в виде numpy массива.
    :param language_code: Код языка (может быть пустым для автоопределения).
    :return: Кортеж из транскрипции, определённого языка и вероятности определения языка.
    """
    audio_data_bytes = audio_data.astype(np.int16).tobytes()
    response = requests.post(
        "http://localhost:8000/predict",
        data=audio_data_bytes,
        params={"language_code": language_code},
        headers={"accept": "application/json", "Content-Type": "application/octet-stream"}
    )
    result = response.json()
    return result["prediction"], result["language"], result["language_probability"]


def dummy_function(stream, new_chunk, max_length, latency_data, current_transcription, transcription_history, language_code):
    """
    Обрабатывает поток аудио данных, выполняет транскрипцию и обновляет историю транскрипций.

    :param stream: Текущий накопленный аудио поток.
    :param new_chunk: Новый кусок аудио данных в формате (sampling_rate, numpy array).
    :param max_length: Максимальная длина аудио в секундах для одного сегмента.
    :param latency_data: Данные о задержке (не используются в данной реализации).
    :param current_transcription: Текущая транскрипция.
    :param transcription_history: История предыдущих транскрипций.
    :param language_code: Выбранный код языка.
    :return: Обновлённые состояния: stream, текст для отображения транскрипции, latency_data,
             current_transcription, transcription_history и информация о предсказанном языке.
    """
    start_time = time.time()
    sampling_rate, y = new_chunk
    y = y.astype(np.float32)
    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    transcription = "ERROR"
    language = "ERROR"
    language_pred = 0.0
    try:
        # Преобразуем аудио поток к требуемой частоте дискретизации (16000 Гц)
        stream_resampled = librosa.resample(stream, orig_sr=sampling_rate, target_sr=16000)
        transcription, language, language_pred = send_audio_to_server(stream_resampled, str(language_code))
        current_transcription = f"{transcription}"
    except Exception as e:
        print(f"Error: {e}")

    end_time = time.time()
    # Если длина потока превышает максимальную, сбрасываем поток и сохраняем транскрипцию в историю
    if len(stream) > sampling_rate * max_length:
        stream = None
        transcription_history.append(current_transcription)
        current_transcription = ""

    display_text = f"{current_transcription}\n\n" + "\n\n".join(transcription_history[::-1])
    return stream, display_text, latency_data, current_transcription, transcription_history, f"Predicted Language: {language} ({language_pred * 100:.2f}%)"


def _reset_button_click(stream_state, transcription_display, latency_data_state, transcription_history_state, current_transcription_state):
    """
    Сбрасывает состояния транскрипции при нажатии кнопки Reset.

    :return: Начальные значения состояний.
    """
    return None, "", None, [], "", ""


# Создание интерфейса Gradio с двумя основными блоками: транскрипция (и TTS) слева и перевод справа
with gr.Blocks() as demo:
    gr.Markdown("# Приложение для живой транскрипции, текст в речь и перевода")

    # Определяем состояния для хранения данных транскрипции
    stream_state = gr.State(None)
    latency_data_state = gr.State(None)
    transcription_history_state = gr.State([])
    current_transcription_state = gr.State("")

    # Интерфейс разделён на два столбца
    with gr.Row():
        # Левый столбец: Живая транскрипция и текст в речь
        with gr.Column():
            gr.Markdown("## Живая транскрипция")
            with gr.Accordion(label="Как использовать", open=False):
                gr.Markdown("""
### Инструкция по использованию сервиса живой транскрипции:
1. **Нажмите 'Start'** и разрешите доступ к микрофону.
2. **Выберите язык** (или оставьте автоопределение).
3. **Нажмите 'Stop'** для завершения и 'Reset' для сброса.
4. **Используйте перевод** во втором столбце, выбрав целевой язык.
""")
            with gr.Row():
                mic_audio_input = gr.Audio(sources=["microphone"], streaming=True, label="Микрофон")
                reset_button = gr.Button("Reset")
                max_length_input = gr.Slider(value=10, minimum=2, maximum=30, step=1, label="Максимальная длина аудио (сек)")
                language_code_input = gr.Dropdown(
                    [("Auto detect", ""), ("English", "en"), ("Spanish", "es"),
                     ("Italian", "it"), ("German", "de"), ("Hungarian", "hu"), 
                     ("Russian", "ru")],
                    value="", label="Код языка", multiselect=False
                )
            gr.Markdown("### Транскрипция")
            transcription_language_prod_output = gr.Text(lines=1, show_label=False, interactive=False)
            transcription_display = gr.Textbox(lines=5, show_label=False, interactive=False, show_copy_button=True)

            gr.Markdown("## Текст в речь")
            load_audio_button = gr.Button("Преобразовать текст в речь")
            with gr.Row():
                tts_text_box = gr.Textbox(label="Текст")
                tts_lanuage_code = gr.Dropdown(
                    [("Auto detect", ""), ("English", "en"), ("Spanish", "es"),
                     ("Italian", "it"), ("German", "de"), ("Hungarian", "hu"), 
                     ("Russian", "ru")],
                    value="", label="Код языка", multiselect=False
                )
            loaded_audio_display = gr.Audio(label="Аудиофайл", interactive=False)

        # Правый столбец: Перевод текста
        with gr.Column():
            gr.Markdown("## Перевод текста")
            text_input = gr.Textbox(label="Введите текст")
            source_lang = gr.Textbox(label="Исходный язык (например, ru)")
            target_lang = gr.Textbox(label="Целевой язык (например, en)")
            translate_button = gr.Button("Перевести")
            output_text = gr.Textbox(label="Переведённый текст")

            # Привязка кнопки перевода к функции перевода
            translate_button.click(translate_interface,
                                   inputs=[text_input, source_lang, target_lang],
                                   outputs=output_text)

    # Привязка аудио потока к функции обработки транскрипции
    mic_audio_input.stream(
        dummy_function,
        inputs=[stream_state, mic_audio_input, max_length_input, latency_data_state,
                current_transcription_state, transcription_history_state, language_code_input],
        outputs=[stream_state, transcription_display, latency_data_state, current_transcription_state,
                 transcription_history_state, transcription_language_prod_output],
        show_progress="hidden"
    )

    # Привязка кнопки Reset к функции сброса состояний
    reset_button.click(
        _reset_button_click,
        inputs=[stream_state, transcription_display, latency_data_state, transcription_history_state, current_transcription_state],
        outputs=[stream_state, transcription_display, latency_data_state, transcription_history_state, current_transcription_state, transcription_language_prod_output]
    )

    # Привязка кнопки преобразования текста в речь к соответствующей функции
    load_audio_button.click(text_to_speech,
                            inputs=[tts_text_box, tts_lanuage_code],
                            outputs=[loaded_audio_display])

# Параметры запуска сервера
SSL_CERT_PATH: Optional[str] = os.environ.get("SSL_CERT_PATH", None)
SSL_KEY_PATH: Optional[str] = os.environ.get("SSL_KEY_PATH", None)
SSL_VERIFY: bool = bool(os.environ.get("SSL_VERIFY", False))
SHARE: bool = True

demo.launch(server_name="0.0.0.0",
            server_port=5656,
            ssl_certfile=SSL_CERT_PATH,
            ssl_keyfile=SSL_KEY_PATH,
            ssl_verify=SSL_VERIFY,
            share=SHARE)
