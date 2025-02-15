import datetime
import os
import time
from typing import Optional, Tuple

import gradio as gr
import librosa
import numpy as np
import pandas as pd
import requests

STEPS_IN_SEC: int = 1  # Частота отправки аудио (в секундах)
LENGHT_IN_SEC: int = 6  # Максимальная длина аудиофрагмента для обработки

# ВНИМАНИЕ: Замените URL на корректный адрес работающего сервера транскрипции!
TRANSCRIPTION_API_ENDPOINT = "http://localhost:8000/predict"


def send_audio_to_server(
        audio_data: np.ndarray,
        language_code: str = ""
    ) -> Tuple[str, str, float]:
    """
    Отправляет аудиоданные на сервер транскрипции и возвращает результаты.

    Аргументы:
        audio_data (np.ndarray): Массив аудиоданных (будет приведён к типу int16 и отправлен как байты).
        language_code (str): Код языка для транскрипции (по умолчанию пустая строка, что означает автоопределение).

    Возвращает:
        tuple: Кортеж из трех значений:
            - prediction (str): Полученная транскрипция или "ERROR" в случае ошибки.
            - language (str): Определённый язык или "ERROR" в случае ошибки.
            - language_probability (float): Вероятность определения языка или 0.0 в случае ошибки.
    """
    audio_data_bytes = audio_data.astype(np.int16).tobytes()
    try:
        response = requests.post(
            TRANSCRIPTION_API_ENDPOINT,
            data=audio_data_bytes,
            params={"language_code": language_code},
            headers={
                "accept": "application/json",
                "Content-Type": "application/octet-stream"
            },
            timeout=10  # Добавлен таймаут на случай зависания запроса
        )
        response.raise_for_status()  # Генерирует исключение для кода ответа отличного от 200
        result = response.json()
    except Exception as e:
        print(f"[*] Ошибка при отправке аудио на сервер транскрипции: {e}")
        return "ERROR", "ERROR", 0.0

    return (
        result.get("prediction", "ERROR"),
        result.get("language", "ERROR"),
        result.get("language_probability", 0.0)
    )


def process_audio_chunk(
    stream,
    new_chunk,
    max_length,
    latency_data,
    current_transcription,
    transcription_history,
    language_code
):
    """
    Обрабатывает новый аудиофрагмент: выполняет ресэмплинг, отправляет данные на сервер транскрипции,
    обновляет историю транскрипций и собирает статистику задержек.

    Аргументы:
        stream (np.ndarray или None): Накопленный аудиопоток.
        new_chunk (tuple): (sampling_rate, y) – данные нового аудиофрагмента.
        max_length (int): Максимальная длина аудио для обработки.
        latency_data (dict): Словарь для накопления данных задержек.
        current_transcription (str): Текущая транскрипция.
        transcription_history (list): История предыдущих транскрипций.
        language_code (str или list): Код языка (или список с кодами).

    Возвращает:
        tuple: Обновлённые состояния:
            - stream: обновлённый аудиопоток.
            - display_text: текст для отображения транскрипции.
            - info_df: DataFrame с данными о задержках.
            - latency_data: обновлённый словарь с данными о задержках.
            - current_transcription: обновлённая текущая транскрипция.
            - transcription_history: обновлённая история транскрипций.
            - language_and_pred_text: строка с информацией о предсказанном языке.
    """
    import time
    import librosa
    import pandas as pd
    import re

    start_time = time.time()

    if latency_data is None:
        latency_data = {
            "total_resampling_latency": [],
            "total_transcription_latency": [],
            "total_latency": [],
        }

    sampling_rate, y = new_chunk
    y = y.astype("float32")

    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    # Если накоплено менее 1 секунды аудио – пропускаем обработку
    if len(stream) < sampling_rate:
        return (
            stream,
            current_transcription,
            None,
            latency_data,
            current_transcription,
            transcription_history,
            ""
        )

    # --- РЕСЭМПЛИНГ ---
    try:
        resample_start = time.time()
        stream_resampled = librosa.resample(stream, orig_sr=sampling_rate, target_sr=16000)
    except Exception as e:
        print("[*] Ошибка при ресэмплинге:", e)
        stream_resampled = stream
        resample_latency = 0.0
    else:
        resample_latency = time.time() - resample_start
    finally:
        latency_data["total_resampling_latency"].append(resample_latency)

    # --- ТРАНСКРИПЦИЯ ---
    transcription_start = time.time()
    try:
        if isinstance(language_code, list):
            language_code = language_code[0] if language_code else ""
        transcription, language, language_pred = send_audio_to_server(
            stream_resampled, str(language_code)
        )
        transcription = re.sub(r"\[.*?\]", "", transcription)
        transcription = re.sub(r"\(.*?\)", "", transcription)
        current_transcription = transcription
    except Exception as e:
        print("[*] Ошибка при транскрипции:", e)
        transcription = "ERROR"
        language = "ERROR"
        language_pred = 0.0
        current_transcription = ""
    finally:
        transcription_latency = time.time() - transcription_start
        latency_data["total_transcription_latency"].append(transcription_latency)

    total_latency = time.time() - start_time
    latency_data["total_latency"].append(total_latency)

    if len(stream) > sampling_rate * max_length:
        stream = None
        transcription_history.append(current_transcription)
        current_transcription = ""

    display_text = f"{current_transcription}\n\n" + "\n\n".join(transcription_history[::-1])

    info_df = pd.DataFrame(latency_data)
    info_df = info_df.apply(lambda x: x * 1000)
    info_df = info_df.describe().loc[["min", "max", "mean"]]
    info_df = info_df.round(2)
    info_df = info_df.astype(str) + " ms"
    info_df = info_df.T
    info_df.index.name = ""
    info_df = info_df.reset_index()

    language_and_pred_text = f"Predicted Language: {language} ({language_pred * 100:.2f}%)"

    return (
        stream,
        display_text,
        info_df,
        latency_data,
        current_transcription,
        transcription_history,
        language_and_pred_text
    )


custom_css = """
.transcription_display_container {max-height: 500px; overflow-y: scroll}
footer {visibility: hidden}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Live Transcription PoC\n\n")
    nb_visitors_output = gr.Text(
        f"Page visits: {-1}",
        interactive=False,
        show_label=False
    )

    # Stores the audio data that we'll process
    stream_state = gr.State(None)
    # Stores the latency data
    latency_data_state = gr.State(None)
    # Stores the transcription history that we can visualize
    transcription_history_state = gr.State([])
    # Stores the current transcription (as we process the audio in chunks, and we have a maximum lengtht of audio that we process together)
    current_transcription_state = gr.State("")

    with gr.Column():
        gr.Markdown("## Controls")
        with gr.Accordion(label="How to use it", open=False):
            gr.Markdown("""
### How to Use the Live Transcription Service

1. **Starting the Transcription**: 
   - Click on the `Start` button to begin. 
   - Make sure to allow microphone access in your browser (Chrome/Firefox).

2. **Language Selection**: 
   - Choose the desired language from the language code options. 
   - If you don't select a language, the system will attempt to auto-detect it based on the audio input.

3. **Stopping and Resetting**: 
   - If you need to stop the transcription, click on `Stop`. 
   - After stopping, it's important to click `Reset` to clear the existing data for a fresh start.

4. **Audio Data Length**: 
   - The `Max length of audio` setting determines the maximum amount of audio data processed at a time.
   - Upon reaching this limit, the system will automatically reset the audio data, indicated by the start of a new line in the transcription. This is visualized below.
5. **Transcribe and Translate**:
   - Set the language code to the target language to translate the transcription.
   - Speak in any other language and you'll see the translated text in the output.
   - This is a **very hacky way to do translation** (and it's not recommended), but it's a good demonstration of the power of Whisper

```
- max_length_of_audio = 4

$t$ is the current time, and in this example 1 step is 1 second

------------------------------------------
1st second: [t,   0,   0,   0] --> "Hi"
2nd second: [t-1, t,   0,   0] --> "Hi I am"
3rd second: [t-2, t-1, t,   0] --> "Hi I am the one"
4th second: [t-3, t-2, t-1, t] --> "Hi I am the one and only Gabor"
5th second: [t,   0,   0,   0] --> "How" --> Here we started the process again as the limit was reached, and the output is in a new line
6th second: [t-1, t,   0,   0] --> "How are"
7th second: [t-2, t-1, t,   0] --> "How are you"
etc...
------------------------------------------
```
""")
        with gr.Row():
            mic_audio_input = gr.Audio(sources=["microphone"], streaming=True)
            reset_button = gr.Button("Reset")
            max_length_input = gr.Slider(value=10,
                                         minimum=2,
                                         maximum=30,
                                         step=1,
                                         label="Max length of audio (sec)")
            language_code_input = gr.Dropdown([("Auto detect", ""),
                                               ("English", "en"),
                                               ("Spanish", "es"),
                                               ("Italian", "it"),
                                               ("German", "de"),
                                               ("Hungarian", "hu"),
                                               ("Russian", "ru")],
                                              value="",
                                              label="Language code",
                                              multiselect=False)

    gr.Markdown(
        "-------\n\n## Transcription\n\n(audio is sent to the server each second)\n\n"
    )
    transcription_language_prod_output = gr.Text(lines=1,
                                                 show_label=False,
                                                 interactive=False)
    transcription_display = gr.Textbox(lines=10,
                                       show_label=False,
                                       interactive=False,
                                       show_copy_button=True)

    gr.Markdown(
        "------\n\n## Statistics\n\nThese are just rough estimates, as the latency can vary a lot based on where are the servers located, resampling is required, etc."
    )

    # information_table_outout = gr.Markdown("(Info about latency will be shown here)")
    information_table_outout = gr.DataFrame(interactive=False,
                                            show_label=False)

    # In gradio the default samplign rate is 48000 (https://github.com/gradio-app/gradio/issues/6526)
    # and the chunks size varies between 24000 and 48000 - so between 0.5sec and 1 sec
    mic_audio_input.stream(
        process_audio_chunk, [
        stream_state, 
        mic_audio_input, 
        max_length_input, 
        latency_data_state,
        current_transcription_state, 
        transcription_history_state,
        language_code_input
    ], [
        stream_state, transcription_display, information_table_outout,
        latency_data_state, current_transcription_state,
        transcription_history_state, transcription_language_prod_output
    ],
                           show_progress="hidden")

    def reset_transcription_session(
        stream_state,
        transcription_display,
        information_table_output,
        latency_data_state,
        transcription_history_state,
        current_transcription_state
    ):
        """
        Сбрасывает состояние сессии транскрипции, очищая аудиопоток, историю транскрипций и данные задержек.
        
        Аргументы:
            stream_state: Текущее состояние аудиопотока.
            transcription_display: Отображаемый текст транскрипции.
            information_table_output: Таблица с информацией о задержках.
            latency_data_state: Состояние данных о задержках.
            transcription_history_state: История транскрипций.
            current_transcription_state: Текущая транскрипция.
        
        Возвращает:
            tuple: Обновлённое состояние с обнулёнными значениями и пустой строкой для информации о языке.
        """
        stream_state = None
        transcription_display = ""
        information_table_output = None
        latency_data_state = None
        transcription_history_state = []
        current_transcription_state = ""
        language_and_pred_text = ""
        return (
            stream_state,
            transcription_display,
            information_table_output,
            latency_data_state,
            transcription_history_state,
            current_transcription_state,
            language_and_pred_text
        )

    def on_page_load(request: gr.Request):
        """
        Обрабатывает загрузку страницы, обновляет счетчик посещений и возвращает статистику.
        
        Аргументы:
            request (gr.Request): Объект запроса от Gradio.
        
        Возвращает:
            str: Информация о количестве посещений и уникальных посетителях.
        """
        import datetime

        params = request.query_params
        user_ip = request.client.host

        try:
            with open("visits.csv", "r") as f:
                last_line = f.readlines()[-1]
                last_number = int(last_line.split(",")[0])
        except Exception as e:
            print("[*] Ошибка при чтении файла с посещениями:", e)
            last_number = 0

        with open("visits.csv", "a") as f:
            f.write(f"{last_number + 1},{datetime.datetime.now()},{user_ip},visited\n")

        try:
            with open("visits.csv", "r") as f:
                nb_unique_visitors = len(set(line.split(",")[2] for line in f.readlines()))
        except Exception as e:
            nb_unique_visitors = 0

        return f"Page visits: {last_number + 1} / Unique visitors: {nb_unique_visitors}"

    demo.load(on_page_load, [], [nb_visitors_output])

SSL_CERT_PATH: Optional[str] = os.environ.get("SSL_CERT_PATH", None)
SSL_KEY_PATH: Optional[str] = os.environ.get("SSL_KEY_PATH", None)
SSL_VERIFY: bool = bool(os.environ.get("SSL_VERIFY", False))
SHARE: bool = True

print(
    f"Settings: SSL_CERT_PATH={SSL_CERT_PATH}, SSL_KEY_PATH={SSL_KEY_PATH}, SSL_VERIFY={SSL_VERIFY}, SHARE={SHARE}"
)

if SHARE:
    print("[*] Running in share mode")
    ssl_certfile_path = None
    ssl_keyfile_path = None

else:
    if SSL_CERT_PATH is not None and SSL_KEY_PATH is not None:
        print("[*] Running in SSL mode")
        ssl_certfile_path = SSL_CERT_PATH
        ssl_keyfile_path = SSL_KEY_PATH
    else:
        print("[*] Running in non-SSL mode")
        ssl_certfile_path = None
        ssl_keyfile_path = None

demo.launch(
    server_name="0.0.0.0",
    server_port=5656,
    ssl_certfile=ssl_certfile_path,
    ssl_keyfile=ssl_keyfile_path,
    ssl_verify=SSL_VERIFY,
    share=SHARE
)
