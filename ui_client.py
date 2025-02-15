### ui_client.py
import os
import time
import json
import datetime
import requests
import librosa
import numpy as np
import pandas as pd
import gradio as gr
from dotenv import load_dotenv
from typing import Optional, Tuple

load_dotenv()

# -----------------------------------------------------------------------------
#                              CSS и тема оформления
# -----------------------------------------------------------------------------
custom_css = """
.transcription_display_container, .auto_translation_display_container {
    max-height: 500px;
    overflow-y: scroll;
}

footer {
    visibility: hidden;
}

label {
    font-size: 18px;
    font-weight: bold;
    color: #333;
    margin-bottom: 8px;
}
"""

# -----------------------------------------------------------------------------
#                                ФУНКЦИИ
# -----------------------------------------------------------------------------
def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Переводит текст с указанного языка на целевой язык, используя внешний API.

    Аргументы:
        text (str): Текст, который требуется перевести.
        source_lang (str): Код исходного языка (например, 'ru' для русского).
        target_lang (str): Код целевого языка (например, 'en' для английского).

    Возвращает:
        str: Результат перевода или сообщение об ошибке.
    """
    api_key = os.getenv("PROXY_API_KEY")
    url = "https://api.proxyapi.ru/google/v1/models/gemini-1.5-pro:generateContent"

    prompt = f"""
    Ты переводишь транскрипты записей речи. Переведи с {source_lang} на {target_lang}: 
    ```{text}```. 
    Постарайся не вносить свои смыслы в текст. В ответе только текст без доп символов.
    """
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


def tts_from_server(text: str, lang: str, save_path: str) -> None:
    """
    Обращается к серверу, который преобразует текст в речь, и сохраняет полученный аудиофайл.

    Аргументы:
        text (str): Текст, который необходимо озвучить.
        lang (str): Код языка (например, 'ru', 'en' и т.п.).
        save_path (str): Путь для сохранения аудиофайла.

    Возвращает:
        None
    """
    # Создаем папку для сохранения файла, если она не существует
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    url = "http://localhost:8002/stream-audio"
    data = {"text": text, "lang": lang}

    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при обращении к TTS-серверу: {e}")


def text_to_speech(text: str, language: str = "en") -> str:
    """
    Преобразует введённый текст в речь (через внешний сервис) и возвращает путь к сохранённому аудиофайлу.

    Аргументы:
        text (str): Текст, который необходимо преобразовать в речь.
        language (str, optional): Код языка. По умолчанию 'en'.

    Возвращает:
        str: Путь к загруженному аудиофайлу.
    """
    os.makedirs("tmp", exist_ok=True)
    save_path = "tmp/downloaded_audio.mp3"
    tts_from_server(text, language, save_path)
    return save_path


def send_audio_to_server(audio_data: np.ndarray, language_code: str = "") -> Tuple[str, str, float]:
    """
    Отправляет аудио на сервер для транскрипции и определения языка.

    Аргументы:
        audio_data (np.ndarray): Массив аудиоданных (моно, float32).
        language_code (str, optional): Код языка для подсказки модели (по умолчанию '').

    Возвращает:
        Tuple[str, str, float]:
            - str: Результат транскрипции.
            - str: Определённый язык.
            - float: Вероятность (0..1) определения языка.
    """
    audio_data_bytes = audio_data.astype(np.int16).tobytes()
    endpoint_url = "http://localhost:8000/predict"

    response = requests.post(
        endpoint_url,
        data=audio_data_bytes,
        params={"language_code": language_code},
        headers={
            "accept": "application/json",
            "Content-Type": "application/octet-stream"
        }
    )
    result = response.json()
    return result["prediction"], result["language"], result["language_probability"]


def _reset_button_click(
    stream_state: np.ndarray,
    latency_data_state: dict,
    transcription_history_state: list,
    current_transcription_state: str,
    auto_translation_history_state: list
) -> Tuple[None, None, list, str, str, str, str, str, list]:
    """
    Сбрасывает все состояния, связанные с транскрипцией и авто-переводом.

    Аргументы:
        stream_state (np.ndarray | None): Текущий аудиопоток.
        latency_data_state (dict | None): Данные о задержке.
        transcription_history_state (list): История транскрипций.
        current_transcription_state (str): Текущий текст транскрипции.
        auto_translation_history_state (list): История авто-переводов.

    Возвращает:
        Tuple: Кортеж с новыми значениями состояний.
    """
    return (None, None, [], "", "", "", "", "", [])


def apply_text_size(text: str, size: str) -> str:
    """
    Применяет указанный размер шрифта к тексту, оборачивая его в HTML.

    Аргументы:
        text (str): Исходный текст.
        size (str): Новый размер шрифта (например, '14px').

    Возвращает:
        str: HTML-строка, в которой текст представлен с новым размером шрифта.
    """
    return f'<div style="font-size: {size}; white-space: pre-wrap;">{text}</div>'


def dummy_function(
    stream: np.ndarray,
    new_chunk: Tuple[int, np.ndarray],
    max_length: int,
    overlap_duration: float,
    latency_data: dict,
    current_transcription: str,
    transcription_history: list,
    language_code: str,
    font_size: str,
    source_lang: str,
    target_lang: str,
    auto_translate_checked: bool,
    auto_translation_history: list
) -> Tuple[
    np.ndarray,
    str,
    str,
    dict,
    str,
    list,
    str,
    str,
    str,
    list
]:
    """
    Обрабатывает поток аудиоданных, выполняет транскрипцию с перекрытием между чанками,
    собирает статистику задержек, управляет историей транскрипций и выполняет авто-перевод.
    При авто-переводе переводится только та строка, которая только что была зафиксирована и ушла в историю.

    Аргументы:
        stream (np.ndarray | None): Текущий накопленный аудиопоток или None.
        new_chunk (Tuple[int, np.ndarray]): Кортеж вида (частота дискретизации, массив семплов).
        max_length (int): Максимальная длина аудио (сек) перед разбиением на части.
        overlap_duration (float): Длительность перекрытия (сек) между чанками.
        latency_data (dict | None): Данные о задержках.
        current_transcription (str): Текущий буфер транскрипции.
        transcription_history (list): История транскрипций.
        language_code (str): Код языка для подсказки модели.
        font_size (str): Размер шрифта для отображения.
        source_lang (str): Исходный язык для перевода.
        target_lang (str): Целевой язык для перевода.
        auto_translate_checked (bool): Флаг авто-перевода.
        auto_translation_history (list): История авто-переводов.

    Возвращает:
        Tuple: Обновленные состояния и HTML-отображения для транскрипции и авто-перевода.
    """
    start_time = time.time()

    # Инициализируем словарь задержек, если он не задан
    if latency_data is None:
        latency_data = {
            "total_resampling_latency": [],
            "total_transcription_latency": [],
            "total_latency": [],
        }

    sampling_rate, y = new_chunk
    y = y.astype(np.float32)

    # Добавляем новый фрагмент к текущему потоку или создаём новый
    if stream is not None:
        stream = np.concatenate([stream, y])
    else:
        stream = y

    transcription = "ERROR"
    language = "ERROR"
    language_pred = 0.0

    try:
        # Ресэмплинг аудио до 16000 Гц
        sampling_start_time = time.time()
        stream_resampled = librosa.resample(stream, orig_sr=sampling_rate, target_sr=16000)
        sampling_end_time = time.time()
        sampling_latency = sampling_end_time - sampling_start_time
        latency_data["total_resampling_latency"].append(sampling_latency)

        # Транскрипция аудио через сервер
        transcription_start_time = time.time()
        if isinstance(language_code, list):
            language_code = language_code[0] if len(language_code) > 0 else ""
        transcription, language, language_pred = send_audio_to_server(stream_resampled, str(language_code))
        current_transcription = f"{transcription}"
        transcription_end_time = time.time()
        transcription_latency = transcription_end_time - transcription_start_time
        latency_data["total_transcription_latency"].append(transcription_latency)
    except Exception as e:
        print(f"Ошибка при транскрипции: {e}")
        # В случае ошибки добавляем значения по умолчанию для выравнивания списков задержек
        latency_data["total_resampling_latency"].append(0.0)
        latency_data["total_transcription_latency"].append(0.0)
        transcription = "ERROR"
        language = "ERROR"
        language_pred = 0.0

    end_time = time.time()
    total_latency = end_time - start_time
    latency_data["total_latency"].append(total_latency)

    # Выравниваем длину всех списков в latency_data
    max_len = max(len(lst) for lst in latency_data.values())
    for key in latency_data:
        while len(latency_data[key]) < max_len:
            latency_data[key].append(0.0)

    # Если длина потока превышает max_length, сохраняем транскрипцию и оставляем перекрытие
    finalized_line = None
    if len(stream) > sampling_rate * max_length:
        finalized_line = current_transcription  # строка, которая только что ушла вниз
        transcription_history.append(current_transcription)
        overlap_samples = int(overlap_duration * sampling_rate)
        stream = stream[-overlap_samples:]
        current_transcription = ""

    # Формируем отображаемый текст транскрипции: текущий буфер + история (обратный порядок)
    display_text = f"{current_transcription}\n\n" + "\n\n".join(transcription_history[::-1])
    transcription_html = apply_text_size(display_text, font_size)

    # Подготавливаем таблицу статистики задержек (в миллисекундах)
    info_df = pd.DataFrame(latency_data)
    info_df = info_df.apply(lambda x: x * 1000)
    info_df = info_df.describe().loc[["min", "max", "mean"]]
    info_df = info_df.round(2)
    info_df = info_df.astype(str) + " ms"
    info_df = info_df.T
    info_df.index.name = ""
    info_df = info_df.reset_index()
    info_df_html = info_df.to_html(index=False)

    language_and_pred_text = f"Predicted Language: {language} ({language_pred * 100:.2f}%)"

    # Выполнение авто-перевода только для финализированной строки
    # (т.е. только если завершился один из чанков)
    if auto_translate_checked and finalized_line is not None:
        try:
            auto_translated_line = translate_text(finalized_line, language if not source_lang else source_lang, target_lang)
            auto_translation_history.append(auto_translated_line)
        except Exception as e:
            print(f"Auto translation error: {e}")

    # Формируем HTML для истории авто-перевода (с обратным порядком)
    auto_translation_display = "\n\n" + "\n".join(auto_translation_history[::-1])
    auto_translation_html = apply_text_size(auto_translation_display, font_size)

    # В данной функции ручной перевод не вычисляется – он вызывается отдельно
    manual_translation = ""

    return (
        stream,                      # обновлённый аудиопоток
        transcription_html,          # HTML транскрипции
        info_df_html,                # статистика задержек (HTML)
        latency_data,                # обновлённые данные задержек
        current_transcription,       # текущий буфер транскрипции
        transcription_history,       # история транскрипций
        language_and_pred_text,      # информация о языке
        manual_translation,          # ручной перевод (пустая строка здесь)
        auto_translation_html,       # HTML авто-перевода
        auto_translation_history     # обновлённая история авто-перевода
    )


# -----------------------------------------------------------------------------
#                            СОЗДАНИЕ ИНТЕРФЕЙСА
# -----------------------------------------------------------------------------
with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Default(spacing_size=gr.themes.sizes.spacing_sm, radius_size=gr.themes.sizes.radius_none)
) as demo:

    gr.Markdown("# Приложение для живой транскрипции, текста в речь и перевода")

    # Состояния
    stream_state = gr.State(None)
    latency_data_state = gr.State(None)
    transcription_history_state = gr.State([])
    current_transcription_state = gr.State("")
    auto_translation_history_state = gr.State([])

    # -------------------------------------------------------------------------
    #                           БЛОК ЖИВОЙ ТРАНСКРИПЦИИ
    # -------------------------------------------------------------------------
    with gr.Column():
        gr.Markdown("## Живая транскрипция")
        with gr.Accordion(label="Как использовать", open=False):
            gr.Markdown("""
**1.** Нажмите **Start** и разрешите доступ к микрофону.  
**2.** Выберите язык (или оставьте Auto detect).  
**3.** Настройте максимальную длину аудио и длительность перекрытия между чанками.  
**4.** Нажмите **Stop**, чтобы остановить запись.  
**5.** Чтобы очистить данные и начать заново, нажмите **Reset**.  
**6.** Для автоматического перевода новых строк установите галочку **Авто-перевод**.  
**7.** Для ручного перевода всего текста используйте блок справа.
            """)

        with gr.Row():
            mic_audio_input = gr.Audio(sources=["microphone"], streaming=True, label="Микрофон")
            language_code_input = gr.Dropdown(
                [
                    ("Auto detect", ""),
                    ("English", "en"),
                    ("Spanish", "es"),
                    ("German", "de"),
                    ("Russian", "ru")
                ],
                value="",
                label="Код языка",
                multiselect=False
            )
            max_length_input = gr.Slider(
                value=7,
                minimum=2,
                maximum=30,
                step=1,
                label="Длина одного чанка (сек)"
            )
            overlap_slider = gr.Slider(
                value=2.0,
                minimum=0.0,
                maximum=5.0,
                step=0.1,
                label="Перекрытие чанков (сек)"
            )
            reset_button = gr.Button("Reset")

        # Ряд, где слева отображается транскрипция, а справа – автоматический перевод
        with gr.Row():
            transcription_html = gr.HTML(label="Транскрипция", elem_id="transcription_display_container")
            auto_translation_html = gr.HTML(label="Автоматический перевод", elem_id="auto_translation_display_container")

        # Вывод информации о языке
        transcription_language_prod_output = gr.Text(
            lines=1,
            show_label=False,
            interactive=False
        )

        with gr.Row():
            auto_translate_checkbox = gr.Checkbox(label="Авто-перевод", value=False, interactive=True)
            font_size_dropdown = gr.Dropdown(
                [
                    "8px", "10px", "12px", "14px", "16px",
                    "18px", "20px", "24px", "36px", "48px"
                ],
                value="14px",
                label="Размер шрифта",
                multiselect=False
            )

        # Блок статистики задержек перемещён вниз
        with gr.Row():
            info_table_output = gr.HTML(label="Статистика задержек")

    # -------------------------------------------------------------------------
    #                           БЛОК ПЕРЕВОДА ТЕКСТА (РУЧНОЙ)
    # -------------------------------------------------------------------------
    with gr.Column():
        gr.Markdown("## Перевод текста")
        with gr.Row():
            source_lang = gr.Dropdown(
                [
                    ("Auto detect", ""),
                    ("English", "en"),
                    ("Spanish", "es"),
                    ("Italian", "it"),
                    ("German", "de"),
                    ("Hungarian", "hu"),
                    ("Russian", "ru")
                ],
                value="",
                label="Исходный язык",
                multiselect=False
            )
            target_lang = gr.Dropdown(
                [
                    ("English", "en"),
                    ("Spanish", "es"),
                    ("Italian", "it"),
                    ("German", "de"),
                    ("Hungarian", "hu"),
                    ("Russian", "ru")
                ],
                value="en",
                label="Целевой язык",
                multiselect=False
            )
        translate_button = gr.Button("Перевести")
        output_text = gr.Textbox(
            lines=5,
            label="Переведённый текст (ручной перевод)",
            interactive=False
        )

        # Кнопка перевода вручную переводит весь текст из окна транскрибации
        translate_button.click(
            translate_text,
            inputs=[transcription_html, source_lang, target_lang],
            outputs=output_text
        )

    # -------------------------------------------------------------------------
    #                           БЛОК ТЕКСТ В РЕЧЬ
    # -------------------------------------------------------------------------
    with gr.Column():
        gr.Markdown("## Текст в речь")
        with gr.Row():
            tts_text_box = gr.Textbox(label="Текст")
            tts_lanuage_code = gr.Dropdown(
                [
                    ("Russian", "ru"),
                    ("English", "en"),
                ],
                value="ru",
                label="Язык для TTS",
                multiselect=False
            )

        load_audio_button = gr.Button("Преобразовать текст в речь")
        loaded_audio_display = gr.Audio(label="Сгенерированный аудиофайл", interactive=False)

        # Кнопка преобразования текста в речь
        load_audio_button.click(
            text_to_speech,
            inputs=[tts_text_box, tts_lanuage_code],
            outputs=[loaded_audio_display]
        )

    # -------------------------------------------------------------------------
    #                      СВЯЗИ МЕЖДУ КОМПОНЕНТАМИ
    # -------------------------------------------------------------------------
    # Потоковая функция для записи/транскрипции с авто-переводом
    mic_audio_input.stream(
        fn=dummy_function,
        inputs=[
            stream_state,              # текущий аудиопоток
            mic_audio_input,           # новые аудиоданные
            max_length_input,          # длина чанка (сек)
            overlap_slider,            # перекрытие (сек)
            latency_data_state,        # данные задержек
            current_transcription_state,
            transcription_history_state,
            language_code_input,
            font_size_dropdown,
            source_lang,               # исходный язык для перевода
            target_lang,               # целевой язык для перевода
            auto_translate_checkbox,   # авто-перевод включён/выключён
            auto_translation_history_state  # история авто-перевода
        ],
        outputs=[
            stream_state,                      # обновлённый аудиопоток
            transcription_html,                # HTML транскрипции
            info_table_output,                 # статистика задержек (HTML)
            latency_data_state,                # обновлённые данные задержек
            current_transcription_state,       # текущий буфер транскрипции
            transcription_history_state,       # история транскрипций
            transcription_language_prod_output,# информация о языке
            output_text,                       # ручной перевод (оставляем пустым в потоке)
            auto_translation_html,             # HTML авто-перевода
            auto_translation_history_state     # обновлённая история авто-перевода
        ],
        show_progress="hidden"
    )

    # Привязка кнопки для сброса состояний (сбросим и историю авто-перевода)
    reset_button.click(
        _reset_button_click,
        inputs=[
            stream_state,
            latency_data_state,
            transcription_history_state,
            current_transcription_state,
            auto_translation_history_state
        ],
        outputs=[
            stream_state,
            latency_data_state,
            transcription_history_state,
            current_transcription_state,
            transcription_language_prod_output,
            transcription_html,
            output_text,
            auto_translation_html,
            auto_translation_history_state
        ]
    )


# -----------------------------------------------------------------------------
#                       ЗАПУСК GRADIO-ПРИЛОЖЕНИЯ
# -----------------------------------------------------------------------------
SSL_CERT_PATH: Optional[str] = os.environ.get("SSL_CERT_PATH", None)
SSL_KEY_PATH: Optional[str] = os.environ.get("SSL_KEY_PATH", None)
SSL_VERIFY: bool = bool(os.environ.get("SSL_VERIFY", False))
SHARE: bool = True

demo.launch(
    server_name="0.0.0.0",
    server_port=5656,
    ssl_certfile=SSL_CERT_PATH,
    ssl_keyfile=SSL_KEY_PATH,
    ssl_verify=SSL_VERIFY,
    share=SHARE
)
