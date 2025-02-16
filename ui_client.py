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


def tts_from_server(text: str, lang: str, save_path: str, endpoint_url="http://localhost:8002/stream-audio") -> None:
    """
    Обращается к серверу, который преобразует текст в речь, и сохраняет полученный аудиофайл.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data = {"text": text, "lang": lang}

    try:
        response = requests.post(endpoint_url, json=data, stream=True)
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
    """
    temp_dir = "tmp_client"
    os.makedirs(temp_dir, exist_ok=True)
    save_path = os.path.join(temp_dir, "downloaded_audio.mp3")
    tts_from_server(text, language, save_path)
    return save_path


def send_audio_to_server(audio_data: np.ndarray, language_code: str = "") -> Tuple[str, str, float]:
    """
    Отправляет аудио на сервер для транскрипции и определения языка.
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
    auto_translation_history_state: list,
    last_finalize_time_state: float
) -> Tuple[None, str, str, None, str, list, str, str, str, list, None]:
    """
    Сбрасывает все состояния, включая новое состояние времени последней финализации.
    """
    return (None, "", "", None, "", [], "", "", "", [], None)


def apply_text_size(text: str, size: str) -> str:
    """
    Применяет указанный размер шрифта к тексту, оборачивая его в HTML.
    """
    return f'<div style="font-size: {size}; white-space: pre-wrap;">{text}</div>'


def html_to_text(html: str) -> str:
    """
    Преобразует HTML-строку, сгенерированную apply_text_size(), в простой текст без тегов.
    """
    if html.startswith("<div"):
        start = html.find(">") + 1
        end = html.rfind("</div>")
        return html[start:end]
    return html


def insert_text_from_transcription(transcription_html: str) -> str:
    """
    Извлекает чистый текст из HTML-блока транскрипции.
    """
    return html_to_text(transcription_html)


def insert_text_from_translation(auto_translation_html: str) -> str:
    """
    Извлекает чистый текст из HTML-блока авто-перевода.
    """
    return html_to_text(auto_translation_html)


def process_audio_stream(
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
    auto_translation_history: list,
    finalize_interval: float,
    last_finalize_time: Optional[float]
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
    list,
    Optional[float]
]:
    """
    Обрабатывает поток аудиоданных, выполняет транскрипцию, собирает статистику задержек и
    выполняет финализацию текущей строки транскрипта каждые N секунд (задаётся слайдером).
    """
    start_time = time.time()

    if latency_data is None:
        latency_data = {
            "total_resampling_latency": [],
            "total_transcription_latency": [],
            "total_latency": [],
        }

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
        latency_data["total_resampling_latency"].append(0.0)
        latency_data["total_transcription_latency"].append(0.0)
        transcription = "ERROR"
        language = "ERROR"
        language_pred = 0.0

    end_time = time.time()
    total_latency = end_time - start_time
    latency_data["total_latency"].append(total_latency)

    max_len = max(len(lst) for lst in latency_data.values())
    for key in latency_data:
        while len(latency_data[key]) < max_len:
            latency_data[key].append(0.0)

    # Инициализируем finalized_line, чтобы избежать ошибок
    finalized_line = None

    # Новая логика финализации по времени
    current_time = time.time()
    if last_finalize_time is None:
        last_finalize_time = current_time

    if (current_time - last_finalize_time) >= finalize_interval:
        finalized_line = current_transcription
        if finalized_line.strip():
            transcription_history.append(finalized_line)
        current_transcription = ""
        last_finalize_time = current_time
    else:
        if len(stream) > sampling_rate * max_length:
            finalized_line = current_transcription
            if finalized_line.strip():
                transcription_history.append(finalized_line)
            overlap_samples = int(overlap_duration * sampling_rate)
            stream = stream[-overlap_samples:]
            current_transcription = ""

    display_text = f"{current_transcription}\n\n" + "\n\n".join(transcription_history[::-1])
    transcription_html = apply_text_size(display_text, font_size)

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

    # Выполнение авто-перевода только для финализированной строки, если она определена
    if auto_translate_checked and finalized_line is not None:
        try:
            auto_translated_line = translate_text(finalized_line, language if not source_lang else source_lang, target_lang)
            auto_translation_history.append(auto_translated_line)
        except Exception as e:
            print(f"Auto translation error: {e}")

    auto_translation_display = "\n\n" + "\n".join(auto_translation_history[::-1])
    auto_translation_html = apply_text_size(auto_translation_display, font_size)

    manual_translation = ""

    return (
        stream,                      # обновлённый аудиопоток
        transcription_html,          # HTML транскрипции
        info_df_html,                # статистика задержек (HTML)
        latency_data,                # данные задержек
        current_transcription,       # текущий буфер транскрипции
        transcription_history,       # история транскрипций
        language_and_pred_text,      # информация о языке
        manual_translation,          # ручной перевод (пустая строка)
        auto_translation_html,       # HTML авто-перевода
        auto_translation_history,    # история авто-перевода
        last_finalize_time           # обновлённое время финализации
    )

def update_font_size_callback(transcription_html: str, auto_translation_html: str, new_font_size: str) -> Tuple[str, str]:
    """
    Обновляет размер шрифта для блоков транскрипции и авто-перевода, используя новый размер.
    """
    transcription_text = html_to_text(transcription_html)
    auto_translation_text = html_to_text(auto_translation_html)
    new_transcription_html = apply_text_size(transcription_text, new_font_size)
    new_auto_translation_html = apply_text_size(auto_translation_text, new_font_size)
    return new_transcription_html, new_auto_translation_html

# -----------------------------------------------------------------------------
#                            СОЗДАНИЕ ИНТЕРФЕЙСА
# -----------------------------------------------------------------------------
with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Default(spacing_size=gr.themes.sizes.spacing_sm, radius_size=gr.themes.sizes.radius_none)
) as demo:

    gr.Markdown("# Live-транскрибация, перевод текста в речь и автоматический перевод")
    gr.Markdown("**Что здесь?** Данный интерфейс позволяет выполнять три основные задачи: живую транскрипцию речи с микрофона, машинный перевод текста и преобразование текста в речь (TTS). Используйте соответствующие разделы для работы с каждым из сервисов.")

    # Состояния
    stream_state = gr.State(None)
    latency_data_state = gr.State(None)
    transcription_history_state = gr.State([])
    current_transcription_state = gr.State("")
    auto_translation_history_state = gr.State([])
    last_finalize_time_state = gr.State(None)

    # -----------------------------------------------------------------------------
    #                           БЛОК ЖИВОЙ ТРАНСКРИПЦИИ
    # -----------------------------------------------------------------------------
    with gr.Column():
        gr.Markdown("## Живая транскрипция")
        gr.Markdown("**Что здесь?** В этом разделе происходит запись аудио с микрофона, его автоматическая транскрипция и, при желании, автоматический перевод. Настройте параметры для управления длительностью аудио чанков, перекрытием и интервалом финализации транскрипции.")

        with gr.Row():
            mic_audio_input = gr.Audio(sources=["microphone"], streaming=True, label="Микрофон (запись аудио)")
            language_code_input = gr.Dropdown(
                [
                    ("Auto detect", ""),
                    ("English", "en"),
                    ("Spanish", "es"),
                    ("German", "de"),
                    ("Russian", "ru")
                ],
                value="",
                label="Код языка (распознавание)",
                multiselect=False
            )
            max_length_input = gr.Slider(
                value=7,
                minimum=2,
                maximum=30,
                step=1,
                label="Длина одного чанка (сек)",
                info="Максимальная длительность аудио перед финализацией"
            )
            overlap_slider = gr.Slider(
                value=2.0,
                minimum=0.0,
                maximum=5.0,
                step=0.1,
                label="Перекрытие чанков (сек)",
                info="Длительность перекрытия между аудио чанками"
            )
            finalize_interval_slider = gr.Slider(
                value=6,
                minimum=2,
                maximum=30,
                step=1,
                label="Интервал обновления строки (сек)",
                info="Интервал, через который текущая строка переносится в историю"
            )

        with gr.Row():
            transcription_html = gr.HTML(label="Транскрипция (автоматическая)", elem_id="transcription_display_container")
            auto_translation_html = gr.HTML(label="Автоматический перевод", elem_id="auto_translation_display_container")

        transcription_language_prod_output = gr.Text(
            lines=1,
            show_label=True,
            label="Информация о распознанном языке",
            interactive=False
        )

        with gr.Row():
            auto_translate_checkbox = gr.Checkbox(label="Авто-перевод (вкл/выкл) [справа от оригинала транскрипта]", value=False, interactive=True)
            font_size_dropdown = gr.Dropdown(
                [
                    "8px", "10px", "12px", "14px", "16px",
                    "18px", "20px", "24px", "36px", "48px"
                ],
                value="14px",
                label="Размер шрифта текста",
                multiselect=False
            )

        with gr.Row():
            info_table_output = gr.HTML(label="Статистика задержек (в миллисекундах)")

        # Кнопка Reset вынесена в отдельный ряд под блоком
        with gr.Row():
            reset_button = gr.Button("Reset (Сброс всех данных)")

    # -----------------------------------------------------------------------------
    #                           БЛОК ПЕРЕВОДА ТЕКСТА (РУЧНОЙ)
    # -----------------------------------------------------------------------------
    with gr.Column():
        gr.Markdown("## Машинный перевод текста")
        gr.Markdown("**Что здесь?** Здесь вы можете вручную перевести текст. Выберите язык оригинала и язык перевода, затем используйте кнопки для вставки текста из транскрипции или автоматического перевода.")
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
                label="Язык оригинала",
                multiselect=False
            )
            target_lang = gr.Dropdown(
                [
                    ("English", "en"),
                    ("Spanish", "es"),
                    ("Italian", "it"),
                    ("German", "de"),
                    ("Hungarian", "hu"),
                    ("Russian", "ru"),
                    ("🐖", "🐖ru_pig🐖(full_stilization)"),
                ],
                value="en",
                label="Язык перевода",
                multiselect=False
            )
        with gr.Row():
            insert_text_button = gr.Button("Вставить транскрипцию")
            insert_translation_button = gr.Button("Вставить авто-перевод")
        manual_translation_input = gr.Textbox(
            lines=5,
            label="Текст для перевода (вставьте или введите текст)",
            interactive=True,
            placeholder="Здесь будет отображаться текст для перевода..."
        )
        translate_button = gr.Button("Перевести текст")
        output_text = gr.Textbox(
            lines=5,
            label="Результат перевода (ручной перевод)",
            interactive=False
        )

        translate_button.click(
            translate_text,
            inputs=[manual_translation_input, source_lang, target_lang],
            outputs=output_text
        )
        insert_text_button.click(
            insert_text_from_transcription,
            inputs=[transcription_html],
            outputs=manual_translation_input
        )
        insert_translation_button.click(
            insert_text_from_translation,
            inputs=[auto_translation_html],
            outputs=manual_translation_input
        )

    # -----------------------------------------------------------------------------
    #                           БЛОК ТЕКСТ В РЕЧЬ (TTS)
    # -----------------------------------------------------------------------------
    with gr.Column():
        gr.Markdown("## Текст в речь (TTS)")
        gr.Markdown("**Что здесь?** В этом разделе вы можете преобразовать введённый текст в аудиофайл с речью. Выберите язык TTS и используйте кнопки для вставки текста из транскрипции или авто-перевода.")
        with gr.Row():
            tts_text_box = gr.Textbox(label="Текст для TTS (введите или вставьте текст)", placeholder="Введите текст для преобразования в речь...")
            tts_lanuage_code = gr.Dropdown(
                [
                    ("Russian", "ru"),
                    ("English", "en"),
                ],
                value="ru",
                label="Язык для TTS",
                multiselect=False
            )
        with gr.Row():
            tts_insert_text_button = gr.Button("Вставить транскрипцию в TTS")
            tts_insert_translation_button = gr.Button("Вставить авто-перевод в TTS")
        load_audio_button = gr.Button("Преобразовать текст в речь")
        loaded_audio_display = gr.Audio(label="Сгенерированный аудиофайл (TTS)", interactive=False)

        tts_insert_text_button.click(
            insert_text_from_transcription,
            inputs=[transcription_html],
            outputs=tts_text_box
        )
        tts_insert_translation_button.click(
            insert_text_from_translation,
            inputs=[auto_translation_html],
            outputs=tts_text_box
        )

        load_audio_button.click(
            text_to_speech,
            inputs=[tts_text_box, tts_lanuage_code],
            outputs=[loaded_audio_display]
        )

    # -----------------------------------------------------------------------------
    #                      СВЯЗИ МЕЖДУ КОМПОНЕНТАМИ
    # -----------------------------------------------------------------------------
    mic_audio_input.stream(
        fn=process_audio_stream,
        inputs=[
            stream_state,                      # текущий аудиопоток
            mic_audio_input,                   # новые аудиоданные с микрофона
            max_length_input,                  # длина чанка (сек)
            overlap_slider,                    # перекрытие чанков (сек)
            latency_data_state,                # данные задержек
            current_transcription_state,       # текущая транскрипция
            transcription_history_state,       # история транскрипций
            language_code_input,               # выбор кода языка
            font_size_dropdown,                # выбранный размер шрифта
            source_lang,                       # исходный язык для перевода
            target_lang,                       # целевой язык для перевода
            auto_translate_checkbox,           # авто-перевод включён/выключён
            auto_translation_history_state,    # история авто-перевода
            finalize_interval_slider,          # интервал финализации (сек)
            last_finalize_time_state           # состояние времени последней финализации
        ],
        outputs=[
            stream_state,                      # обновлённый аудиопоток
            transcription_html,                # HTML транскрипции
            info_table_output,                 # статистика задержек (HTML)
            latency_data_state,                # обновлённые данные задержек
            current_transcription_state,       # текущий буфер транскрипции
            transcription_history_state,       # история транскрипций
            transcription_language_prod_output,# информация о языке
            output_text,                       # ручной перевод (пустой)
            auto_translation_html,             # HTML авто-перевода
            auto_translation_history_state,    # обновлённая история авто-перевода
            last_finalize_time_state           # обновлённое время финализации
        ],
        show_progress="hidden"
    )

    reset_button.click(
        _reset_button_click,
        inputs=[
            stream_state,
            latency_data_state,
            transcription_history_state,
            current_transcription_state,
            auto_translation_history_state,
            last_finalize_time_state
        ],
        outputs=[
            stream_state,
            transcription_html,
            info_table_output,
            latency_data_state,
            current_transcription_state,
            transcription_history_state,
            transcription_language_prod_output,
            output_text,
            auto_translation_html,
            auto_translation_history_state,
            last_finalize_time_state
        ]
    )
    
    # -----------------------------------------------------------------------------
    # Обновление размера шрифта в реальном времени при изменении значения dropdown
    # -----------------------------------------------------------------------------
    font_size_dropdown.change(
        fn=update_font_size_callback,
        inputs=[transcription_html, auto_translation_html, font_size_dropdown],
        outputs=[transcription_html, auto_translation_html]
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
