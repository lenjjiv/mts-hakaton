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
from dotenv import load_dotenv

load_dotenv()

# Функция перевода текста
def translate_text(text, source_lang, target_lang):
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
    return translate_text(text, source_lang, target_lang)

# Создание интерфейса с переводчиком
with gr.Blocks() as demo:
    gr.Markdown("# Простое приложение для перевода текста")
    
    text_input = gr.Textbox(label="Введите текст")
    source_lang = gr.Textbox(label="Исходный язык (например, ru)")
    target_lang = gr.Textbox(label="Целевой язык (например, en)")
    translate_button = gr.Button("Translate")
    output_text = gr.Textbox(label="Переведённый текст")
    
    translate_button.click(translate_interface, inputs=[text_input, source_lang, target_lang], outputs=output_text)

    # Разделение между переводом и аудио-функционалом
    gr.Markdown("---")
    gr.Markdown("# Live Transcription PoC")

    # Хранение состояния
    stream_state = gr.State(None)
    latency_data_state = gr.State(None)
    transcription_history_state = gr.State([])
    current_transcription_state = gr.State("")

    with gr.Column():
        gr.Markdown("## Controls")
        with gr.Accordion(label="How to use it", open=False):
            gr.Markdown("""
### How to Use the Live Transcription Service
1. **Click 'Start'** and allow microphone access.
2. **Choose language** (or leave it on auto-detect).
3. **Click 'Stop'** when done and 'Reset' to clear.
4. **Use translation** by selecting a target language.
""")
        with gr.Row():
            mic_audio_input = gr.Audio(sources=["microphone"], streaming=True)
            reset_button = gr.Button("Reset")
            max_length_input = gr.Slider(value=10, minimum=2, maximum=30, step=1, label="Max length of audio (sec)")
            language_code_input = gr.Dropdown([("Auto detect", ""), ("English", "en"), ("Spanish", "es"),
                                               ("Italian", "it"), ("German", "de"), ("Hungarian", "hu"), 
                                               ("Russian", "ru")], value="", label="Language code", multiselect=False)

    gr.Markdown("## Transcription")
    transcription_language_prod_output = gr.Text(lines=1, show_label=False, interactive=False)
    transcription_display = gr.Textbox(lines=5, show_label=False, interactive=False, show_copy_button=True)

    # Text-to-Speech функционал
    with gr.Column():
        gr.Markdown("## Text to Speech")
        load_audio_button = gr.Button("Convert text to speech")
        with gr.Row():
            tts_text_box = gr.Textbox(label="Text")
            tts_lanuage_code = gr.Dropdown([("Auto detect", ""), ("English", "en"), ("Spanish", "es"),
                                            ("Italian", "it"), ("German", "de"), ("Hungarian", "hu"), 
                                            ("Russian", "ru")], value="", label="Language code", multiselect=False)
        loaded_audio_display = gr.Audio(label="Audio file", interactive=False)

    def tts_from_server(text: str, lang: str, save_path: str):
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
        save_path = "tmp/downloaded_audio.mp3"
        tts_from_server(text, language, save_path)
        return save_path

    load_audio_button.click(text_to_speech, inputs=[tts_text_box, tts_lanuage_code], outputs=[loaded_audio_display])

    def dummy_function(stream, new_chunk, max_length, latency_data, current_transcription, transcription_history, language_code):
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
            stream_resampled = librosa.resample(stream, orig_sr=sampling_rate, target_sr=16000)
            transcription, language, language_pred = send_audio_to_server(stream_resampled, str(language_code))
            current_transcription = f"{transcription}"
        except Exception as e:
            print(f"Error: {e}")

        end_time = time.time()
        if len(stream) > sampling_rate * max_length:
            stream = None
            transcription_history.append(current_transcription)
            current_transcription = ""

        display_text = f"{current_transcription}\n\n" + "\n\n".join(transcription_history[::-1])
        return stream, display_text, latency_data, current_transcription, transcription_history, f"Predicted Language: {language} ({language_pred * 100:.2f}%)"

    def send_audio_to_server(audio_data: np.ndarray, language_code: str = "") -> Tuple[str, str, float]:
        audio_data_bytes = audio_data.astype(np.int16).tobytes()
        response = requests.post("http://localhost:8000/predict", data=audio_data_bytes,
                                 params={"language_code": language_code},
                                 headers={"accept": "application/json", "Content-Type": "application/octet-stream"})
        result = response.json()
        return result["prediction"], result["language"], result["language_probability"]

    mic_audio_input.stream(dummy_function, [stream_state, mic_audio_input, max_length_input, latency_data_state,
                                            current_transcription_state, transcription_history_state, language_code_input],
                           [stream_state, transcription_display, latency_data_state, current_transcription_state,
                            transcription_history_state, transcription_language_prod_output], show_progress="hidden")

    def _reset_button_click(stream_state, transcription_display, latency_data_state, transcription_history_state, current_transcription_state):
        return None, "", None, [], "", ""

    reset_button.click(_reset_button_click, [stream_state, transcription_display, latency_data_state,
                                             transcription_history_state, current_transcription_state],
                       [stream_state, transcription_display, latency_data_state, transcription_history_state,
                        current_transcription_state, transcription_language_prod_output])

SSL_CERT_PATH: Optional[str] = os.environ.get("SSL_CERT_PATH", None)
SSL_KEY_PATH: Optional[str] = os.environ.get("SSL_KEY_PATH", None)
SSL_VERIFY: bool = bool(os.environ.get("SSL_VERIFY", False))
SHARE: bool = True

demo.launch(server_name="0.0.0.0", server_port=5656, ssl_certfile=SSL_CERT_PATH, ssl_keyfile=SSL_KEY_PATH, ssl_verify=SSL_VERIFY, share=SHARE)
