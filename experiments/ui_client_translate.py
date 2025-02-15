import gradio as gr
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def translate_text(text, source_lang, target_lang):
    api_key = os.getenv("PROXY_API_KEY")
    url = "https://api.proxyapi.ru/google/v1/models/gemini-1.5-pro:generateContent"
    
    prompt = (
        f"Переведи с {source_lang} на {target_lang}: {text}. В ответе только текст без доп символов."
    )
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": 150
        }
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    if response.status_code == 200:
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    else:
        return f"Ошибка {response.status_code}: {response.text}"

def translate_interface(text, source_lang, target_lang):
    return translate_text(text, source_lang, target_lang)

with gr.Blocks() as demo:
    gr.Markdown("# Простое приложение для перевода текста")
    text_input = gr.Textbox(label="Введите текст")
    source_lang = gr.Textbox(label="Исходный язык (например, ru)")
    target_lang = gr.Textbox(label="Целевой язык (например, en)")
    translate_button = gr.Button("Translate")
    output_text = gr.Textbox(label="Переведённый текст")
    
    translate_button.click(translate_interface, inputs=[text_input, source_lang, target_lang], outputs=output_text)

demo.launch()
