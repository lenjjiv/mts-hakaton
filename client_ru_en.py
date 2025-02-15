import requests

def stream_audio(text: str, lang: str, save_path: str):
    """Отправляет текст на сервер и сохраняет аудиофайл."""
    url = "http://localhost:8000/stream-audio"
    data = {"text": text, "lang": lang}

    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Аудио файл сохранен в {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении аудио: {e}")

if __name__ == "__main__":
    # Ввод текста и выбора языка
    text = "Hello my friend"
    lang = 'en'  # ru или en

    if lang not in ["en", "ru"]:
        print("Некорректный выбор языка. Используйте 'en' или 'ru'.")
    else:
        save_path = f"downloaded_audio.mp3"
        stream_audio(text, lang, save_path)
