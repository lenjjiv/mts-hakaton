import requests

def stream_audio(text: str, save_path: str):
    url = "http://localhost:8000/stream-audio"
    data = {"text": text}

    try:
        response = requests.post(url, json=data, stream=True)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Аудио файл успешно сохранен в {save_path}")

    except requests.exceptions.RequestException as e:
        print(f"Ошибка при получении аудио: {e}")

if __name__ == "__main__":
    text = "Hello! This is a test of the Coqui TTS system."
    save_path = "downloaded_audio.mp3"
    stream_audio(text, save_path)
