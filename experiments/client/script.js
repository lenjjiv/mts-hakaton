/**
 * Скрипт для записи аудио с микрофона и отправки его на сервер по WebSocket.
 */

let mediaRecorder;
let socket;

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const statusP = document.getElementById("status");
const resultP = document.getElementById("result");

/**
 * Инициализирует WebSocket соединение.
 */
function initWebSocket() {
    socket = new WebSocket("ws://localhost:8765");

    socket.onopen = function() {
        statusP.textContent = "Соединение с сервером установлено.";
    };

    socket.onmessage = function(event) {
        // При получении транскрипции от сервера выводим её на страницу
        const data = JSON.parse(event.data);
        resultP.textContent = data.text;
    };

    socket.onerror = function(error) {
        console.error("WebSocket ошибка: ", error);
        statusP.textContent = "Ошибка соединения с сервером.";
    };

    socket.onclose = function() {
        statusP.textContent = "Соединение закрыто.";
    };
}

startBtn.onclick = async function() {
    // Инициализируем WebSocket, если ещё не установлено
    if (!socket || socket.readyState !== WebSocket.OPEN) {
        initWebSocket();
    }

    // Запрашиваем доступ к микрофону
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = function(e) {
        // Отправляем аудиоданные (если они есть) на сервер
        if (e.data.size > 0 && socket.readyState === WebSocket.OPEN) {
            socket.send(e.data);
        }
    };

    // Запускаем запись с интервалом в 250 мс (отправка чанков)
    mediaRecorder.start(250);

    startBtn.disabled = true;
    stopBtn.disabled = false;
    statusP.textContent = "Запись началась...";
};

stopBtn.onclick = function() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    // Отправляем команду завершения записи на сервер
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send("stop");
    }

    startBtn.disabled = false;
    stopBtn.disabled = true;
    statusP.textContent = "Запись остановлена, ожидается результат...";
};
