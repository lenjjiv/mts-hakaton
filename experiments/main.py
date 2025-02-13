import asyncio
from server.server import start_server

async def server_main():
    """
    Асинхронная функция для запуска WebSocket-сервера.
    """
    server = await start_server(host="0.0.0.0", port=8765)
    print("Сервер запущен на ws://0.0.0.0:8765")
    await server.wait_closed()  # Ожидание закрытия сервера

def main():
    """
    Основная точка входа. Запускает WebSocket сервер асинхронно.
    """
    asyncio.run(server_main())  # Запускаем сервер через asyncio.run()

if __name__ == "__main__":
    main()
