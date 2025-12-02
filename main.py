import subprocess
import threading
import time
import uvicorn
from api.llm_server import app as llm_app
from config.settings import settings


def run_llm_api():
    uvicorn.run(
        llm_app,
        host="localhost",
        port=settings.llm_api_port,
        log_level="info"
    )

def run_streamlit():
    subprocess.run([
        "streamlit", "run", 
        "ui/streamlit_app.py",
        "--server.port", str(settings.streamlit_port),
        "--server.headless", "true"
    ])

if __name__ == "__main__":
    print(f"Запуск API сервера на порту {settings.llm_api_port}...")
    api_thread = threading.Thread(target=run_llm_api)
    api_thread.daemon = True
    api_thread.start()

    time.sleep(3)

    print(f"Запуск Streamlit интерфейса на порту {settings.streamlit_port}...")
    run_streamlit()
