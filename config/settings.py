import configparser
import os

config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), 'settings.cfg'))

class Settings:
    # Пути
    json_path = config.get('PATHS', 'json_path')
    vdb_path = config.get('PATHS', 'vdb_path')
    llm_model = config.get('PATHS', 'llm_model')
    
    # Параметры API LLM
    # llm_api_host больше нет в .cfg
    llm_api_port = config.getint('API', 'llm_api_port') # <-- Только порт
    llm_api_endpoint = config.get('API', 'llm_api_endpoint')
    # <-- llm_api_url больше нет в .cfg
    # Генерируем URL динамически, используя фиксированный хост 'localhost'
    llm_api_url = f"http://localhost:{llm_api_port}" # <-- Теперь зашито в коде
    
    # Параметры UI
    streamlit_port = config.getint('UI', 'streamlit_port')
    
    # Гиперпараметры генерации
    default_num_ctx = config.getint('GENERATION', 'default_num_ctx')
    reduced_num_ctx = config.getint('GENERATION', 'reduced_num_ctx')
    temperature = config.getfloat('GENERATION', 'temperature')
    top_p = config.getfloat('GENERATION', 'top_p')
    keep_alive = config.get('GENERATION', 'keep_alive')
    
    # Параметры поиска
    initial_search_k = config.getint('SEARCH', 'initial_search_k')
    rerank_top_k = config.getint('SEARCH', 'rerank_top_k')
    max_articles_in_context = config.getint('SEARCH', 'max_articles_in_context')

    # Настройки системы
    device = config.get('SYSTEM', 'device')
    if device == 'auto':
        try:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            device = 'cpu'

settings = Settings()