import json
from typing import List, Dict, Any
from src.embeddings import EmbeddingEngine
from src.reranker import Reranker
from src.vector_db import VectorDatabase
from api.client import call_llm_api
from config.settings import settings
import requests

class ConstitutionRAG:
    def __init__(self):
        self.articles = self._load_articles()
        self.embedding_engine = EmbeddingEngine()
        self.reranker = Reranker()
        self.vector_db = VectorDatabase(
            vdb_path=settings.vdb_path,
            embedding_function=self.embedding_engine.get_function()
        )

    def _load_articles(self) -> List[Dict[str, Any]]:
        with open(settings.json_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        return [art for art in articles if art["section"] != "РАЗДЕЛ ВТОРОЙ"]

    def retrieve_relevant_articles(self, query: str) -> List[Dict]:
        initial_results = self.vector_db.search(query, k=settings.initial_search_k)
        reranked_results = self.reranker.rerank(query, initial_results, settings.rerank_top_k)
        
        results = []
        for doc in reranked_results:
            results.append({
                "article_number": doc.metadata["article_number"],
                "section": doc.metadata["section"],
                "chapter": doc.metadata["chapter"],
                "content": doc.page_content,
                "source": doc.metadata["source"]
            })
        return results

    def generate_answer(self, query: str) -> Dict:
        relevant_articles = self.retrieve_relevant_articles(query)
        
        if not relevant_articles:
            return {"answer": "Не удалось найти релевантные статьи в Конституции РФ для ответа на ваш вопрос.", "used_articles": []}

        context = ""
        for article in relevant_articles:
            context += f"СТАТЬЯ {article['article_number']}\n{article['content']}\n\n"

        system_prompt = (
            "ТЫ — ЮРИДИЧЕСКИЙ АССИСТЕНТ ПО КОНСТИТУЦИИ РОССИЙСКОЙ ФЕДЕРАЦИИ.\n"
            "СТРОГО СЛЕДУЙ ПРАВИЛАМ:\n"
            "1. Отвечай ИСКЛЮЧИТЕЛЬНО на основе предоставленного контекста.\n"
            "2. Если информация отсутствует в контексте — ответь точной фразой:\n"
            "'В Конституции Российской Федерации не указано'.\n"
            "3. Цитируй конкретные статьи при ответе.\n"
            "4. Не интерпретируй текст, не добавляй примеры, не выдумывай информацию.\n"
            "5. Используй формальный юридический стиль."
        )

        full_prompt = f"КОНТЕКСТ:\n{context}\n\nВОПРОС:\n{query}"

        try:
            answer = call_llm_api(full_prompt, num_ctx=settings.default_num_ctx)
        except requests.exceptions.RequestException:
            answer = call_llm_api(full_prompt, num_ctx=settings.reduced_num_ctx)

        return {
            "answer": answer,
            "used_articles": [
                {
                    "number": article['article_number'],
                    "section": article['section'],
                    "chapter": article['chapter']
                } for article in relevant_articles
            ]
        }