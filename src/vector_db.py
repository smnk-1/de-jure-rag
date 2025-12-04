import torch
from langchain_community.vectorstores import Chroma
from langchain_classic.schema import Document
from pathlib import Path
from typing import List
from config.settings import settings
import json
from typing import Dict, Any

class VectorDatabase:
    def __init__(self, vdb_path: str, embedding_function):
        self.vdb_path = vdb_path
        self.embedding_function = embedding_function
        self.db = self._init_db()

    def _init_db(self):
        if Path(self.vdb_path).exists() and list(Path(self.vdb_path).glob("*")):
            return Chroma(persist_directory=self.vdb_path, embedding_function=self.embedding_function)
        else:
            self._create_vector_db_from_articles(self.vdb_path, self.embedding_function)
            return Chroma(persist_directory=self.vdb_path, embedding_function=self.embedding_function)

    @staticmethod
    def _create_vector_db_from_articles(vdb_path: str, embedding_function):
        articles = VectorDatabase._load_articles_static()
        documents = []
        for article in articles:
            doc_text = f"Статья {article['article_number']}. {article['content']}"
            metadata = {
                "article_number": article["article_number"],
                "section": article["section"],
                "chapter": article["chapter"],
                "source": article["metadata"]["source"]
            }
            documents.append(Document(page_content=doc_text, metadata=metadata))

        Chroma.from_documents(
            documents=documents,
            embedding=embedding_function,
            persist_directory=vdb_path
        )

    @staticmethod
    def _load_articles_static() -> List[Dict[str, Any]]:
        with open(settings.json_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        return [art for art in articles if art["section"] != "РАЗДЕЛ ВТОРОЙ"]

    def search(self, query: str, k: int) -> List[Document]:
        return self.db.similarity_search(query, k=k)