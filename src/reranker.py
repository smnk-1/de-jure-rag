from sentence_transformers import CrossEncoder
from langchain_classic.schema import Document
from typing import List

class Reranker:
    def __init__(self):
        self.model = CrossEncoder('BAAI/bge-reranker-base', max_length=512)

    def rerank(self, query: str, candidates: List[Document], top_k: int) -> List[Document]:
        if not candidates:
            return []
        
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.model.predict(pairs)
        scored_docs = list(zip(candidates, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_k]]
    