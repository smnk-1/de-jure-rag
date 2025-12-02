# Убираем: from langchain.schema import Document (не используется)
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List
from config.settings import settings

class EmbeddingEngine:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-small')
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Embedding model loaded on GPU")
        else:
            print("Embedding model loaded on CPU")

    def _average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_function(self):
        embedding_engine_instance = self

        def _embed_documents(texts: List[str]) -> List[List[float]]:
            embeddings = []
            for text in texts:
                input_text = f"passage: {text}"
                inputs = embedding_engine_instance.tokenizer(
                    input_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = embedding_engine_instance.model(**inputs)

                embeddings_vec = embedding_engine_instance._average_pool(
                    outputs.last_hidden_state, 
                    inputs['attention_mask']
                )
                
                embeddings_vec = F.normalize(embeddings_vec, p=2, dim=1)
                embeddings.append(embeddings_vec.cpu().numpy().flatten().tolist())
            
            return embeddings

        class EmbeddingFunction:
            def embed_query(self, query: str):
                input_text = f"query: {query}"
                inputs = embedding_engine_instance.tokenizer(
                    input_text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                )
                
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = embedding_engine_instance.model(**inputs)

                embeddings_vec = embedding_engine_instance._average_pool(
                    outputs.last_hidden_state, 
                    inputs['attention_mask']
                )
                
                embeddings_vec = F.normalize(embeddings_vec, p=2, dim=1)
                return embeddings_vec.cpu().numpy().flatten().tolist()
            
            def embed_documents(self, texts: List[str]):
                return _embed_documents(texts)
        
        return EmbeddingFunction()
