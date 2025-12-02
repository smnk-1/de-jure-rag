import requests
from config.settings import settings
from typing import Optional

def call_llm_api(prompt: str, num_ctx: Optional[int] = None) -> str:
    if num_ctx is None:
        num_ctx = settings.default_num_ctx

    response = requests.post(
        f"{settings.llm_api_url}{settings.llm_api_endpoint}",
        json={
            "prompt": prompt,
            "num_ctx": num_ctx,
            "temperature": settings.temperature,
            "top_p": settings.top_p
        }
    )
    response.raise_for_status()
    return response.json()["answer"]