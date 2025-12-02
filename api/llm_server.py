from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from config.settings import settings

app = FastAPI(title="LLM API for RAG System")

class GenerateRequest(BaseModel):
    prompt: str
    num_ctx: int = settings.default_num_ctx
    temperature: float = settings.temperature
    top_p: float = settings.top_p

class GenerateResponse(BaseModel):
    answer: str

@app.post(settings.llm_api_endpoint, response_model=GenerateResponse)
def generate(request: GenerateRequest):
    try:
        response = ollama.chat(
            model=settings.llm_model,
            messages=[{"role": "user", "content": request.prompt}],
            options={
                'temperature': request.temperature,
                'num_ctx': request.num_ctx,
                'top_p': request.top_p
            },
            keep_alive=settings.keep_alive
        )
        return GenerateResponse(answer=response['message']['content'])
    except ollama.ResponseError as e:
        if "CUDA out of memory" in str(e):
            try:
                response = ollama.chat(
                    model=settings.llm_model,
                    messages=[{"role": "user", "content": request.prompt}],
                    options={
                        'temperature': request.temperature,
                        'num_ctx': settings.reduced_num_ctx,
                        'top_p': request.top_p
                    },
                    keep_alive=settings.keep_alive
                )
                return GenerateResponse(answer=response['message']['content'])
            except Exception:
                raise HTTPException(status_code=500, detail="VRAM error even with reduced context")
        else:
            raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=settings.llm_api_port, log_level="info")