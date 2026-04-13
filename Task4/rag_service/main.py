from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import RAGPipeline

app = FastAPI()
rag = RAGPipeline()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask")
def ask(request: QueryRequest):
    result = rag.run(request.query)
    return result

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
    }