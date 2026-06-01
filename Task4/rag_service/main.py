from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from rag_pipeline import RAGPipeline

app = FastAPI()
rag = RAGPipeline()

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)

    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return v.strip()

@app.post("/ask")
def ask(request: QueryRequest):
    result = rag.run(request.query)
    return result

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
    }