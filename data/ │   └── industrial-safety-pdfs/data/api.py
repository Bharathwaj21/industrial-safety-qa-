from fastapi import FastAPI
from pydantic import BaseModel
from reranker_hybrid import HybridReranker

app = FastAPI()
reranker = HybridReranker()

class Query(BaseModel):
    q: str
    k: int = 3
    mode: str = "rerank"  # or "baseline"

@app.post("/ask")
def ask(query: Query):
    results = reranker.rerank(query.q) if query.mode == "rerank" else reranker.baseline.search(query.q)
    if not results or results[0]['score'] < 0.3:
        return {"answer": None, "contexts": results, "reranker_used": query.mode}

    top_chunks = " ".join(r['chunk'] for r in results)
    answer = results[0]['chunk'].split(".")[0] + "."
    return {"answer": answer, "contexts": results, "reranker_used": query.mode}
