Setup

Clone repo and install dependencies

git clone <repo_url>
cd industrial-safety-qa
pip install -r requirements.txt


Ingest PDFs into SQLite

python ingest.py


Build FAISS embeddings index

python embeddings_index.py


Run the API

uvicorn api:app --reload


POST /ask with JSON:

{
  "q": "What personal protective equipment is required?",
  "k": 3,
  "mode": "rerank"
}


Run evaluation

python evaluate.py



Learnings

This project demonstrates the value of combining vector embeddings (semantic search) with BM25 keyword matching for domain-specific Q&A. Even a small, CPU-friendly system can achieve noticeable improvements in retrieval quality. Chunking large documents and maintaining a lightweight FAISS index makes the pipeline efficient and suitable for on-premise deployment.
