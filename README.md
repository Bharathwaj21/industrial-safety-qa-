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
