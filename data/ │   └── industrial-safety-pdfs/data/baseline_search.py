import faiss
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

DB_PATH = "db/chunks.sqlite"
INDEX_PATH = "db/faiss.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class BaselineSearch:
    def __init__(self, k=5):
        self.k = k
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(INDEX_PATH)
        self.conn = sqlite3.connect(DB_PATH)

    def search(self, query):
        q_emb = self.model.encode(query, normalize_embeddings=True).astype("float32")
        D, I = self.index.search(np.expand_dims(q_emb, axis=0), self.k)
        results = []
        for score, idx in zip(D[0], I[0]):
            chunk = self.conn.execute("SELECT chunk, doc_name FROM chunks WHERE id=?", (int(idx),)).fetchone()
            if chunk:
                results.append({"chunk": chunk[0], "doc": chunk[1], "score": float(score)})
        return results
