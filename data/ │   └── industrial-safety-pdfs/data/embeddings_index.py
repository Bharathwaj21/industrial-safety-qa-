import sqlite3
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

DB_PATH = "db/chunks.sqlite"
INDEX_PATH = "db/faiss.index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    model = SentenceTransformer(MODEL_NAME)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, chunk FROM chunks")
    rows = c.fetchall()

    ids = []
    embeddings = []
    for r in tqdm(rows):
        ids.append(r[0])
        embeddings.append(model.encode(r[1], normalize_embeddings=True))

    embeddings = np.vstack(embeddings).astype("float32")

    index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
    index.add_with_ids(embeddings, np.array(ids))
    faiss.write_index(index, INDEX_PATH)

    print(f"✅ FAISS index built at {INDEX_PATH}")

if __name__ == "__main__":
    main()
