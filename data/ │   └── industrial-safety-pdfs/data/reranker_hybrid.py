from baseline_search import BaselineSearch
import sqlite3
import math

class HybridReranker:
    def __init__(self, alpha=0.7, k=5):
        self.alpha = alpha
        self.baseline = BaselineSearch(k=k)
        self.conn = sqlite3.connect("db/chunks.sqlite")
        self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk, content='chunks', content_rowid='id')")

    def rerank(self, query):
        base_results = self.baseline.search(query)

        bm25_scores = {}
        for row in self.conn.execute("SELECT rowid, bm25(chunks_fts) FROM chunks_fts WHERE chunks_fts MATCH ?", (query,)):
            bm25_scores[row[0]] = -row[1]  # bm25() gives lower = better

        reranked = []
        for r in base_results:
            rowid = self.conn.execute("SELECT id FROM chunks WHERE chunk=? LIMIT 1", (r['chunk'],)).fetchone()[0]
            bm25_score = bm25_scores.get(rowid, 0.0)
            final_score = self.alpha * r['score'] + (1 - self.alpha) * (bm25_score / 10)
            reranked.append({**r, "rerank_score": final_score})

        reranked.sort(key=lambda x: x['rerank_score'], reverse=True)
        return reranked
