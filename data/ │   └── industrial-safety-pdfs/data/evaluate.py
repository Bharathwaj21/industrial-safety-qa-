import json
from baseline_search import BaselineSearch
from reranker_hybrid import HybridReranker

# Load evaluation questions
with open("eight_questions.json") as f:
    questions = json.load(f)

baseline = BaselineSearch(k=3)
reranker = HybridReranker(alpha=0.7, k=3)

print(f"{'Question':<70} | {'Top Baseline':<50} | {'Top Rerank':<50}")
print("-" * 180)

for q in questions:
    query = q["q"]
    base_res = baseline.search(query)
    rerank_res = reranker.rerank(query)

    top_base = base_res[0]['chunk'][:50].replace("\n", " ") if base_res else "None"
    top_rerank = rerank_res[0]['chunk'][:50].replace("\n", " ") if rerank_res else "None"

    print(f"{query:<70} | {top_base:<50} | {top_rerank:<50}")
