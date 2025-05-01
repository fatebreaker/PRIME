from flask import Flask, render_template, request
from functools import lru_cache
import math
import os
from dotenv import load_dotenv

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
import pickle

load_dotenv()

INDEX_NAME = "textbooks"
INDEX_ROOT = "/data/data_user_alpha/MedRAG/corpus/experiments/notebook/indexes/textbooks"
with open('/data/data_user_alpha/MedRAG/corpus/textbooks/all_chunks.pickle', 'rb') as file:
    collection = pickle.load(file)

app = Flask(__name__)

# with Run().context(RunConfig(experiment='notebook')):
searcher = Searcher(index=INDEX_ROOT, index_root= INDEX_ROOT, collection=collection)
# searcher = Searcher(index=INDEX_NAME, index_root=INDEX_ROOT)
counter = {"api" : 0}

@lru_cache(maxsize=1000000)
def api_search_query(query, k):
    # print(f"Query={query}")
    if k == None: k = 10
    k = min(int(k), 100)
    pids, ranks, scores = searcher.search(query, k=100)
    pids, ranks, scores = pids[:k], ranks[:k], scores[:k]
    # passages = [collection[pid] for pid in pids]
    probs = [math.exp(score) for score in scores]
    probs = [prob / sum(probs) for prob in probs]
    topk = []
    for pid, rank, score, prob in zip(pids, ranks, scores, probs):
        text = searcher.collection[pid]
        d = {'text': text, 'pid': pid, 'rank': rank, 'score': score, 'prob': prob, 'passage': collection[pid]}
        topk.append(d)
    topk = list(sorted(topk, key=lambda p: (-1 * p['score'], p['pid'])))
    return {"query" : query, "topk": topk}

@app.route("/api/search", methods=["GET"])
def api_search():
    if request.method == "GET":
        counter["api"] += 1
        # print("API request count:", counter["api"])
        return api_search_query(request.args.get("query"), request.args.get("k"))
    else:
        print(request.method)
        return ('', 405)

if __name__ == "__main__":
    app.run("0.0.0.0", 6000)