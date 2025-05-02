# modules/ir.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
from modules.preprocessing import preprocess

# Load documents from JSONL
with open("asset/pelangipuisi.jsonl", "r", encoding="utf-8") as f:
    DOCS_RAW = [json.loads(line)["text"] for line in f]

DOCS = [preprocess(doc) for doc in DOCS_RAW]

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(DOCS)

def run_ir(query):
    processed_query = preprocess(query)
    query_vec = vectorizer.transform([processed_query])
    sim = cosine_similarity(query_vec, X_tfidf).flatten()
    results = [(DOCS_RAW[i], sim[i]) for i in sim.argsort()[::-1]]
    return results
