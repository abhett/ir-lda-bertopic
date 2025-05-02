import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from modules.preprocessing import preprocess
from umap import UMAP
from hdbscan import HDBSCAN

# Load original documents
with open("asset/pelangipuisi.jsonl", "r", encoding="utf-8") as f:
    DOCS_RAW = [json.loads(line)["text"] for line in f]

# Preprocess for embedding
DOCS = [preprocess(doc) for doc in DOCS_RAW]

# Model untuk teks pendek multilingual
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# UMAP untuk reduksi dimensi
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

# HDBSCAN untuk clustering
hdbscan_model = HDBSCAN(min_cluster_size=3, min_samples=1, prediction_data=True)

# Inisialisasi dan fit BERTopic
bertopic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    language="indonesian",
    min_topic_size=3,
    calculate_probabilities=True,
    verbose=True
)

bertopic_model.fit_transform(DOCS)

def run_bertopic(query):
    if len(query.strip().split()) < 3:
        return "too_short", 0.0, [], []

    processed_query = preprocess(query)
    topic, probs = bertopic_model.transform([processed_query])
    topic_id = topic[0]
    confidence = probs[0][topic_id] if topic_id != -1 else 0.0

    # Ambil dokumen yang termasuk dalam topik ini
    related_docs = []
    if topic_id != -1:
        topics, _ = bertopic_model.transform(DOCS)
        related_docs = [DOCS_RAW[i] for i, t in enumerate(topics) if t == topic_id]

    # Ambil top words dari topik
    topic_words = get_topic_words(topic_id) if topic_id != -1 else []

    return topic_id, confidence, related_docs, topic_words

def get_topic_words(topic_id):
    words_scores = bertopic_model.get_topics().get(topic_id, [])
    return [word for word, _ in words_scores[:10]]  # Top 10 kata
