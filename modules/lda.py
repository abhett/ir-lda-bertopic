# modules/lda.py
import json
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from modules.preprocessing import preprocess
# Load documents from JSONL
with open("asset/pelangipuisi.jsonl", "r", encoding="utf-8") as f:
    DOCS_RAW = [json.loads(line)["text"] for line in f]

DOCS = [preprocess(doc) for doc in DOCS_RAW]

count_vectorizer = CountVectorizer(stop_words='english') 
X_counts = count_vectorizer.fit_transform(DOCS)

lda_model = LatentDirichletAllocation(n_components=10, random_state=42)
lda_model.fit(X_counts)

def run_lda(query):
    processed_query = preprocess(query)
    query_vec = count_vectorizer.transform([processed_query])
    topic_distribution = lda_model.transform(query_vec)[0]
    top_topic = np.argmax(topic_distribution)
    return top_topic, topic_distribution.tolist()

def get_all_topics():
    topics = []
    words = count_vectorizer.get_feature_names_out()
    for idx, topic in enumerate(lda_model.components_):
        top_features_idx = topic.argsort()[::-1][:5]
        top_words = [(words[i], topic[i]) for i in top_features_idx]
        topics.append({
            "id": idx,
            "words": [w for w, _ in top_words]
        })
    return topics

def get_topic_detail(topic_id):
    topic = lda_model.components_[topic_id]
    words = count_vectorizer.get_feature_names_out()
    word_scores = sorted(
        [(words[i], topic[i]) for i in range(len(topic))],
        key=lambda x: x[1],
        reverse=True
    )
    return word_scores[:20]

def search_word_in_topics(word_query):
    word_query = word_query.lower()
    words = count_vectorizer.get_feature_names_out()
    matching_topics = []

    for idx, topic in enumerate(lda_model.components_):
        top_features_idx = topic.argsort()[::-1]
        topic_words = [(words[i], topic[i]) for i in top_features_idx if words[i] == word_query]
        if topic_words:
            matching_topics.append({
                "id": idx,
                "word": topic_words[0][0],
                "score": topic_words[0][1]
            })
    return matching_topics

