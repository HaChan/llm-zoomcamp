# Q1 Getting embeddings model

from sentence_transformers import SentenceTransformer
embedding_model = SentenceTransformer('multi-qa-distilbert-cos-v1')

user_question = "I just discovered the course. Can I still join it?"
ev = embedding_model.encode(user_question)

print(ev[0])

# Q2 Creating the embeddings

import requests

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/documents-with-ids.json'
docs_url = f'{base_url}/{relative_url}?raw=1'
docs_response = requests.get(docs_url)
documents = docs_response.json()

ml_docs = [doc for doc in documents if doc['course'] == 'machine-learning-zoomcamp']
embeddings = []

for doc in ml_docs:
    qa_text = f"{doc['question']} {doc['text']}"
    embed = embedding_model.encode(qa_text)
    embeddings.append(embed)

import numpy as np

X = np.array(embeddings)
print(X.shape)

# Q3 Search
scores = X.dot(ev)
print(max(scores))

# Q4 Hit-rate for our search engine
class VectorSearchEngine():
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, query, num_results=10):
        v_query = embedding_model.encode(query)
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [self.documents[i] for i in idx]

search_engine = VectorSearchEngine(documents=ml_docs, embeddings=X)
search_result = search_engine.search(v, num_results=5)

import pandas as pd

base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/ground-truth-data.csv'
ground_truth_url = f'{base_url}/{relative_url}?raw=1'

df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records')

def hit_rate(relevance_total):
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt = cnt + 1
    return cnt / len(relevance_total)

def mrr(relevance_total):
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)
    return total_score / len(relevance_total)

def evaluate(ground_truth, search_engine):
    relevance_total = []
    for q in ground_truth:
        doc_id = q['document']
        results = search_engine.search(q['question'], 5)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)
    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }

d_res = evaluate(ground_truth, search_engine)
print(d_res['hit_rate'])

# Q5 Indexing with Elasticsearch
for doc in ml_docs:
    qa_text = f"{doc['question']} {doc['text']}"
    doc['qa_embed'] = embedding_model.encode(qa_text)

from elasticsearch import Elasticsearch

es_client = Elasticsearch('http://localhost:9200')

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "qa_embed": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}
index_name = "course-qa"
es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)

for doc in ml_docs:
    es_client.index(index=index_name, document=doc)

class ESEngine():
    def __init__(self, es_client):
        self.es_client = es_client

    def search(self, query, num_results=10):
        v_query = embedding_model.encode(query)
        knn = {
            "field": 'qa_embed',
            "query_vector": v_query,
            "k": num_results,
            "num_candidates": 10000,
        }
        es_query = {
            'knn': knn,
            '_source': ["text", "section", "question", "course", "id"]
        }
        es_results = es_client.search(index=index_name, body=es_query)

        return [hit['_source'] for hit in es_results['hits']['hits']]

es_engine = ESEngine(es_client)
es_results = es_engine.search(user_question, 5)
print(es_results[0]['id'])

# Q6 Hit-rate for Elasticsearch
es_res = evaluate(ground_truth, search_engine)
print(es_res['hit_rate'])
