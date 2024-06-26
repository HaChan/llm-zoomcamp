# Question 2
import requests
from elasticsearch import Elasticsearch

docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)


es = Elasticsearch('http://localhost:9200')

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "course": {"type": "keyword"},
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "section": {"type": "text"}
        }
    }
}

index_name = "courses_questions"

es.indices.create(index=index_name, body=index_settings)
for doc in documents:
    resp = es.index(index=index_name, body=doc)
    print(f"Indexed document ID: {resp['_id']}")

# Question 3
def search_query(query):
    return {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^4", "answer", "section"],
                            "type": "best_fields"
                        }
                    }
                ],
            }
        },
        "size": 5
    }

query = "How do I execute a command in a running docker container?"

response = es.search(index=index_name, body=search_query(query))
scores = [hit['_score'] for hit in response['hits']['hits']]
print(max(scores))

# Question 4
def search_filter(query, course_name):
    return {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^4", "answer", "section"],
                            "type": "best_fields"
                        }
                    }
                ],
                "filter": [
                    {
                        "term": {"course": course_name}
                    }
                ]
            }
        },
        "size": 3
    }
course_name = "machine-learning-zoomcamp"
response = es.search(index=index_name, body=search_filter(query, course_name))
questions = [hit['_source']['question'] for hit in response['hits']['hits']]
print(questions[2])

# Question 5
context = ""
context_template = """
Q: {question}
A: {text}
""".strip()
for hit in response['hits']['hits']:
    doc = hit['_source']
    context += context_template.format(question=doc['question'], text=doc['text']).strip()
    context += '\n\n'

context = context.strip()
question = 'How do I execute a command in a running docker container?'

prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()
print(len(prompt_template.format(question=question, context=context)))

# Question 6
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o")
len(encoding.encode(prompt_template.format(question=question, context=context)))
