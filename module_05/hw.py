import io
import requests
import docx
import numpy as np
from elasticsearch import Elasticsearch

from datetime import datetime

# Q2. Reading the documents
def clean_line(line):
    line = line.strip()
    line = line.strip('\uFEFF')
    return line

def read_faq(file_id):
    url = f'https://docs.google.com/document/d/{file_id}/export?format=docx'

    response = requests.get(url)
    response.raise_for_status()

    with io.BytesIO(response.content) as f_in:
        doc = docx.Document(f_in)

    questions = []

    question_heading_style = 'heading 2'
    section_heading_style = 'heading 1'

    heading_id = ''
    section_title = ''
    question_title = ''
    answer_text_so_far = ''

    for p in doc.paragraphs:
        style = p.style.name.lower()
        p_text = clean_line(p.text)

        if len(p_text) == 0:
            continue

        if style == section_heading_style:
            section_title = p_text
            continue

        if style == question_heading_style:
            answer_text_so_far = answer_text_so_far.strip()
            if answer_text_so_far != '' and section_title != '' and question_title != '':
                questions.append({
                    'text': answer_text_so_far,
                    'section': section_title,
                    'question': question_title,
                })
                answer_text_so_far = ''

            question_title = p_text
            continue

        answer_text_so_far += '\n' + p_text

    answer_text_so_far = answer_text_so_far.strip()
    if answer_text_so_far != '' and section_title != '' and question_title != '':
        questions.append({
            'text': answer_text_so_far,
            'section': section_title,
            'question': question_title,
        })

    return questions

def load_data():
    faq_documents = {
        'llm-zoomcamp': '1qZjwHkvP0lXHiE4zdbWyUXSVfmVGzougDD6N37bat3E',
    }
    documents = []

    for course, file_id in faq_documents.items():
        print(course)
        course_documents = read_faq(file_id)
        documents.append({'course': course, 'documents': course_documents})

    return documents

data = load_data()

# Q3. Chunking
import hashlib

def generate_document_id(doc):
    combined = f"{doc['course']}-{doc['question']}-{doc['text'][:10]}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    document_id = hash_hex[:8]
    return document_id

def transform(data):
    print(data['course'])

    documents = []

    for doc in data['documents']:
        doc['course'] = data['course']
        # previously we used just "id" for document ID
        doc['document_id'] = generate_document_id(doc)
        documents.append(doc)

    print(len(documents))

    return documents

documents = transform(data[0])

# Q4. Export
def index_generation():
    index_name_prefix = 'documents'
    current_time = datetime.now().strftime("%Y%m%d_%M%S")
    return f"{index_name_prefix}_{current_time}"

index_name = index_generation()

def index_elasticsearch(documents, index_name):
    connection_string = 'http://localhost:9200'
    print("index name:", index_name)

    es_client = Elasticsearch(connection_string)
    print(f'Connecting to Elasticsearch at {connection_string}')
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
                "document_id": {"type": "keyword"}
            }
        }
    }

    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name)
        print('Index created with properties:', index_settings)

    print(f'Indexing {len(documents)} documents to Elasticsearch index {index_name}')
    for document in documents:
        print(f'Indexing document {document["document_id"]}')
        doc = document
        es_client.index(index=index_name, document=document)

    print(doc)

index_elasticsearch(documents, index_name)

# Q5. Testing the retrieval
from elasticsearch import Elasticsearch
es = Elasticsearch('http://localhost:9200')

def search_query(query):
    return {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^4", "text", "section"],
                            "type": "best_fields"
                        }
                    }
                ],
            }
        },
        "size": 10
    }

query = 'When is the next cohort?'
result = es.search(index=index_name, body=search_query(query))
print(result["hits"]["hits"][0]['_source']['document_id'])

# Q6
def load_data_1():
    faq_documents = {
        'llm-zoomcamp': '1T3MdwUvqCL3jrh3d3VCXQ8xE0UqRzI3bfgpfBq3ZWG0',
    }
    documents = []

    for course, file_id in faq_documents.items():
        print(course)
        course_documents = read_faq(file_id)
        documents.append({'course': course, 'documents': course_documents})

    return documents

data = load_data_1()
documents = transform(data[0])
index_name = index_generation()
index_elasticsearch(documents, index_name)
result = es.search(index=index_name, body=search_query(query))
print(result["hits"]["hits"][0]['_source']['document_id'])
