import pandas as pd
github_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv"
url = f'{github_url}?raw=1'
df = pd.read_csv(url)
df = df.iloc[:300]

# Q1 Getting the embeddings model
from sentence_transformers import SentenceTransformer
model_name = "multi-qa-mpnet-base-dot-v1"
embedding_model = SentenceTransformer(model_name)

answer = df.iloc[0].answer_llm
ev = embedding_model.encode(answer)
print(ev[0])

# Q2 Computing the dot product
import numpy as np

def compute_similiarity(data):
    answer_llm_ev = embedding_model.encode(row['answer_llm'])
    answer_ev = embedding_model.encode(row['answer_orig'])
    return answer_llm_ev.dot(answer_ev)

evaluations = []

for i, row in df.iterrows():
    evaluations.append(compute_similiarity(row))

p75 = np.percentile(evaluations, 75)
print(p75)

# Q3 Computing the cosine
evaluations_q3 = []
def norm_vector(v):
    norm = np.sqrt((v * v).sum())
    return v / norm

def compute_normalize_similiarity(data):
    answer_llm_ev = norm_vector(embedding_model.encode(row['answer_llm']))
    answer_ev = norm_vector(embedding_model.encode(row['answer_orig']))
    return answer_llm_ev.dot(answer_ev)

for i, row in df.iterrows():
    evaluations_q3.append(compute_normalize_similiarity(row))

p75 = np.percentile(evaluations_q3, 75)
print(p75)

# Q4 Rouge
from rouge import Rouge
rouge_scorer = Rouge()

row = df.iloc[10]
scores = rouge_scorer.get_scores(row['answer_llm'], row['answer_orig'])[0]
print(scores['rouge-1']['f'])

# Q5 Avg rouge score
print((scores['rouge-1']['f'] + scores['rouge-2']['f'] + scores['rouge-l']['f']) / 3)

# Q6 Average rouge score for all the data points
def compute_rouge2_f(row):
    scores = rouge_scorer.get_scores(row['answer_llm'], row['answer_orig'])[0]
    return scores['rouge-2']['f']

df['rouge-2-f'] = df.apply(compute_rouge2_f, axis=1)
print(df['rouge-2-f'].mean())
