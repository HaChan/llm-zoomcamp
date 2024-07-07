# Question 1 Ollama docker version

```
docker exec -it ollama ollama -v
```

# Question 2 LLM manifest file content

```
docker exec -it ollama cat /root/.ollama/models/manifests/registry.ollama.ai/library/gemma/2b
```

# Question 3 Running the LLM

```
docker exec -it ollama ollama run gemma:2b
```

# Question 4 Donwloading the weights

```
du -h ./ollama_files
```

# Question 5 Adding the weights

```
FROM ollama/ollama

COPY ./ollama_files /root/.ollama
```

# Q6. Serving it

```python
from openai import OpenAI
client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

prompt = "What's the formula for energy?"

response = client.chat.completions.create(
    model='gemma:2b',
    temperature=0.0,
    messages=[{"role": "user", "content": prompt}]
)

print(response.usage.completion_tokens)
```
