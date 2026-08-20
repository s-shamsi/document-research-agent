import requests

response = requests.post(
    "http://localhost:8787/api/research",
    json={"request": "What is machine learning?"},
    stream=True
)

print("Status:", response.status_code)
for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    print(chunk, end="", flush=True)
print()