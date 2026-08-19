import requests

# Reads file as raw binary, uploads via sources endpoint, prints confirmation
files = {'files': open('test.txt', 'rb')}
response = requests.post('http://localhost:8787/api/sources', files=files)
print(response.status_code)
print(response.text)