import requests

base_url = "http://localhost:9898/"

def update(table: str, model: str, task: str, value: float):
    url = f"{base_url}/add"
    payload = {
        'table': table,
        'model': model,
        'task_name': task,
        'value': value
    }
    response = requests.post(url, json=payload)
    return response.json()

