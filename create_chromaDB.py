import chromadb
import json
from transformers.utils import get_json_schema
chroma_client = chromadb.PersistentClient(path="./chromaDB")

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="intents")

# switch `add` to `upsert` to avoid adding the same documents every time
with open("data/intents.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        id = f"{item['id']}_{item['action']}"

        collection.upsert(
            documents=[
                line
            ],
            ids=[id]
        )

results = collection.query(
    query_texts=["I want to set an alarm at 7:30"], # Chroma will embed this for you
    n_results=2
)

print(results)
