import chromadb
import json
from transformers.utils import get_json_schema
chroma_client = chromadb.PersistentClient(path="./chromaDB")

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="functions")

# switch `add` to `upsert` to avoid adding the same documents every time
with open("data/api.jsonl", "r") as f:
    for line in f:
        item = json.loads(line)
        id = f"{item['name']}"
        doc = json.dumps(item, indent=2, ensure_ascii=False)

        collection.upsert(
            documents=[
                doc
            ],
            ids=[id]
        )

results = collection.query(
    query_texts=["I want to check Alice's phone"], # Chroma will embed this for you
    n_results=4
)

for doc in results["documents"][0]:
    print(doc)
