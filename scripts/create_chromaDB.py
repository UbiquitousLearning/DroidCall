import chromadb
import json
from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers.utils import get_json_schema
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

import os
os.sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../")

from utils.retriever import GTEEmbedding
import chromadb.utils.embedding_functions as ef
from utils.formatter import FunctionFormatter

    
chroma_client = chromadb.PersistentClient(path="./chromaDB")

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(
    name="functions",
    metadata={"hnsw:space": "ip"}, # l2 is the default
    embedding_function=GTEEmbedding(path="/data/share/gte-small")
)

formatter = FunctionFormatter(format_type="code")

if __name__ == "__main__":
    apis = []
    
    with open("data/api.jsonl", "r") as f:
        apis = [json.loads(line) for line in f]
    
    docs = [formatter.format(tools=[api]) for api in apis]
    
    emb = GTEEmbedding(path="/data/share/gte-small")
    
    embeddings = emb(docs)
    with open("api_vec.jsonl", "w") as f:
        for api, emb in zip(apis, embeddings):
            item = {}
            item["embedding"] = emb
            item["doc"] = json.dumps(api, ensure_ascii=False)
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    

    # switch `add` to `upsert` to avoid adding the same documents every time
    # with open("data/api.jsonl", "r") as f:
    #     for line in f:
    #         item = json.loads(line)
    #         id = f"{item['name']}"
    #         doc = formatter.format(tools=[item])
    #         json_str = json.dumps(item, ensure_ascii=False)
    #         # doc = json_str

    #         collection.upsert(
    #             documents=[
    #                 doc
    #             ],
    #             ids=[id],
    #             metadatas=[{"json_str": json_str}]
    #         )

    # results = collection.query(
    #     query_texts="Dial Anne", # Chroma will embed this for you
    #     n_results=2
    # )

    
    # for meta in results["metadatas"][0]:
    #     print(f"doc: {meta}\n\n\n")
