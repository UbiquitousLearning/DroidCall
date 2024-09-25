import json
from typing import List
import chromadb
import random


class Retriever:
    def retrieve(self, query: str, n_results: int) -> List[str]:
        pass
    

class ChromaDBRetriever(Retriever):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.client = chromadb.PersistentClient(path="./chromaDB")
        self.collection = self.client.get_or_create_collection('functions')
    
    def retrieve(self, query: str, n_results: int) -> List[str]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        docs = results['documents'][0]
        documents = [
            json.dumps(json.loads(doc), indent=2, ensure_ascii=False)
            for doc in docs
        ]
        return documents


class FakeRetriever(Retriever):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.query_to_functions = {}
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.query_to_functions[item["query"]] = [d["name"] for d in item["answers"]]
                
        self.api_info = {}
        with open("data/api.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                self.api_info[item["name"]] = item
    
    def retrieve(self, query: str, n_results: int) -> List[str]:
        # retrieve n actual intent and n_results - n fake intents
        actual_functions = self.query_to_functions[query]
        fake_functions = list(self.api_info.keys())
        fake_functions = [f for f in fake_functions if f not in actual_functions]
        if len(actual_functions) > n_results:
            fake_functions = []
        else:
            fake_functions = random.sample(fake_functions, n_results - len(actual_functions))
        
        all_functions = actual_functions + fake_functions
        documents = [
            json.dumps(self.api_info[func], indent=2, ensure_ascii=False)
            for func in all_functions
        ]
        
        return documents
    