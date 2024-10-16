import json
from typing import List
import chromadb
import random
from utils import GenerateResponse
from string import Template
from utils import get_json_obj


RETRIEVE_PROMPT = Template("""
You are supposed to help me find the relevant APIs for the given query.
I will give you a series of API names and descriptions with a query, you shoul help me
pick the top $n_results APIs that are relevant to the query.

Below is the APIs:
$apis

The query is:
$query

You should give me the top $n_results APIs that are relevant to the query in a json list like ["api1", "api2", ...]
REMEMBER TO STRICTLY FOLLOW THE FORMAT, AND GIVE THE CORRECT API NAME.
ALSO REMEMBER YOU SHOULD GIVE $n_results APIs BASE ON THE RELEVANCE.
""")

class Retriever:
    def retrieve(self, query: str, n_results: int) -> List[str]:
        pass
    
class LLMRetriever(Retriever):
    def __init__(self, data_path: str, llm: GenerateResponse):
        self.llm = llm
        self.apis = {}
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.apis[item["name"]] = item
        
        self.apis_text = "\n".join(
            [f"name: {api['name']}\ndescription: {api['description'].strip().split("\n")[0]}\n\n" for api in self.apis.values()]
        )
        
        # print(self.apis_text)    
    
        
    def retrieve(self, query: str, n_results: int) -> List[str]:
        user_message = RETRIEVE_PROMPT.substitute(
            apis=self.apis_text,
            query=query,
            n_results=n_results,
        )
        
        resp = self.llm('', [user_message], max_new_tokens=500)[0]
        # print(f"response: {resp["text"]}\n")
        apis = get_json_obj(resp["text"])
        documents = [
            json.dumps(self.apis[name]) for name in apis if name in self.apis
        ]
        return documents
                
    

class ChromaDBRetriever(Retriever):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.client = chromadb.PersistentClient(path=data_path)
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
    