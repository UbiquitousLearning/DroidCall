import json
from typing import List
import chromadb
import random
from utils import GenerateResponse
from string import Template
from utils.extract import get_json_obj


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
            [
                f"name: {api['name']}\ndescription: {api['description'].strip().splitlines()[0]}\n\n" 
                for api in self.apis.values()
            ]
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

from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoTokenizer, AutoModel
from torch import Tensor

class GTEEmbedding(EmbeddingFunction):
    def __init__(self, path: str, device: str="cpu"):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, device_map=device)
    
    @staticmethod
    def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def __call__(self, input_texts: Documents) -> Embeddings:
        # Tokenize the input texts
        batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**batch_dict.to(self.model.device))
        # tokens = self.tokenizer.tokenize(input_texts[0])
        # print(f"tokens: {tokens}\n\n")
        # print(f"{batch_dict.input_ids.tolist()}\n\n")
        embeddings = GTEEmbedding.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        return embeddings.tolist()

class ChromaDBRetriever(Retriever):
    def __init__(self, data_path: str, name: str="functions", distance_type: str="l2", emb_func=None) -> None:
        super().__init__()
        self.client = chromadb.PersistentClient(path=data_path)
        if not emb_func:
            self.collection = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": distance_type}, # l2 is the default
            )
        else:
            self.collection = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": distance_type}, # l2 is the default
                embedding_function=emb_func
            )
    
    def retrieve(self, query: str, n_results: int) -> List[str]:
        results = self.collection.query(
            query_texts=query,
            n_results=n_results,
        )
        metas = results["metadatas"][0]
        documents = [
            json.dumps(json.loads(meta["json_str"]), indent=2, ensure_ascii=False)
            for meta in metas
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
    