import chromadb
import json
from typing import List
from openai import OpenAI
from string import Template
from tqdm import tqdm
import os
from utils import get_json_obj
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import random
from peft import PeftModelForCausalLM


PROMPT_FOR_CHATMODEL = Template("""
You are an Android development expert, skilled in understanding user needs and you are very familiar with Android intents.
Now you will be given a user query and a list of relevant intents that may be related to the query. You need to 
Based on the user query, you should select proper intent that can achieve the user's goal. If none of the intents can be used,
point it out. You should only return the intent and the solution to the user query. You should not provide any additional information.
                                
If you can use one of the intent to solve the user query, please provide the intent and the solution in the following format:
{
    "intent": ... ,
    "uri": ... ,
    "mime": ... ,
    "extras": {
        ...
    },
}

Below is an example:

User query: I need to set an alarm for 10:30 PM every Sunday to remember to prepare for next week's work.
{
    "intent": "ACTION_SET_ALARM",
    "uri": "",
    "mime": "",
    "extras": {
        "EXTRA_HOUR": 22,
        "EXTRA_MINUTES": 30,
        "EXTRA_MESSAGE": "Prepare for Work",
        "EXTRA_DAYS": [
            1
        ],
        "EXTRA_RINGTONE": "",
        "EXTRA_VIBRATE": true
    },
}
If some field is not applicable, you can leave it empty. Remember to only provide the intent and the solution in the specific format. Do not provide any additional information.
                                
Now I will give you a user query and a list of intents. You should provide the intent and the solution to the user query.
REMEMBER YOU MUST GIVE JUST AN JSON STRICTLY FOLLOWING THE FORMAT I GAVE YOU ABOVE AS OUTPUT. NOTHING ELSE.

User query: $user_query
Here are the intents you can use to solve the user query:
$intents_info
""")

class Handler:
    model_name: str

    def __init__(self, model_name, path, adapter_path, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.path = path
        self.adapter_path = adapter_path

    def inference(self, user_query: str, documents: List[str]) -> str:
        # This method is used to retrive model response for each model.
        pass
    

class OpenAIHandler(Handler):
    def __init__(self, model_name, path, adapter_path, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, path, adapter_path, temperature, top_p, max_tokens)
        self.client = OpenAI()

    def inference(self, user_query: str, documents: List[str]) -> str:
        # This method is used to retrive model response for each model.
        message = [
            {
                "role": "user",
                "content": PROMPT_FOR_CHATMODEL.substitute(user_query=user_query, intents_info="\n".join(documents))
            },
        ]
        # print(message)
        response = self.client.chat.completions.create(
            messages=message,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        return response.choices[0].message.content


class HFCausalLMHandler(Handler):
    def __init__(self, model_name, path, adapter_path, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, path, adapter_path, temperature, top_p, max_tokens)
        
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
        
    def inference(self, user_query: str, documents: List[str]) -> str:
        # This method is used to retrive model response for each model.
        message = [
            {
                "role": "user",
                "content": PROMPT_FOR_CHATMODEL.substitute(user_query=user_query, intents_info="\n".join(documents))
            },
        ]
        
        tokenized_chat = self.tok.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        outputs = self.model.generate(tokenized_chat.to(self.model.device), 
                                       max_new_tokens=self.max_tokens, 
                                       top_p=self.top_p, temperature=self.temperature,
                                       do_sample=True)
        text = self.tok.decode(outputs[0])
        prefix = self.tok.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
        response = text[len(prefix):]
        return response
    

class LoraCausalLMHandler(HFCausalLMHandler):
    def __init__(self, model_name, path, adapter_path, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, path, adapter_path, temperature, top_p, max_tokens)
        self.base_model = self.model
        self.model = PeftModelForCausalLM.from_pretrained(self.base_model, adapter_path)


HANDLER_MAP = {
    "openai": OpenAIHandler,
    "hf_causal_lm": HFCausalLMHandler,
    "lora_causal_lm": LoraCausalLMHandler
}

parser = argparse.ArgumentParser(description='Generate solution for the task')
parser.add_argument('--input', type=str, default='./data/filtered_data.jsonl', help='Path to the input file')
parser.add_argument('--retrieve_doc_num', type=int, default=2, help='Number of documents to retrieve')
parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='model name')
parser.add_argument('--handler', type=str, default='openai', help='Handler to use for inference',
                    choices=["openai", "hf_causal_lm", "lora_causal_lm"])
parser.add_argument('--path', type=str, default="/data/share/Qwen2-1.5B-Instruct", help='local dir if model is in local')
parser.add_argument('--adapter_path', type=str, default="./checkpoint/Qwen2-1.5B-Instruct", help='adapter path')
parser.add_argument('--task_name', type=str, default='', help='task name')
parser.add_argument('--retriever', type=str, default='fake', help='retriever to use', choices=["chromadb", "fake"])
arg = parser.parse_args()


HANDLER = arg.handler # "openai"
MODEL_NAME = arg.model_name # "gpt-4o-mini"

class Retriever:
    def retrieve(self, query: str, n_results: int) -> List[str]:
        pass
    

class ChromaDBRetriever(Retriever):
    def __init__(self, data_path: str) -> None:
        super().__init__()
        self.client = chromadb.PersistentClient(path="./chromaDB")
        self.collection = self.client.get_or_create_collection('intents')
    
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
        self.query_to_intents = {}
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.query_to_intents[item["query"]] = item["intent"]
                
        self.intents_info = {}
        with open("data/intents.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                self.intents_info[item["action"]] = item
    
    def retrieve(self, query: str, n_results: int) -> List[str]:
        # retrieve 1 actual intent and n_results - 1 fake intents
        actual_intent = self.query_to_intents[query]
        fake_intents = list(self.intents_info.keys())
        fake_intents.remove(actual_intent)
        fake_intents = random.sample(fake_intents, n_results - 1)
        
        all_intents = [actual_intent] + fake_intents
        documents = [
            json.dumps(self.intents_info[intent], indent=2, ensure_ascii=False)
            for intent in all_intents
        ]
        
        return documents
    

RETRIEVER_MAP = {
    "chromadb": ChromaDBRetriever,
    "fake": FakeRetriever
}

def test_retriever_accuracy():
    print("Testing retriever accuracy")
    retriever = RETRIEVER_MAP[arg.retriever](arg.input)
    all_items = []
    with open(arg.input, "r") as f:
        for line in f:
            item = json.loads(line)
            all_items.append(item)
    
    data = {
        "n_docs": [],
        "accuracy": []
    }
    
    max_n_docs = 10
    for n_doc in range(1, max_n_docs + 1):
        correct = 0
        for item in all_items:
            query = item["query"]
            actual_intent = item["intent"]
            
            documents = retriever.retrieve(query, n_doc)
            for doc in documents:
                doc_obj = json.loads(doc)
                if doc_obj["action"] == actual_intent:
                    correct += 1
                    break
        
        accuracy = correct / len(all_items)
        print(f"Accuracy for {n_doc} documents: {accuracy}")
        data["n_docs"].append(n_doc)
        data["accuracy"].append(accuracy)
        
    import pandas as pd
    df = pd.DataFrame(data)
    print(df)
            


def main():
    handler = HANDLER_MAP[HANDLER](MODEL_NAME, arg.path, arg.adapter_path)
    
    queries = []
    with open(arg.input, "r") as f:
        for line in f:
            j = json.loads(line)
            user_query = j["query"]
            queries.append(user_query)
    
    # create output directory if not exists
    if not os.path.exists("./results"):
        os.makedirs("./results")
    output_file = open(f"./results/{HANDLER}_{MODEL_NAME}_{arg.task_name}_result.jsonl", "w")
    
    retriever = RETRIEVER_MAP[arg.retriever](arg.input)
    
    for query in tqdm(queries):
        documents = retriever.retrieve(query, arg.retrieve_doc_num)
        
        retry_num = 4
        
        while retry_num > 0:
            response = handler.inference(query, documents)
            res = get_json_obj(response)
            if res:
                break
            retry_num -= 1
            
        output_file.write(json.dumps({"query": query, "response": res}, ensure_ascii=False) + "\n")
        output_file.flush()
        
    output_file.close()


if __name__ == '__main__':
    # test_retriever_accuracy()
    main()
    # path = "/data/share/Qwen2-1.5B-Instruct"
    
    # tokenizer = AutoTokenizer.from_pretrained(path)
    # model = AutoModelForCausalLM.from_pretrained(path, device_map="auto")
    
    # message = [
    #     {
    #         "role": "user",
    #         "content": "How can I keep fit"
    #     }
    # ]
    
    # tokenized_chat = tokenizer.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    # outputs = model.generate(tokenized_chat, max_new_tokens=1000, top_p=1, temperature=0.7, do_sample=True)
    # print(tokenizer.decode(outputs[0]))

    
    