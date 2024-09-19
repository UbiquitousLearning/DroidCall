import chromadb
import json
from typing import List
from openai import OpenAI
from string import Template
from tqdm import tqdm
import os
from utils import get_json_obj, extract_and_parse_jsons
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import random
from peft import PeftModelForCausalLM
from utils import Colors

SYSTEM_PROMPT = """
You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. You should only return the function call in tools call sections.
"""

NEST_PROMT = """
If an argument is a response from a previous function call, you can reference it in the following way like the argument value of arg2 in func3:
{
    "name": "func3",
    "arguments": {
        "arg1": "value1",
        "arg2": "@func2",
        ...
    }
}
This means that the value of arg2 in func3 is the response from func2.
"""

PROMPT_FOR_CHATMODEL = Template("""
Here is a list of functions in JSON format that you can invoke:
$functions

Should you decide to return the function call(s), Put it in the format of 
[
    {
        "name": "func1",
        "arguments": {
            "arg1": "value1",
            "arg2": "value2",
            ...
        }
    },
    {
        "name": "func2",
        "arguments": {
            "arg1": "value1",
            "arg2": "value2",
            ...
        }
    },
    ...
]

$nest_prompt

$example

If there is a way to achieve the purpose using the given functions, please provide the function call(s) in the above format.
REMEMBER TO ONLY RETURN THE FUNCTION CALLS LIKE THE EXAMPLE ABOVE, NO OTHER INFORMATION SHOULD BE RETURNED.

Now my query is: $user_query
""")

class Handler:
    model_name: str

    def __init__(self, model_name, path, adapter_path, temperature=0.7, top_p=1, max_tokens=1000,
                 is_nested: bool=False, add_examples: bool = False) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.path = path
        self.adapter_path = adapter_path
        self.is_nested = is_nested
        self.add_examples = add_examples
        
    def format_message(self, user_query: str, documents: List[str], is_nested: bool=False, 
                       add_examples: bool = False) -> str:
        nest_prompt = NEST_PROMT if is_nested else ""
        example_text = ""
        if add_examples:
            sampled_examples = []
            if not hasattr(self, "examples"):
                with open("data/instructions_train.jsonl", "r") as f:
                    examples = [json.loads(line) for line in f]
                self.examples = examples
            for doc in documents:
                doc = json.loads(doc)
                func_name = doc["name"]
                for example in self.examples:
                    ok = False
                    for ans in example["answers"]:
                        if ans["name"] == func_name:
                            ok = True
                            break
                    if ok:
                        sampled_examples.append(example)
                        break
            
            example_text =  "Here is some examples:\n" + "\n".join(f"query: {example["query"]} \nanwsers: {json.dumps(example["answers"], ensure_ascii=False, indent=2)}" for example in sampled_examples)
                        
        message = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": PROMPT_FOR_CHATMODEL.substitute(user_query=user_query, functions="\n".join(documents), nest_prompt=nest_prompt, example=example_text)
            },
        ]
        
        # print(f"{Colors.BOLD}message: {json.dumps(message, indent=2, ensure_ascii=False)}{Colors.ENDC}")
        return message

    def inference(self, user_query: str, documents: List[str]) -> str:
        # This method is used to retrive model response for each model.
        pass
    

class OpenAIHandler(Handler):
    def __init__(self, model_name, path, adapter_path, temperature=0.7, top_p=1, max_tokens=1000,
                 is_nested: bool=False, add_examples: bool = False) -> None:
        super().__init__(model_name, path, adapter_path, temperature, top_p, max_tokens, is_nested, add_examples)
        self.client = OpenAI()

    def inference(self, user_query: str, documents: List[str]) -> str:
        # This method is used to retrive model response for each model.
        message = self.format_message(user_query, documents, self.is_nested, self.add_examples)
        # print(message)
        response = self.client.chat.completions.create(
            messages=message,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        return response.choices[0].message.content

class DeepseekHandler(OpenAIHandler):
    def __init__(self, model_name, path, adapter_path, temperature=0.7, top_p=1, max_tokens=1000,
                 is_nested: bool=False, add_examples: bool = False) -> None:
        super().__init__(model_name, path, adapter_path, temperature, top_p, max_tokens, is_nested, add_examples)
        self.client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", None),
                             base_url="https://api.deepseek.com/")
    

class HFCausalLMHandler(Handler):
    def __init__(self, model_name, path, adapter_path, temperature=0.7, top_p=1, max_tokens=1000,
                 is_nested: bool=False, add_examples: bool = False) -> None:
        super().__init__(model_name, path, adapter_path, temperature, top_p, max_tokens, is_nested, add_examples)
        
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True)
        
    def inference(self, user_query: str, documents: List[str]) -> str:
        # This method is used to retrive model response for each model.
        message = self.format_message(user_query, documents, self.is_nested, self.add_examples)
        
        tokenized_chat = self.tok.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        if self.temperature > 0:
            outputs = self.model.generate(tokenized_chat.to(self.model.device), 
                                        max_new_tokens=self.max_tokens, 
                                        top_p=self.top_p, temperature=self.temperature,
                                        do_sample=True)
        else:
            outputs = self.model.generate(tokenized_chat.to(self.model.device), 
                                        max_new_tokens=self.max_tokens, 
                                        do_sample=False)
        text = self.tok.decode(outputs[0])
        prefix = self.tok.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        response = text[len(prefix):]
        return response
    

class LoraCausalLMHandler(HFCausalLMHandler):
    def __init__(self, model_name, path, adapter_path, temperature=0.7, top_p=1, max_tokens=1000,
                 is_nested: bool=False, add_examples: bool = False) -> None:
        super().__init__(model_name, path, adapter_path, temperature, top_p, max_tokens, is_nested, add_examples)
        self.base_model = self.model
        self.model = PeftModelForCausalLM.from_pretrained(self.base_model, adapter_path)


HANDLER_MAP = {
    "openai": OpenAIHandler,
    "hf_causal_lm": HFCausalLMHandler,
    "lora_causal_lm": LoraCausalLMHandler,
    "deepseek": DeepseekHandler
}

parser = argparse.ArgumentParser(description='Generate solution for the task')
parser.add_argument('--input', type=str, default='./data/instructions_test.jsonl', help='Path to the input file')
parser.add_argument('--retrieve_doc_num', type=int, default=2, help='Number of documents to retrieve')
parser.add_argument('--model_name', type=str, default='gpt-4o-mini', help='model name')
parser.add_argument('--handler', type=str, default='openai', help='Handler to use for inference',
                    choices=["openai", "hf_causal_lm", "lora_causal_lm", "deepseek"])
parser.add_argument('--path', type=str, default="/data/share/Qwen2-1.5B-Instruct", help='local dir if model is in local')
parser.add_argument('--adapter_path', type=str, default="./checkpoint/Qwen2-1.5B-Instruct", help='adapter path')
parser.add_argument('--task_name', type=str, default='', help='task name')
parser.add_argument('--retriever', type=str, default='fake', help='retriever to use', choices=["chromadb", "fake"])
parser.add_argument('--temperature', type=float, default=0.0, help='temperature for generation')
parser.add_argument('--top_p', type=float, default=1, help='top_p for generation')
parser.add_argument('--max_tokens', type=int, default=500, help='max tokens for generation')
parser.add_argument('--is_nested', action="store_true", help='use nested function calling or not')
parser.add_argument('--add_examples', action="store_true", help='add examples in the prompt or not')
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
    

RETRIEVER_MAP = {
    "chromadb": ChromaDBRetriever,
    "fake": FakeRetriever
}

def test_retriever_accuracy():
    print(f"Testing retriever({arg.retriever}) accuracy")
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
            actual_functions = item["answers"]
            
            documents = retriever.retrieve(query, n_doc)
            documents_function_names = [json.loads(doc)["name"] for doc in documents]
            # check if all actual functions are in the retrieved documents
            if all([f["name"] in documents_function_names for f in actual_functions]):
                correct += 1
        
        accuracy = correct / len(all_items)
        print(f"Accuracy for {n_doc} documents: {accuracy}")
        data["n_docs"].append(n_doc)
        data["accuracy"].append(accuracy)
        
    import pandas as pd
    df = pd.DataFrame(data)
    print(df)
            
def check_format(ans):
    if not isinstance(ans, dict):
        return False
    if "name" not in ans or "arguments" not in ans:
        return False
    if not isinstance(ans["arguments"], dict):
        return False
    return True

def main():
    handler = HANDLER_MAP[HANDLER](MODEL_NAME, arg.path, arg.adapter_path, arg.temperature, arg.top_p, arg.max_tokens,
                                   arg.is_nested, arg.add_examples)
    
    all_instructions = []
    with open(arg.input, "r") as f:
        for line in f:
            j = json.loads(line)
            all_instructions.append(j)
    
    # create output directory if not exists
    if not os.path.exists("./results"):
        os.makedirs("./results")
    output_file = open(f"./results/{HANDLER}_{MODEL_NAME}_{arg.task_name}_result.jsonl", "w")
    
    retriever = RETRIEVER_MAP[arg.retriever](arg.input)
    
    for instruction in tqdm(all_instructions):
        query = instruction["query"]
        documents = retriever.retrieve(query, arg.retrieve_doc_num)
        
        retry_num = 4
        
        while retry_num > 0:
            response = handler.inference(query, documents)
            print(f"{Colors.OKGREEN}response: {response}{Colors.ENDC}\n\n")
            res = [call for call in extract_and_parse_jsons(response)]
            if res and all([check_format(ans) for ans in res]):
                break
            retry_num -= 1
            if arg.temperature <= 0:
                break
            
        output_file.write(json.dumps({"query": query, "response": res, "answers": instruction["answers"]}, ensure_ascii=False) + "\n")
        output_file.flush()
        
    output_file.close()


if __name__ == '__main__':
    # test_retriever_accuracy()
    main()
    
    