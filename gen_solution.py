import torch
import json
from typing import List
from openai import OpenAI
from string import Template
from tqdm import tqdm
import os
from utils.extract import extract_and_parse_jsons
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import random
from utils import Colors
from utils.extract import get_json_obj
from utils.prompt import SYSTEM_PROMPT_FOR_FUNCTION_CALLING, JSON_NESTED_CALLING_PROMT, FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL, JSON_CALL_FORMAT
from utils.formatter import *

SYSTEM_PROMPT = SYSTEM_PROMPT_FOR_FUNCTION_CALLING

NEST_PROMT = JSON_NESTED_CALLING_PROMT

PROMPT_FOR_CHATMODEL = Template(FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL)

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
        self.format_type = "json"
        self.sep_start = ""
        self.sep_end = ""
    
    def set_format_type(self, format_type: str):
        self.format_type = format_type
        
    def set_sep(self, sep_start: str, sep_end: str):
        self.sep_start = sep_start
        self.sep_end = sep_end
        
    _SYSTEM_PROMPT_MAP = {
        "json": SYSTEM_PROMPT_FOR_FUNCTION_CALLING,
        "code": SYSTEM_PROMPT_FOR_FUNCTION_CALLING,
        "code_short": SHORT_SYSTEM_PROMPT_FOR_FUNCTION_CALLING,
        "json_short": SHORT_SYSTEM_PROMPT_FOR_FUNCTION_CALLING
    }
        
    _CALL_PROMPT_MAP = {
        "json": FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL,
        "code": FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL,
        "code_short": SHORT_FUNCTION_CALLING_PROMPT,
        "json_short": FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL
    }
    
    _CALL_FORMAT_MAP = {
        "json": ConstantCallingFormatter(JSON_CALL_FORMAT),
        "code": ConstantCallingFormatter(CODE_CALL_FORMAT),
        "code_short": ConstantFormatter(""),
        "json_short": ConstantFormatter("")
    }
    
    _NEST_CALL_MAP = {
        "json": ConstantFormatter(JSON_NESTED_CALLING_PROMT),
        "code": ConstantFormatter(CODE_NESTED_CALLING_PROMPT),
        "code_short": ConstantFormatter(""),
        "json_short": ConstantFormatter("")
    }
    
    _FUNCTION_CALL_MAP = {
        "json": JsonFunctionCallingFormatter(),
        "code": CodeFunctionCallingFormatter(),
        "code_short": CodeFunctionCallingFormatter(),
        "json_short": JsonFunctionCallingFormatter()
    }
        
    def format_message(self, user_query: str, documents: List[str], is_nested: bool=False, 
                       add_examples: bool = False) -> str:
        func_format_type = self.format_type
        if self.format_type in ["code", "code_short"]:
            func_format_type = "code"
        if self.format_type in ["json", "json_short"]:
            func_format_type = "json"
            
        user_formatter= Formatter(
            Handler._CALL_PROMPT_MAP[self.format_type],
            functions=FunctionFormatter(func_format_type),
            call_format=Handler._CALL_FORMAT_MAP[self.format_type],
            nest_prompt=Handler._NEST_CALL_MAP[self.format_type] if is_nested else ConstantFormatter(""),
            example=GetFunctionExampleFormatter("data/DroidCall_train.jsonl", Handler._FUNCTION_CALL_MAP[self.format_type]) if add_examples else ConstantFormatter(""),
            user_query=FieldFormatter("query"),
        )
        
        if isinstance(user_formatter.call_format, FunctionCallingFormatter):
            user_formatter.call_format.set_sep(self.sep_start, self.sep_end)
        if isinstance(user_formatter.example, GetFunctionExampleFormatter):
            user_formatter.example.call_formatter.set_sep(self.sep_start, self.sep_end)
        
        tools = [json.loads(doc) for doc in documents]
        user_message = user_formatter.format(
            query=user_query,
            tools=tools
        )
        
                   
        message = [
            {
                "role": "system",
                "content": Handler._SYSTEM_PROMPT_MAP[self.format_type]
            },
            {
                "role": "user",
                "content": user_message # PROMPT_FOR_CHATMODEL.substitute(user_query=user_query, functions="\n".join(documents), nest_prompt=nest_prompt, example=example_text, call_format=JSON_CALL_FORMAT)
            },
        ]
        
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
    total_tokens = 0
    input_tokens = 0
    inference_count = 0
    
    def __init__(self, model_name, path, adapter_path, temperature=0.7, top_p=1, max_tokens=1000,
                 is_nested: bool=False, add_examples: bool = False) -> None:
        super().__init__(model_name, path, adapter_path, temperature, top_p, max_tokens, is_nested, add_examples)
        
        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True,
                                                          torch_dtype=torch.bfloat16)
        
    def inference(self, user_query: str, documents: List[str]) -> str:
        # This method is used to retrive model response for each model.
        message = self.format_message(user_query, documents, self.is_nested, self.add_examples)
        
        # we modify gemma-2-2b-it's prompt template to make it compatible with system prompt
        # if "gemma-2-2b-it" in self.model_name:
        #     message = message[1:] # gemmma-2-2b-it not support system prompt
        
        tokenized_chat = self.tok.apply_chat_template(message, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        # count input tokens
        self.input_tokens += tokenized_chat.size(1)
        
        if self.temperature > 0:
            outputs = self.model.generate(tokenized_chat.to(self.model.device), 
                                        max_new_tokens=self.max_tokens, 
                                        top_p=self.top_p, temperature=self.temperature,
                                        do_sample=True)
        else:
            self.model.generation_config.temperature=None
            self.model.generation_config.top_p=None
            self.model.generation_config.top_k=None
            outputs = self.model.generate(tokenized_chat.to(self.model.device), 
                                        max_new_tokens=self.max_tokens, 
                                        do_sample=False, temperature=0, top_p=None, top_k=None)
        
        text = self.tok.decode(outputs[0][len(tokenized_chat[0]):], skip_special_tokens=True)
        # count total tokens
        self.total_tokens += outputs.size(1)
        self.inference_count += 1
        prefix = self.tok.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        print(f"{Colors.BOLD} {prefix}\n\n{Colors.ENDC}")
        return text
    

class LoraCausalLMHandler(HFCausalLMHandler):
    def __init__(self, model_name, path, adapter_path, temperature=0.7, top_p=1, max_tokens=1000,
                 is_nested: bool=False, add_examples: bool = False) -> None:
        super().__init__(model_name, path, adapter_path, temperature, top_p, max_tokens, is_nested, add_examples)
        self.base_model = self.model
        from peft import PeftModelForCausalLM
        self.model = PeftModelForCausalLM.from_pretrained(self.base_model, adapter_path)


HANDLER_MAP = {
    "openai": OpenAIHandler,
    "hf_causal_lm": HFCausalLMHandler,
    "lora_causal_lm": LoraCausalLMHandler,
    "deepseek": DeepseekHandler
}

parser = argparse.ArgumentParser(description='Generate solution for the task')
parser.add_argument('--input', type=str, default='./data/DroidCall_test.jsonl', help='Path to the input file')
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
parser.add_argument('--format_type', type=str, default="json", help='format type for the prompt', choices=["json", "code", "code_short", "json_short"])
parser.add_argument('--sep_start', type=str, default="", help='start separator for function call')
parser.add_argument('--sep_end', type=str, default="", help='end separator for function call')
arg = parser.parse_args()


HANDLER = arg.handler # "openai"
MODEL_NAME = arg.model_name # "gpt-4o-mini"

from utils.retriever import ChromaDBRetriever, FakeRetriever, Retriever

RETRIEVER_MAP = {
    "chromadb": ChromaDBRetriever,
    "fake": FakeRetriever
}

from utils.extract import *

CALL_EXTRACTOR_MAP: Dict[str, CallExtractor] = {
    "json": JsonCallExtractor(),
    "code": CodeCallExtractor(),
    "code_short": CodeCallExtractor(),
    "json_short": JsonCallExtractor()
}

            
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
    handler.set_format_type(arg.format_type)
    handler.set_sep(arg.sep_start, arg.sep_end)
    call_extractor = CALL_EXTRACTOR_MAP[arg.format_type]
    
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
            res = [call for call in call_extractor.extract(response)]
            if res and all([check_format(ans) for ans in res]):
                break
            retry_num -= 1
            if arg.temperature <= 0:
                break
            
        output_file.write(json.dumps({"query": query, "response": res, "answers": instruction["answers"]}, ensure_ascii=False) + "\n")
        output_file.flush()
        
    output_file.close()
    if isinstance(handler, HFCausalLMHandler):
        print(f"Average tokens: {handler.total_tokens / handler.inference_count}")
        print(f"Average input tokens: {handler.input_tokens / handler.inference_count}")


if __name__ == '__main__':
    # test_retriever_accuracy()
    main()
    
    