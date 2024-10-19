from .extract import extract_and_parse_jsons
from .utils import GenerateResponse
from .executor import Executor, Call, Result
import json
from .prompt import JSON_NESTED_CALLING_PROMT, FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL, JSON_CALL_FORMAT
from string import Template
from .retriever import Retriever
    

class Planner:
    def __init__(self, llm: GenerateResponse, executor: Executor, retriever: Retriever, retriever_num: int = 2,
                 fewshot: bool = False, examples_file: str = None, is_nested: bool = False, verbose: bool = False):
        self.calls = []
        self.llm = llm
        self.executor = executor
        self.retriever = retriever
        self.is_nested = is_nested
        self.retriever_num = retriever_num
        self.fewshot = fewshot
        self.PROMPT = Template(FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL)
        self.verbose = verbose
        
        if fewshot:
            assert examples_file is not None
            with open(examples_file) as f:
                self.examples = [json.loads(line) for line in f]
                
                
    def format_user_message(self, query: str, docs: list[str], is_nested: bool = False):
        nest_prompt = JSON_NESTED_CALLING_PROMT if is_nested else ""
        example_text = ""
        if self.fewshot:
            sampled_examples = []
            for doc in docs:
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
                    
            example_text =  "Here is some examples:\n" + "\n".join(f"query: {example["query"]} \nanwsers: {json.dumps(example["answers"], ensure_ascii=False, indent=2)}" 
                                                                   for example in sampled_examples)

        return self.PROMPT.substitute(user_query=query, functions="\n".join(docs), nest_prompt=nest_prompt, example=example_text, call_format=JSON_CALL_FORMAT)
    
    
    def plan(self, query: str):
        docs = self.retriever.retrieve(query, self.retriever_num)
        if self.verbose:
            from utils import Colors
            for i, doc in enumerate(docs):
                print(f"{Colors.FAIL}doc {i}: {doc}\n{Colors.ENDC}")
        user_message = self.format_user_message(query, docs, self.is_nested)
        response = self.llm("", [user_message], max_new_tokens=200, do_sample=False)[0]["text"]
        # print(f"user: {user_message}")
        print(f"{Colors.OKBLUE}response: {response}\n{Colors.ENDC}")
        res = [call for call in extract_and_parse_jsons(response)]
        
        def filter(item):
            if isinstance(item, dict) and "name" in item:
                if "arguments" not in item or not isinstance(item["arguments"], dict):
                    item["arguments"] = {}
                return True
            return False
        
        if self.verbose:
            for i, call in enumerate(res):
                print(f"{Colors.BOLD}call {i}: {call}\n{Colors.ENDC}")
        self.calls = [Call(name=call["name"], arguments=call["arguments"]) for call in res if filter(call)]
        
    def plan_and_execute(self, query: str)->tuple[bool, str]:
        self.plan(query)
        
        for call in self.calls:
            result = self.executor.execute(call)
            if result.state == "error":
                return False, result.message
            if self.verbose:
                from utils import Colors
                print(f"{Colors.OKGREEN}result: {result}{Colors.ENDC}")
                
        return True, "All calls executed successfully"
