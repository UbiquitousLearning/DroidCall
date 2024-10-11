from utils.retriever import ChromaDBRetriever
from utils.planner import Planner
from utils.executor import Executor
from utils import HuggingfaceGenerateResponse, OpenAiGenerateResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.prompt import SYSTEM_PROMPT_FOR_FUNCTION_CALLING
from openai import OpenAI

path = "checkpoint/Qwen2-1.5B-instruct-mixed"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True)
    
    # llm = HuggingfaceGenerateResponse(tokenizer, model, SYSTEM_PROMPT_FOR_FUNCTION_CALLING)
    
    client = OpenAI()
    llm = OpenAiGenerateResponse(client, "gpt-4o-mini", SYSTEM_PROMPT_FOR_FUNCTION_CALLING)
    retriever = ChromaDBRetriever("./chromaDB")
    executor = Executor("http://10.129.7.240:8080", verbose=True)
    planner = Planner(llm, executor, retriever, 4, verbose=True)
    
    while True:
        query = input("query>>>")
        if query == "exit":
            break
        res = planner.plan_and_execute(query)
        print(f"result: {res}")
