from utils.retriever import ChromaDBRetriever, LLMRetriever
from utils.planner import Planner
from utils.executor import Executor
from utils import HuggingfaceGenerateResponse, OpenAiGenerateResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.prompt import SYSTEM_PROMPT_FOR_FUNCTION_CALLING, SHORT_SYSTEM_PROMPT_FOR_FUNCTION_CALLING
from openai import OpenAI

# path = "checkpoint/qwen2.5-best"
# path = "../data/xllm-best_ckpt/xllm_DroidCall_code_short_directly"
# path = "../data/xllm-best_ckpt/xllm_DroidCall_code_short"
# path = "../data/xllm-best_ckpt/xllm_DroidCall_code"
# path = "/data/shrelic/data/xllm-best_ckpt/xllm_DroidCall_code_short_directly_2"
path = "/data/share/Qwen2.5-1.5B-Instruct-DroidCall_code_short"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True)
    llm = HuggingfaceGenerateResponse(tokenizer, model, SHORT_SYSTEM_PROMPT_FOR_FUNCTION_CALLING)
    
    # client = OpenAI()
    # llm = OpenAiGenerateResponse(client, "gpt-4o-mini", SYSTEM_PROMPT_FOR_FUNCTION_CALLING)
    
    retriever = ChromaDBRetriever("./chromaDB")
    # retriever = LLMRetriever("data/api.jsonl", llm)
    
    executor = Executor("http://10.129.7.240:8080", verbose=True)
    planner = Planner(llm, executor, retriever, 4, verbose=True, format_type="code_short")
    planner.set_sep("$", "$")
    
    while True:
        query = input("query>>>")
        if query == "exit":
            break
        res = planner.plan_and_execute(query)
        print(f"result: {res}")
