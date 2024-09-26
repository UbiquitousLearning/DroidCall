from utils.retriever import ChromaDBRetriever
from utils.planner import Planner
from utils.executor import Executor
from utils import HuggingfaceGenerateResponse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen2VLForConditionalGeneration


path = "checkpoint/Qwen2-1.5B-instruct-mixed"

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", trust_remote_code=True)
    
    llm = HuggingfaceGenerateResponse(tokenizer, model, "you are a helpful assistant")
    retriever = ChromaDBRetriever("./chromaDB")
    executor = Executor("http://10.129.7.240:8080", verbose=True)
    planner = Planner(llm, executor, retriever, 4)
    
    while True:
        query = input("query>>>")
        if query == "exit":
            break
        res = planner.plan_and_execute(query)
        print(f"result: {res}")
