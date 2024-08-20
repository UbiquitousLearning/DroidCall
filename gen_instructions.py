from string import Template
import json
from utils import SimilarityRecord, OpenAiGenerateResponse, HuggingFaceTokenizer, extract_and_parse_jsons
from transformers import AutoTokenizer
import random
from openai import OpenAI
import os
from tqdm import tqdm


INIT_PROMPT = Template("""
I need your help to generate some function calling datasets. I will provide you with a tool description, and you need to generate queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, you may omit its value.
7. The query-answer pairs should cover as many possible uses of the tool as possible.
8. The generated data must be presented in the format given in my example.

following are some examples:
$examples

Now I will give you a tool, and you help me generate 20 query-answer pairs.
REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE
tool: $tool
""")

GEN_PROMPT = Template("""
I need your help to generate some function calling datasets. I will provide you with a tool description and some example data for you. 
You need to generate queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, it is not necessary to provide its value.
7. The query-answer pairs should cover as many possible uses of the tool as possible.
8. The generated data must be presented in the format given in my example.

following are tool I provided and some examples of query-answer pairs:
tool: $tool
examples: $examples

Now please help me generate 20 query-answer pairs.
REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE
""")


def format_example(example):
    tool = example["tools"][0]
    resp = {
        "query": example["query"],
        "answers": example["answers"]
    }
    return f'tool: {json.dumps(tool, indent=2, ensure_ascii=False)}\nresponse: {json.dumps(resp, indent=2)}'


def check_format(data):
    if "query" not in data or "answers" not in data:
        return False
    if not isinstance(data["query"], str):
        return False
    if not isinstance(data["answers"], list):
        return False
    for ans in data["answers"]:
        if not isinstance(ans, dict):
            return False
        if "name" not in ans or "arguments" not in ans:
            return False
        if not isinstance(ans["arguments"], dict):
            return False
    return True

OUTPUT_FILE = "data/instructions.jsonl"
NUM_GENERATED = 40
SIMILARITY_THRESHOLD = 0.75
SAMPLE_NUM = 4

if __name__ == "__main__":
    all_examples = []
    with open("data/processed_xlam.jsonl", "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            if example["tools_num"] == 1 and example["answers_num"] == 1:
                all_examples.append(example)
        
    path = "../xLLM/tokenizer_qwen2"
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer = HuggingFaceTokenizer(tokenizer)

    records = SimilarityRecord(tokenizer)
    client = OpenAI()
    generate_response = OpenAiGenerateResponse(client=client, model="gpt-4o-mini", system_prompt="")

    with open("data/api.jsonl") as f:
        all_tools = [json.loads(line) for line in f.readlines()]
    
    if os.path.exists(".tmp.json"):
        with open(".tmp.json", "r") as f:
            j = json.load(f)
            processed_num = j["processed_num"]
            all_tools = all_tools[processed_num:]
    
    for tool_idx, tool in enumerate(all_tools):
        process_bar = tqdm(total=NUM_GENERATED, desc=f"processing tool {tool_idx}")
        data = []
        tool_text = json.dumps(tool, indent=4, ensure_ascii=False)
        print(f"started to process tool {tool_idx}: {tool_text}")
        examples = random.sample(all_examples, 2)
        examples_text = "\n".join([format_example(example) for example in examples])
        prompt_text = INIT_PROMPT.substitute(examples=examples_text, tool=tool_text)
        # print(prompt_text)
        # print("\n\n")
        
        output_file = open(OUTPUT_FILE, "a")
        resps = generate_response('', [prompt_text])
        
        for resp in resps:
            if resp["finish_reason"] == "stop":
                # print(f'text: {resp["text"]}\n\n')
                for j in extract_and_parse_jsons(resp["text"]):
                    # print(json.dumps(j, indent=2, ensure_ascii=False))
                    if not check_format(j):
                        continue
                    most_similar, score = records.update(j["query"], SIMILARITY_THRESHOLD)
                    if score > SIMILARITY_THRESHOLD:
                        print(f"most similar: {most_similar}, score: {score}")
                    else:
                        process_bar.update(1)
                        data.append(j)
                    
        
        while len(data) < NUM_GENERATED:
            sampled_query_answer_pairs = random.sample(data, SAMPLE_NUM)
            examples_text = "\n".join([json.dumps(pair, indent=2) for pair in sampled_query_answer_pairs])
            prompt_text = GEN_PROMPT.substitute(examples=examples_text, tool=tool_text)
            # print(f"prompt: {prompt_text}")
            resps = generate_response('', [prompt_text])
            for resp in resps:
                if resp["finish_reason"] == "stop":
                    for j in extract_and_parse_jsons(resp["text"]):
                        if not check_format(j):
                            continue
                        most_similar, score = records.update(j["query"], SIMILARITY_THRESHOLD)
                        if score > SIMILARITY_THRESHOLD:
                            print(f"most similar: {most_similar}, score: {score}")
                        else:
                            process_bar.update(1)
                            data.append(j)
            
            for d in data:
                d["tools"] = [tool]
                output_file.write(json.dumps(d, ensure_ascii=False)+"\n")
            output_file.flush()
            with open(".tmp.json", "w") as f:
                f.write(json.dumps({"processed_num": tool_idx}))
            
        
        
    
