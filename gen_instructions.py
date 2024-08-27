from string import Template
import json
from utils import SimilarityRecord, OpenAiGenerateResponse, HuggingFaceTokenizer, extract_and_parse_jsons
from transformers import AutoTokenizer
import random
from openai import OpenAI
import os
from tqdm import tqdm
from typing import List, Dict, Iterable
import argparse

import logging

logging.basicConfig(level=logging.INFO)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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
9. The parameter values generated with function call generated must be values that can be inferred from the user's query; you cannot fabricate a value out of thin air.

following are some examples:
$examples

Now I will give you a tool, and you help me generate 40 query-answer pairs.
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
9. The parameter values generated with function call generated must be values that can be inferred from the user's query; you cannot fabricate a value out of thin air.

following are tool I provided and some examples of query-answer pairs:
tool: $tool
examples: $examples

Now please help me generate 40 query-answer pairs.
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

argparser = argparse.ArgumentParser()
argparser.add_argument("--output", type=str, default="data/instructions.jsonl")
argparser.add_argument("--num_generate", type=int, default=300)
argparser.add_argument("--similarity_threshold", type=float, default=0.75)
argparser.add_argument("--sample_num", type=int, default=8)
argparser.add_argument("--model_class", type=str, default="gpt", choices=["gpt", "deepseek"])
argparser.add_argument("--model_name", type=str, default="gpt-4o")
args = argparser.parse_args()

MODEL_CLASS_MAP = {
    "gpt": {
        "base_url": None,
        "api_key": None,
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/",
        "api_key": os.environ.get("DEEPSEEK_API_KEY", None),
    }
}

OUTPUT_FILE = args.output
NUM_GENERATE = args.num_generate
SIMILARITY_THRESHOLD = args.similarity_threshold
SAMPLE_NUM = args.sample_num

from utils import RandomListSampler, JsonlSampler, LLMDataCollector, JsonExtractor, SimilarityFilter, DataFilter

class FormatFilter(DataFilter):
    def validate(self, data: Dict[str, str]) -> bool:
        return check_format(data)

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
    
    func2instructions = {}
    with open(args.output) as f:
        for l in f.readlines():
            d = json.loads(l)
            all_examples.append(d)
            func_name = d["answers"][0]["name"]
            if func_name not in func2instructions:
                func2instructions[func_name] = []
            func2instructions[func_name].append(d)

    records = SimilarityRecord(tokenizer)
    client = OpenAI(api_key=MODEL_CLASS_MAP[args.model_class]["api_key"], base_url=MODEL_CLASS_MAP[args.model_class]["base_url"])
    generate_response = OpenAiGenerateResponse(client=client, model=args.model_name, system_prompt="")

    with open("data/api.jsonl") as f:
        all_tools = [json.loads(line) for line in f.readlines()]
    
    output_file = open(OUTPUT_FILE, "a")
    similarity_filter = SimilarityFilter(records, key="query", bound=SIMILARITY_THRESHOLD)
    filters = [JsonExtractor(), FormatFilter(), similarity_filter]
    for tool_idx, tool in enumerate(all_tools):
        data = []
        for example in func2instructions.get(tool["name"], []):
            records.add(example["query"])
            data.append(example)
        
        initial_num = len(data)
        if initial_num >= NUM_GENERATE:
            continue
        
        tool_text = json.dumps(tool, indent=4, ensure_ascii=False)
        print(f"started to process tool {tool_idx}: {tool_text}")
        class ExampleSampler(RandomListSampler):
            def format(self, samples: List[Dict[str, str]])->Dict[str, str]:
                examples_text = "\n".join([format_example(sample) for sample in samples])
                return {"examples": examples_text, "tool": tool_text}
        
        collector = LLMDataCollector(INIT_PROMPT, ExampleSampler(all_examples, 2), filters,
                                     generate_response=generate_response, verbose=True)
        
        # this is initial collection
        while len(data) <= 0:
            for d in collector.collect(NUM_GENERATE, "init collection", num_generated=len(data), once=True):
                data.append(d)
                d["tools"] = [tool]
                output_file.write(json.dumps(d, ensure_ascii=False)+"\n")
                output_file.flush()
        
        class QuerySampler(RandomListSampler):
            def format(self, samples: List[Dict[str, str]])->Dict[str, str]:
                samples = [
                    {k: v for k, v in sample.items() if k not in["tools"]}
                    for sample in samples
                ]
                examples_text = "\n".join([json.dumps(sample, indent=2, ensure_ascii=False) for sample in samples])
                return {"examples": examples_text, "tool": tool_text}
            
        collector.switch(GEN_PROMPT, QuerySampler(data, SAMPLE_NUM))
        for d in collector.collect(NUM_GENERATE, "gen collection", len(data)):
            data.append(d)
            d["tools"] = [tool]
            output_file.write(json.dumps(d, ensure_ascii=False)+"\n")
            output_file.flush()
        
        # for d in data[initial_num:]:
        #     d["tools"] = [tool]
        #     output_file.write(json.dumps(d, ensure_ascii=False)+"\n")
        # output_file.flush()
        
        records = SimilarityRecord(tokenizer)
        similarity_filter.change_record(records)
    
