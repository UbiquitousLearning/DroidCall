from string import Template
import json
from utils import SimilarityRecord, OpenAiGenerateResponse, HuggingFaceTokenizer
from transformers import AutoTokenizer
import random
from openai import OpenAI
import os
from tqdm import tqdm
from typing import List, Dict, Iterable
import argparse
from utils.extract import extract_and_parse_jsons
from utils.prompt import SEED_GENERATION_PROMPT, DATA_GENERATION_PROMPT

import logging

logging.basicConfig(level=logging.INFO)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


INIT_PROMPT = Template(SEED_GENERATION_PROMPT)

GEN_PROMPT = Template(DATA_GENERATION_PROMPT)


def format_example(example):
    tool = example["tools"][0]
    resp = {
        "query": example["query"],
        "answers": [
            {"id": id, **ans}
            for id, ans in enumerate(example["answers"])
        ] 
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
argparser.add_argument("--sample_file", type=str, default="data/function_call/processed_xlam.jsonl")
argparser.add_argument("--api_file", type=str, default="data/api.jsonl")
argparser.add_argument("--tokenizer_path", type=str, default="path/to/tokenizer")
argparser.add_argument("--output", type=str, default="data/instructions.jsonl")
argparser.add_argument("--num_generate", type=int, default=300)
argparser.add_argument("--similarity_threshold", type=float, default=0.75)
argparser.add_argument("--sample_num", type=int, default=8)
argparser.add_argument("--model_class", type=str, default="gpt", choices=["gpt", "deepseek"])
argparser.add_argument("--model_name", type=str, default="gpt-4-turbo")
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
    with open(args.sample_file, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            if example["tools_num"] == 1 and example["answers_num"] == 1: # simple call
                all_examples.append(example)
        
    path = args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer = HuggingFaceTokenizer(tokenizer)
    
    func2instructions = {}
    if os.path.exists(args.output):
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

    with open(args.api_file) as f:
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
                d["tools"] = [tool]
                for id in range(len(d["answers"])):
                    d["answers"][id]["id"] = id
                data.append(d)
                output_file.write(json.dumps(d, ensure_ascii=False)+"\n")
                output_file.flush()
        
        class QuerySampler(RandomListSampler):
            def format(self, samples: List[Dict[str, str]])->Dict[str, str]:
                samples = [
                    {k: v for k, v in sample.items() if k not in["tools"]}
                    for sample in samples
                ]
                for i in range(len(samples)):
                    for id in range(len(samples[i]["answers"])):
                        samples[i]["answers"][id]["id"] = id
                
                examples_text = "\n".join([json.dumps(sample, indent=2, ensure_ascii=False) for sample in samples])
                return {"examples": examples_text, "tool": tool_text}
            
        collector.switch(GEN_PROMPT, QuerySampler(data, SAMPLE_NUM))
        for d in collector.collect(NUM_GENERATE, "gen collection", len(data)):
            d["tools"] = [tool]
            for id in range(len(d["answers"])):
                d["answers"][id]["id"] = id
            data.append(d)
            output_file.write(json.dumps(d, ensure_ascii=False)+"\n")
            output_file.flush()
        
        records = SimilarityRecord(tokenizer)
        similarity_filter.change_record(records)
    
