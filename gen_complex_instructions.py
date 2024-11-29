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
import copy
from utils.extract import extract_and_parse_jsons
from utils.prompt import COMPLEX_INSTRUCTION_SEED_PROMPT, COMPLEX_INSTRUCTION_GEN_PROMPT

import logging

logging.basicConfig(level=logging.INFO)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


INIT_PROMPT = Template(COMPLEX_INSTRUCTION_SEED_PROMPT)

GEN_PROMPT = Template(COMPLEX_INSTRUCTION_GEN_PROMPT)


def format_example(example):
    tool = example["tools"]
    resp = {
        "query": example["query"],
        "answers": [
            {"id": id, **ans}
            for id, ans in enumerate(example["answers"])
        ] 
    }
    return f'tools:\n {json.dumps(tool, indent=2, ensure_ascii=False)}\nresponse: \n{json.dumps(resp, indent=2)}'


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
argparser.add_argument("--output", type=str, default="data/instructions_complex.jsonl")
argparser.add_argument("--num_generate", type=int, default=80)
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

from utils import Sampler, LLMDataCollector, JsonExtractor, SimilarityFilter, DataFilter

class FormatFilter(DataFilter):
    def validate(self, data: Dict[str, str]) -> bool:
        return check_format(data)
    
class ApiSampler(Sampler):
    combinations = [
        ["ACTION_SET_ALARM", "ACTION_GET_RINGTONE"],
        ["ACTION_GET_CONTENT", "send_email"],
        ["ACTION_GET_CONTENT", "send_message"],
        ["ACTION_OPEN_DOCUMENT", "send_email"],
        ["ACTION_OPEN_DOCUMENT", "send_message"],
        ["get_contact_info", "send_email"],
        ["get_contact_info", "send_message"],
        ["get_contact_info", "dial"],
        ["ACTION_IMAGE_CAPTURE", "ACTION_GET_CONTENT", "send_email"],
        ["ACTION_PICK", "get_contact_from_uri", "send_email"],
        ["get_contact_info", "send_message", "dial"],
        ["get_contact_info", "search_location"],
        ["ACTION_PICT", "ACTION_VIEW_CONTACT"],
        ["ACTION_CREATE_DOCUMENT", "send_email"],
        ["ACTION_CREATE_DOCUMENT", "send_message"],
        ["ACTION_VIDEO_CAPTURE", "send_email"],
        ["get_contact_info", "web_search"],
        ["get_contact_info", "ACTION_INSERT_EVENT"],
        ["ACTION_PICK", "ACTION_EDIT_CONTACT"],
        ["get_contact_info", "ACTION_EDIT_CONTACT"],
    ]
    
    def __init__(self, api_file: str, data:list, seed_samples_num: int=5, max_api_num: int=4):
        with open(api_file) as f:
            self.apis = [json.loads(line) for line in f.readlines()]
        self.max_api_num = max_api_num
        self.seed_samples_num = seed_samples_num
        self.apis2data = {}
        self.last_apis = []
        self.in_seed_generation_state = True
        self.data = data
            
    def add_samples(self, samples_data: list[dict]):
        key = self.get_keys(self.last_apis)
        if key not in self.apis2data:
            self.apis2data[key] = []
        self.apis2data[key].extend(samples_data)
        
    def add_sample(self, sample_data: dict):
        key = self.get_keys(self.last_apis)
        if key not in self.apis2data:
            self.apis2data[key] = []
        self.apis2data[key].append(sample_data)
    
    def random_sample_apis(self)->list:
        # first chose how much apis to sample
        # then sample from apis
        num_apis = random.randint(2, self.max_api_num)
        return random.sample(self.apis, num_apis)
    
    def sample_preset_apis(self)->list:
        names = random.choice(self.combinations)
        return [api for api in self.apis if api["name"] in names]
    
    def sample_apis(self)->list:
        # 30% to sample_preset_apis
        # 70% to random_sample_apis
        if random.random() < 0.30:
            self.last_apis = self.sample_preset_apis()
        else:
            self.last_apis = self.random_sample_apis()
    
    @staticmethod
    def get_keys(apis: list)->str:
        sorted_api_names = sorted([api["name"] for api in apis])
        return '#'.join(sorted_api_names)
    
    def seed_generation(self):
        self.in_seed_generation_state = True
    
    def sample(self)->dict:
        apis = self.last_apis
        
        tools_text = json.dumps(apis, indent=2, ensure_ascii=False)
        if self.in_seed_generation_state:
            examples = random.sample(self.data, 2)
            examples_data_text = "\n".join([format_example(example) for example in examples])
            self.in_seed_generation_state = False
        else:
            key = self.get_keys(apis)
            data = self.apis2data.get(key, [])
            examples_data = random.sample(data, min(len(data), self.seed_samples_num))
            examples_data_text = "\n".join([json.dumps(d, indent=2, ensure_ascii=False) for d in examples_data])
        return {"tools": tools_text, "examples": examples_data_text}

if __name__ == "__main__":
    all_examples = []
    with open(args.sample_file, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            if example["tools_num"] > 1 and example["tools_num"] <= example["answers_num"]: # complex call
                all_examples.append(example)
        
    path = args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer = HuggingFaceTokenizer(tokenizer)
    

    records = SimilarityRecord(tokenizer)
    client = OpenAI(api_key=MODEL_CLASS_MAP[args.model_class]["api_key"], base_url=MODEL_CLASS_MAP[args.model_class]["base_url"])
    generate_response = OpenAiGenerateResponse(client=client, model=args.model_name, system_prompt="")
    
    output_file = open(OUTPUT_FILE, "a")
    similarity_filter = SimilarityFilter(records, key="query", bound=SIMILARITY_THRESHOLD)
    filters = [JsonExtractor(), FormatFilter(), similarity_filter]
    
    sampler = ApiSampler(args.api_file, all_examples, seed_samples_num=7, max_api_num=3)
    collector = LLMDataCollector(INIT_PROMPT, sampler, filters,
                                     generate_response=generate_response, verbose=True)
    
    for i in range(100):
        # this is initial collection
        num = 0
        sampler.sample_apis()
        collector.switch(INIT_PROMPT)
        while num <= 8:
            sampler.seed_generation()
            for d in collector.collect(NUM_GENERATE, "init collection", num_generated=0, once=True):
                t = copy.deepcopy(d)
                t["tools"] = sampler.last_apis
                output_file.write(json.dumps(t, ensure_ascii=False)+"\n")
                output_file.flush()
                num += 1
                sampler.add_sample(d)
                
        collector.switch(GEN_PROMPT)
        for d in collector.collect(NUM_GENERATE, "gen collection", num_generated=num):
            t = copy.deepcopy(d)
            t["tools"] = sampler.last_apis
            output_file.write(json.dumps(t, ensure_ascii=False)+"\n")
            output_file.flush()
            num += 1
            sampler.add_sample(d)
        
        records = SimilarityRecord(tokenizer)
        similarity_filter.change_record(records)
    

    
    