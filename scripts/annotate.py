from string import Template
import json
from openai import OpenAI
import os
os.sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../")
from utils.prompt import ANNOTATION_PROMPT

PROMPT = Template(ANNOTATION_PROMPT)

API_FILE = "data/api.jsonl"
OUTPUT_FILE = "data/annotated_api.jsonl"

from utils import JsonlSampler, JsonExtractor, OpenAiGenerateResponse, LLMDataCollector
from typing import List, Dict

class JsonFormatSampler(JsonlSampler):
    def format(self, samples: List[Dict[str, str]])->Dict[str, str]:
        self.last_sample = samples[0]
        return {"function": json.dumps(samples[0], indent=2, ensure_ascii=False)}

if __name__ == "__main__":
    sampler = JsonFormatSampler(API_FILE)
    extractor = JsonExtractor()
    
    client = OpenAI()
    generate_response = OpenAiGenerateResponse(client=client, model="gpt-4-turbo", system_prompt="")
    
    collector = LLMDataCollector(PROMPT, sampler, [extractor], generate_response)
    
    with open(OUTPUT_FILE, "w") as out_f:
        for annotation in collector.collect(1000, lower_num=1):
            api = sampler.last_sample
            print(f"api: {json.dumps(api, indent=2, ensure_ascii=False)}\n")
            print(f"annotation: {json.dumps(annotation, indent=2, ensure_ascii=False)}")
            for k in api["arguments"]:
                if k in annotation:
                    api["arguments"][k].update(annotation[k])
            out_f.write(json.dumps(api, ensure_ascii=False)+"\n")
    
    
    
    