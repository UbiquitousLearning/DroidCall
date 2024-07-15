import re
import argparse
import json
import random
from typing import List, Dict, Tuple, Optional, Union, Callable
from rouge_score import rouge_scorer
import multiprocessing as mp
from functools import partial
from transformers import PreTrainedTokenizer, AutoTokenizer, LlamaForCausalLM
import torch
import os

# 将以@@@@分割的input和output分割开
# 返回一个个字典，包含input和output(generator)
def parse_input(text: str, finish_reason: str = 'stop'):
    raw_instructions = re.split('@@@@', text)
    
    for idx, raw_instruction in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is not complete, so we skip it
        if idx == len(raw_instructions) - 1 and finish_reason == 'length':
            continue
        
        pattern = r'(\d+\.)?(input|output|Input|Output):'
        parts = re.split(pattern, raw_instruction)
        if len(parts) != 7:
            continue
        
        yield {'input': parts[3].strip(), 'output': parts[6].strip()}
                
def encode_prompt(prompt_instructions: List[Dict[str, str]]):
    """Encode multiple prompt instructions into a single string."""
    with open("./prompt.txt", "r") as f:
        prompt = f.read()
        
    with open("./customizedGPT.txt", "r") as f:
        customizedGPT = f.read()
        prompt = prompt.format(slot=customizedGPT)

    for idx, task_dict in enumerate(prompt_instructions):
        input, output = task_dict["input"], task_dict["output"]
        prompt += f"{idx + 1}.input: {input}\n"
        prompt += f"{idx + 1}.output: {output}\n"
        prompt += "@@@@\n"
    return prompt

def generate_prompts_(tasks: List[Dict[str, str]], num_prompts: int, num_tasks: int):
    for _ in range(num_prompts):
        sample_tasks = random.sample(tasks, num_tasks)
        yield encode_prompt(sample_tasks)

def generate_prompts(file: str, num_prompts: int, num_tasks: int):
    tasks = []
    with open(file, 'r') as f:
        for line in f:
            j = json.loads(line)
            tasks.append(j)
    
    yield from generate_prompts_(tasks, num_prompts, num_tasks)
        
class GenerateResponse:
    """
    a callable class that can generate response from a prefix and a list of queries.
    tokenizer: a PreTrainedTokenizer that can tokenize the sentence to help calculate similarity.
    model: a LlamaForCausalLM that can generate response.
    usage:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = LlamaForCausalLM.from_pretrained(model_path)
        generate_response = GenerateResponse(tokenizer, model)
        prefix = "帮我翻译一下如下句子："
        queries = ["I like to eat apples.", "I do not like to eat apples."]
        r = generate_response(prefix, queries)
        
    return: List[Dict[str, str]]
        e.g. [{'text': '我喜欢吃苹果。', 'finish_reason': 'stop'}, {'text': '我不喜欢吃苹果。', 'finish_reason': 'stop'}]
        finish_reason: 'stop' means the model stops generating response.
                          'length' means the model reach the max_new_tokens or max_length.
    """
    tokenizer: PreTrainedTokenizer
    model: LlamaForCausalLM
    
    PROMPT_TEMPLATE = (
        "[INST] <<SYS>>\n"
        "{system_prompt}\n"
        "<</SYS>>\n\n"
        "{instruction} [/INST]"
    )
    
    SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, model: LlamaForCausalLM):
        self.tokenizer = tokenizer
        self.model = model
    
    def __call__(self, prefix:str, queries: List[str], **kwargs):
        sentences = [
            self.PROMPT_TEMPLATE.format(
                system_prompt=self.SYSTEM_PROMPT,
                instruction=prefix + q
            ) for q in queries
        ]
        inp = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inp, **kwargs)
        r = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        
        res = [None] * len(sentences)
        for i in range(len(sentences)):
            resp = {'text': r[i][len(sentences[i]):], 'finish_reason': 'length'}
            if out[i][-1] == self.tokenizer.eos_token_id or out[i][-1] == self.tokenizer.pad_token_id:
                resp['finish_reason'] = 'stop'
            res[i] = resp
        
        return res
    

from abc import ABC, abstractmethod

class Tokenizer(ABC):
    """
    a tokenizer interface that can tokenize and detokenize the sentence.
    
    tokenize(sentence: str)-> List[str]: tokenize the sentence to a list of tokens.
    detokenize(tokens: List[str])-> str: detokenize the tokens to a sentence.
    """
    @abstractmethod
    def tokenize(self, sentence: str)-> List[str]:
        pass

    @abstractmethod
    def detokenize(self, tokens: List[str])-> str:
        pass
    
    
class HuggingFaceTokenizer(Tokenizer):
    """
    a tokenizer that can tokenize and detokenize the sentence using HuggingFace Tokenizer.
    tokenizer: a PreTrainedTokenizer that can tokenize the sentence.
    """
    tokenizer: PreTrainedTokenizer
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer
        
    def tokenize(self, sentence: str)-> List[str]:
        return self.tokenizer.tokenize(sentence)
    
    def detokenize(self, tokens: List[str])-> str:
        return self.tokenizer.convert_tokens_to_string(tokens)
    
        
class SimilarityRecord:
    """
    a record to record the sentences that have been added, and filter out the similar sentences.
    tokenizer: a Tokenizer that can tokenize the sentence to help calculate similarity.
    num_processes: number of processes to calculate similarity.
    
    updata(sentence: str, bound: float = 0.7)-> (str, float): check if the sentence is similar to the sentences in the record.
        if its similarity is larger than bound, return the most similar sentence and its similarity but not add the sentence to the record.
        else add the sentence to the record and return the most similar sentence and its similarity.
    """
    tokenizer: Tokenizer
    num_processes: int
    sentences: List[List[str]] # List of tokenized sentences
    
    def __init__(self, tokenizer: Tokenizer, num_processes: int=mp.cpu_count()):
        self.tokenizer = tokenizer
        self.num_processes = num_processes
        self.sentences = []
        
    @staticmethod
    def _score(sentence: List[str], other_sentence: List[str])-> tuple[List[str], float]:
        scores = rouge_scorer._score_lcs(sentence, other_sentence)
        return other_sentence, scores.fmeasure
        
    def update(self, sentence: str, bound: float = 0.7)-> tuple[str, float]:
        sentence = self.tokenizer.tokenize(sentence)

        if len(self.sentences) == 0:
            self.sentences.append(sentence)
            return ''.join(sentence), 0.0

        with mp.Pool(self.num_processes) as pool:
            scores = pool.map(partial(self._score, sentence), self.sentences)
        
        most_similar, score = max(scores, key=lambda x: x[1])
        
        if score <= bound:
            self.sentences.append(sentence)
        
        return self.tokenizer.detokenize(most_similar), score
    
    def add(self, sentence: str):
        sentence = self.tokenizer.tokenize(sentence)
        self.sentences.append(sentence)

        
def extract_input_output(arg):
    """
    python utils.py -f extract_input_output\
        --input input_file\
        --output output_file\
        --similarity_bound 0.7\
        --model_path hfl/chinese-alpaca-2-7b
    where input_file contains user input and output pairs, separated by @@@@.
    output_file contains a json object per line, with keys "input" and "output".
    this script will extract the input and output from input_file and write to output_file (duplicated input will be filtered).
    """
    try:
        with open(arg.input, 'r') as f:
            text = f.read()
    except:
        print("Error reading input file")
        
    huggingface_tokenizer = AutoTokenizer.from_pretrained(arg.model_path)
    tokenizer = HuggingFaceTokenizer(huggingface_tokenizer)
    print('tokenizer loaded')
    r = SimilarityRecord(tokenizer)
    try:
        with open(arg.output, 'r') as f:
            for line in f:
                j = json.loads(line)
                r.add(j['input'])
    except:
        print(f"{arg.output} not exist, create new file")
    
    with open(arg.output, 'a') as f:
        for instruction in parse_input(text):
            most_similar, score = r.update(instruction['input'], arg.similarity_bound)
            if score > arg.similarity_bound:
                print(f'input: {instruction["input"]} is too similar to {most_similar}, score: {score}')
                continue
            
            f.write(json.dumps(instruction, ensure_ascii=False) + '\n')
                
        
def gen_prompts(arg):
    """
    python utils.py -f gen_prompts\
        --input input_file\
        --num_prompts 10 \
        --num_tasks 3
    where input_file contains json object per line, with keys "input" and "output".
    num_prompts is the number of prompts to generate.
    num_tasks is the number of tasks used in prompt.
    """
    
    for prompt in generate_prompts(arg.input, arg.num_prompts, arg.num_tasks):
        print(prompt)
        print("=======================================")
    

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    parser = argparse.ArgumentParser(description='args for utils.py')
    parser.add_argument('--input', type=str, help='input file', default='tasks.txt')
    parser.add_argument('--output', type=str, help='output file', default='seeds.jsonl')
    parser.add_argument('-f', type=str, help='specify the function to run', default='extract_input_output')
    parser.add_argument('--num_tasks', type=int, help='number of tasks used in prompt', default=3)
    parser.add_argument('--num_prompts', type=int, help='number of prompts to generate', default=1)
    parser.add_argument('--similarity_bound', type=float, help='similarity bound to filter prompts', default=0.7)
    parser.add_argument('--model_path', type=str, help='tokenizer and model path', default='hfl/chinese-alpaca-2-7b')

    arg = parser.parse_args()

    globals()[arg.f](arg)
    
