import json
import random
from typing import List, Dict, Tuple, Optional, Union, Callable, Generator, Iterable
from rouge_score import rouge_scorer
import multiprocessing as mp
from functools import partial
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
import os
from abc import ABC, abstractmethod

from utils.extract import extract_and_parse_jsons
       
class GenerateResponse(ABC):
    @abstractmethod
    def __call__(self, prefix:str, queries: List[str], **kwargs)->List[Dict[str, str]]:
        pass
    

from openai import OpenAI

class OpenAiGenerateResponse(GenerateResponse):
    client: OpenAI
    model: str
    system_prompt: str
    
    def __init__(self, client: OpenAI, model: str, system_prompt: str):
        super().__init__()
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        
    def __call__(self, prefix:str, queries: List[str], **kwargs)->List[Dict[str, str]]:
        responses = []
        for query in queries:
            prompt = f"{prefix} {query}"
            completion = self.client.chat.completions.create(
                model = self.model,
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                # **kwargs
            )
            resp = {'text': completion.choices[0].message.content, 'finish_reason': completion.choices[0].finish_reason}
            responses.append(resp)
        
        return responses
            
    

class HuggingfaceGenerateResponse(GenerateResponse):
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
    model: AutoModelForCausalLM
    
    
    def __init__(self, tokenizer: PreTrainedTokenizer, model: AutoModelForCausalLM, system_prompt: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.system_prompt = system_prompt
    
    def __call__(self, prefix:str, queries: List[str], **kwargs):
        sentences = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prefix + q}
                ], 
                tokenize=False,
                add_generation_prompt=True,
            ) for q in queries
        ]
        inp = self.tokenizer(sentences, padding=True, return_tensors="pt").to(self.model.device)
        import torch
        with torch.no_grad():
            out = self.model.generate(**inp, **kwargs)
        r = self.tokenizer.batch_decode(out)
        
        res = [None] * len(sentences)
        for i in range(len(sentences)):
            # print(f"input: {sentences[i]}\n")
            # print(f"output: {r[i]}\n\n\n")
            resp = {'text': r[i][len(sentences[i]):], 'finish_reason': 'length'}
            if out[i][-1] == self.tokenizer.eos_token_id or out[i][-1] == self.tokenizer.pad_token_id:
                resp['finish_reason'] = 'stop'
            res[i] = resp
        
        return res
    

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
        

from string import Template

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class Sampler(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def sample(self)->dict:
        raise NotImplementedError
    
class RandomListSampler(Sampler):
    def __init__(self, data: List[Dict[str, str]],
                 num_samples_per_query: int = 1):
        self.data = data
        self.num_samples_per_query = num_samples_per_query
    
    @abstractmethod
    def format(self, samples: List[Dict[str, str]])->Dict[str, str]:
        raise NotImplementedError
    
    def sample(self)->dict:
        sample_num = min(len(self.data), self.num_samples_per_query)
        samples = random.sample(self.data, sample_num)
        return self.format(samples)
    
    def add_data(self, data: List[Dict[str, str]]):
        self.data.extend(data)
        
    def renew_data(self, data: List[Dict[str, str]]):
        self.data = data
        
class JsonlSampler(Sampler):
    def __init__(self, file: str, num_samples_per_query: int = 1):
        self.file = file
        self.num_samples_per_query = num_samples_per_query
        self.f = open(file, 'r')
        self.eof = False
        
    @abstractmethod
    def format(self, samples: List[Dict[str, str]])->Dict[str, str]:
        raise NotImplementedError
    
    def sample(self)->dict:
        if self.eof:
            return None
        
        # read num_samples_per_query lines
        # if reach the end of file, close the file and return None
        samples = []
        for _ in range(self.num_samples_per_query):
            line = self.f.readline()
            if not line:
                self.f.close()
                self.eof = True
                break
            else:
                samples.append(json.loads(line))

        if not samples:
            return None
        return self.format(samples)
    
    
class DataFilter(ABC):
    def __init__(self, fail_callback: Callable[[Dict[str, str]], None] = None):
        self.fail_callback = fail_callback
    
    def preprocess(self, data: Iterable[Dict[str, str]])->Iterable[Dict[str, str]]:
        # you can create new data from the original data here.
        # not just filter the data.
        return data
    
    @abstractmethod
    def validate(self, data: Dict[str, str])->bool:
        # filter the data generated by preprocess.
        raise NotImplementedError
    
    def filter(self, data: Iterable[Dict[str, str]])->Iterable[Dict[str, str]]:
        for d in self.preprocess(data):
            if self.validate(d):
                yield d
            else:
                if self.fail_callback:
                    self.fail_callback(d)
                    
class CombinedFilter(DataFilter):
    def __init__(self, filters: List[DataFilter], fail_callback: Callable[[Dict[str, str]], None] = None):
        super().__init__(fail_callback)
        self.filters = filters
        
    @abstractmethod
    def combine(self, origin_data: Iterable[Dict[str, str]], 
                filterd_data: Iterable[Dict[str, str]])->Iterable[Dict[str, str]]:
        for d1, d2 in zip(origin_data, filterd_data):
            yield d1.update(d2)
    
    def filter(self, data: Iterable[Dict[str, str]]) -> Iterable[Dict[str, str]]:
        origin = data
        for f in self.filters:
            data = f.filter(data)
        return self.combine(origin, data)
    
class JsonExtractor(DataFilter):
    def preprocess(self, data: Iterable[Dict[str, str]]) -> Iterable[Dict[str, str]]:
        for d in data:
            # if d["finish_reason"] == "stop":
            for item in extract_and_parse_jsons(d["text"]):
                yield item
            else:
                if self.fail_callback:
                    self.fail_callback(d)
    
    def validate(self, data: Dict[str, str]) -> bool:
        return True
    
                    
import logging

class SimilarityFilter(DataFilter):
    def __init__(self, similarity_record: SimilarityRecord, key = "query", bound: float = 0.7,
                 fail_callback: Callable[[Dict[str, str]], None] = None):
        super().__init__(fail_callback)
        self.similarity_record = similarity_record
        self.key = key
        self.bound = bound
        
    def validate(self, data: Dict[str, str]) -> bool:
        most_similar, score = self.similarity_record.update(data[self.key], self.bound)
        if score <= self.bound:
            return True
        else:
            logging.warning(f"{data[self.key]} is too similar to {most_similar}, score: {score}")
            return False
        
    def change_record(self, similarity_record: SimilarityRecord, key: str=None, bound: float=None):
        self.similarity_record = similarity_record
        if key:
            self.key = key
        if bound:
            self.bound = bound
        

from tqdm import tqdm

class LLMDataCollector:
    def __init__(self, prompt: Template,
                 sampler: Sampler,
                 data_filters: List[DataFilter],
                 generate_response: GenerateResponse,
                 num_queries: int = 1,
                 verbose: bool = False):
        self.prompt = prompt
        self.sampler = sampler
        self.data_filters = data_filters
        self.generate_response = generate_response
        self.num_queries = num_queries
        self.verbose = verbose
    
    def add_filter(self, data_filter: DataFilter):
        self.data_filters.append(data_filter)
        
    def switch(self, prompt: Template=None, sampler: Sampler=None, data_filters: List[DataFilter]=None,
               generate_response: GenerateResponse=None):
        if prompt:
            self.prompt = prompt
        if sampler:
            self.sampler = sampler
        if data_filters:
            self.data_filters = data_filters
        if generate_response:
            self.generate_response = generate_response
            
        
    def collect(self, num_data: int, desc: str = "collecting data", num_generated: int = 0,
                once: bool = False, retry_num: int = 2, lower_num: int = 5)->Iterable[Dict[str, str]]:
        process_bar = tqdm(total=num_data, desc=desc)
        process_bar.update(num_generated)
        retry = retry_num
        while num_generated < num_data:
            samples = [self.sampler.sample() for _ in range(self.num_queries)]
            samples = [sample for sample in samples if sample is not None]
            if not samples:
                break
            prompts = [self.prompt.substitute(sample) for sample in samples]
            if self.verbose:
                for prompt in prompts:
                    logging.info(f"{Colors.OKBLUE}prompt: {prompt}{Colors.ENDC}\n\n\n")
            
            responses = self.generate_response('', prompts)
            
            if self.verbose:
                for prompt, response in zip(prompts, responses):
                    logging.info(f"\033[32m prompt: {prompt} finish_reason: {response['finish_reason']}\033[0m\n\033[31mresponse: {response['text']}\033[0m\n\n")
            
            for filter_idx, filter in enumerate(self.data_filters):
                # if self.verbose:
                    # print(f"started filter: {filter_idx}: {filter}")
                responses = filter.filter(responses)
                # if self.verbose:
                    # print(f"end filter: {filter_idx}")
            
            num_filtered = 0
            for response in responses:
                if self.verbose:
                    logging.info(f"\033[34mresponse: {response}\033[0m]")
                yield response
                num_filtered += 1
                num_generated += 1
                process_bar.update(1)
            
            if num_filtered < lower_num:
                retry -= 1
                if retry <= 0:
                    break
            else:
                retry = retry_num
                
            if once:
                break
            

if __name__ == '__main__':
    pass
    
