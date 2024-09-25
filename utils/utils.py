import re
import argparse
import json
import random
from typing import List, Dict, Tuple, Optional, Union, Callable, Generator, Iterable
from rouge_score import rouge_scorer
import multiprocessing as mp
from functools import partial
from transformers import PreTrainedTokenizer, AutoTokenizer, AutoModelForCausalLM
import os
from abc import ABC, abstractmethod


# a text parser that can parse the text to specific object by a specific rule.
class TextParser(ABC):
    @abstractmethod
    def parse(self, text: str)-> List[Dict[str, str]]:
        pass
    
    def __call__(self, text: str)-> Generator[Dict[str, str], None, None]:
        return self.parse(text)
    

class InputOutputParser(TextParser):
    def _parse_item(self, text: str)-> Dict[str, str]:
        pattern = r'(\d+\.)?(input|output|Input|Output):'
        parts = re.split(pattern, text.strip())
        if len(parts) != 7:
            return {}
        return {'input': parts[3].strip(), 'output': parts[6].strip()}
    
    def parse(self, text: str)-> Generator[Dict[str, str], None, None]:
        raw_instructions = re.split('@@@@', text)
        raw_instructions = raw_instructions[1:-1] # remove the first and last since GPT often generates output that doesn't match the pattern at the beginning and end.
        
        for _, raw_instruction in enumerate(raw_instructions):
            d = self._parse_item(raw_instruction)
            
            if not d:
                continue
            
            yield d
        

class JsonParser(TextParser):
    def parse(self, text: str)-> Generator[Dict[str, str], None, None]:
        # 更新后的正则表达式模式，用于匹配JSON对象或数组
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}|\[(?:[^\[\]]|(?:\[[^\[\]]*\]))*\]'
        
        # 使用finditer来查找所有匹配的JSON字符串
        matches = re.finditer(json_pattern, text)
        
        for match in matches:
            json_string = match.group()
            try:
                # 解析JSON字符串为Python对象
                python_obj = json.loads(json_string)
                if isinstance(python_obj, list):
                    for item in python_obj:
                        yield item
                else:
                    yield python_obj
            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}")
                continue
            except Exception as e:
                print(f"An error occurred: {e}")
                continue


DEFAULT_PARSER = InputOutputParser()

# 将以@@@@分割的input和output分割开
# 返回一个个字典，包含input和output(generator)
def parse_input(text: str, parser: TextParser = DEFAULT_PARSER)->Generator[Dict[str, str], None, None]:
    yield from parser.parse(text)
                
                
class TaskFormatter(ABC):  
    @abstractmethod
    def format(self, task: Dict[str, str])-> str:
        pass
    
    def __call__(self, task: Dict[str, str])-> str:
        return self.format(task)

class InputOutputFormatter(TaskFormatter):
    def __init__(self):
        super().__init__()
        self.num = 0
    
    def format(self, task: Dict[str, str])-> str:
        self.num += 1
        return f"{self.num}.input: {task['input']}\n{self.num}.output: {task['output']}\n"
    

class JsonFormatter(TaskFormatter):
    def format(self, task: Dict[str, str])-> str:
        return json.dumps(task, ensure_ascii=False, indent=2)


DEFAULT_FORMATTER = InputOutputFormatter()


def encode_prompt(prompt: str, slot: str, prompt_instructions: List[Dict[str, str]], formatter: TaskFormatter)->str:
    """Encode multiple prompt instructions into a single string."""
    prompt = prompt.format(slot=slot)

    for idx, task_dict in enumerate(prompt_instructions):
        task_text = formatter(task_dict)
        prompt += task_text
        prompt += "\n@@@@\n"
    return prompt


def generate_prompts_(prompt: str, slot: str, tasks: List[Dict[str, str]], num_prompts: int, num_tasks: int,formatter: TaskFormatter = DEFAULT_FORMATTER):
    for _ in range(num_prompts):
        sample_tasks = random.sample(tasks, num_tasks)
        yield encode_prompt(prompt, slot, sample_tasks, formatter=formatter)

def generate_prompts(file: str, num_prompts: int, num_tasks: int):
    tasks = []
    with open(file, 'r') as f:
        for line in f:
            j = json.loads(line)
            tasks.append(j)
    
    yield from generate_prompts_(tasks, num_prompts, num_tasks)
     
     
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
                **kwargs
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
        r = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        
        res = [None] * len(sentences)
        for i in range(len(sentences)):
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
        

import pyparsing as pp
from pyparsing import pyparsing_common as ppc

def get_json_obj(text: str):
    def make_keyword(kwd_str, kwd_value):
        return pp.Keyword(kwd_str).setParseAction(pp.replaceWith(kwd_value))

    if not hasattr(get_json_obj, "jsonDoc"):
        # set to False to return ParseResults
        RETURN_PYTHON_COLLECTIONS = True

        TRUE = make_keyword("true", True)
        FALSE = make_keyword("false", False)
        NULL = make_keyword("null", None)

        LBRACK, RBRACK, LBRACE, RBRACE, COLON = map(pp.Suppress, "[]{}:")

        jsonString = pp.dblQuotedString().setParseAction(pp.removeQuotes)
        jsonNumber = ppc.number().setName("jsonNumber")

        jsonObject = pp.Forward().setName("jsonObject")
        jsonValue = pp.Forward().setName("jsonValue")

        jsonElements = pp.delimitedList(jsonValue).setName(None)

        jsonArray = pp.Group(
            LBRACK + pp.Optional(jsonElements) + RBRACK, aslist=RETURN_PYTHON_COLLECTIONS
        ).setName("jsonArray")

        jsonValue << (jsonString | jsonNumber | jsonObject | jsonArray | TRUE | FALSE | NULL)

        memberDef = pp.Group(
            jsonString + COLON + jsonValue, aslist=RETURN_PYTHON_COLLECTIONS
        ).setName("jsonMember")

        jsonMembers = pp.delimitedList(memberDef).setName(None)
        jsonObject << pp.Dict(
            LBRACE + pp.Optional(jsonMembers) + RBRACE, asdict=RETURN_PYTHON_COLLECTIONS
        )

        jsonComment = pp.cppStyleComment
        jsonObject.ignore(jsonComment)
        jsonDoc = jsonObject | jsonArray
        get_json_obj.jsonDoc = jsonDoc
    for _, l, r in get_json_obj.jsonDoc.scanString(text):
        json_string = text[l:r]
        try:
            # 尝试解析找到的JSON字符串
            parsed_data = json.loads(json_string)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        
def extract_and_parse_jsons(text):
    def make_keyword(kwd_str, kwd_value):
        return pp.Keyword(kwd_str).setParseAction(pp.replaceWith(kwd_value))

    if not hasattr(extract_and_parse_jsons, "jsonDoc"):
        # set to False to return ParseResults
        RETURN_PYTHON_COLLECTIONS = True

        TRUE = make_keyword("true", True)
        FALSE = make_keyword("false", False)
        NULL = make_keyword("null", None)

        LBRACK, RBRACK, LBRACE, RBRACE, COLON = map(pp.Suppress, "[]{}:")

        jsonString = pp.dblQuotedString().setParseAction(pp.removeQuotes)
        jsonNumber = ppc.number().setName("jsonNumber")

        jsonObject = pp.Forward().setName("jsonObject")
        jsonValue = pp.Forward().setName("jsonValue")

        jsonElements = pp.delimitedList(jsonValue).setName(None)

        jsonArray = pp.Group(
            LBRACK + pp.Optional(jsonElements) + RBRACK, aslist=RETURN_PYTHON_COLLECTIONS
        ).setName("jsonArray")

        jsonValue << (jsonString | jsonNumber | jsonObject | jsonArray | TRUE | FALSE | NULL)

        memberDef = pp.Group(
            jsonString + COLON + jsonValue, aslist=RETURN_PYTHON_COLLECTIONS
        ).setName("jsonMember")

        jsonMembers = pp.delimitedList(memberDef).setName(None)
        jsonObject << pp.Dict(
            LBRACE + pp.Optional(jsonMembers) + RBRACE, asdict=RETURN_PYTHON_COLLECTIONS
        )

        jsonComment = pp.cppStyleComment
        jsonObject.ignore(jsonComment)
        jsonDoc = jsonObject | jsonArray
        extract_and_parse_jsons.jsonDoc = jsonDoc
    for _, l, r in extract_and_parse_jsons.jsonDoc.scanString(text):
        json_string = text[l:r]
        try:
            # 尝试解析找到的JSON字符串
            parsed_data = json.loads(json_string)
            
            # 如果解析结果是列表，则返回列表中的每个项
            if isinstance(parsed_data, list):
                for item in parsed_data:
                    yield item
            else:
                # 如果不是列表，则直接返回解析结果
                yield parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

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
    def sample()->dict:
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
            print(f"{Colors.WARNING}filter: {self}, started validate{Colors.ENDC}")
            if self.validate(d):
                print(f"{Colors.WARNING}filter: {self}, end validate (pass){Colors.ENDC}")
                yield d
            else:
                print(f"{Colors.WARNING}filter: {self}, end validate (fail){Colors.ENDC}")
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
        print(f"{Colors.BOLD} started to extract jsons{Colors.ENDC}")
        for d in data:
            # if d["finish_reason"] == "stop":
            for item in extract_and_parse_jsons(d["text"]):
                print(f"{Colors.BOLD} extract !!{Colors.ENDC}")
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
    
