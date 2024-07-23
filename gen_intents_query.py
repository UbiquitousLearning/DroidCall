import argparse
import random
import json
from tqdm import tqdm 
from utils import *
from transformers import AutoTokenizer
from openai import OpenAI


system_prompt = """
You are an Android development expert, skilled in understanding user needs and translating them into appropriate Android intent usage. Your task is to generate realistic user queries and provide solutions using Android intents.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for gen_intents_query.py')
    parser.add_argument('--intents_file', type=str, help='file that stores the intents info', default='intents.jsonl')
    parser.add_argument('--seed', type=str, help='seed file', default='seeds.jsonl')
    parser.add_argument('--output', type=str, help='output file', default='machine_generated.jsonl')
    parser.add_argument('--model_path', type=str, help='tokenizer and model path', default='./tokenizer_qwen2')
    parser.add_argument('--similarity_bound', type=float, help='similarity bound to filter prompts', default=0.7)
    parser.add_argument('--num_data', type=int, help='number of data to generate', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size', default=1)
    parser.add_argument('--num_tasks', type=int, help='number of tasks used in prompt', default=2)

    arg = parser.parse_args()
    
    print(f'started to load model from {arg.model_path}')
    huggingface_tokenizer = AutoTokenizer.from_pretrained(arg.model_path)
    tokenizer = HuggingFaceTokenizer(huggingface_tokenizer)
    
    intents_info = {}
    with open(arg.intents_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            j = json.loads(line)
            intents_info[j['id']] = j
    
    records = {id: SimilarityRecord(tokenizer) for id in intents_info}
    client = OpenAI()
    generate_response = OpenAiGenerateResponse(client=client, model="gpt-4o mini", system_prompt=system_prompt)
    
    intents_queries = {
        id: [] for id in intents_info
    }
    with open(arg.seed, 'r') as f:
        for line in f:
            j = json.loads(line)
            id = j['id']
            intents_queries[id].append(j)
            records[id].add(j['query'])
            
    print(f"loaded seed tasks")
    
    num_generated = 0
    try:
        with open(arg.output, 'r') as f:
            for line in f:
                j = json.loads(line)
                id = j['id']
                intents_queries[id].append(j)
                records[id].add(j['query'])
                num_generated += 1
        print(f"loaded {num_generated} generated tasks")
    except FileNotFoundError:
        print(f"{arg.output} not found, creating a new file")
                        
    intents_ids = list(intents_info.keys())
    json_formatter = JsonFormatter()
    json_parser = JsonParser()
    
    with open('prompt.txt', 'r') as f:
        prompt = f.read()
    
    with tqdm(total=arg.num_data) as pbar:
        pbar.update(num_generated)
        with open(arg.output, 'a') as f:
            while num_generated < arg.num_data:
                sampled_intent_id = random.choice(intents_ids)
                slot = json.dumps(intents_info[sampled_intent_id], ensure_ascii=False)
                prompts = list(generate_prompts_(prompt, slot, intents_queries[id], arg.batch_size, arg.num_tasks, 
                                                 formatter=json_formatter))
                print(f"prompt: {prompts}")
                resps = generate_response('', prompts)
                
                # print all responses
                for resp in resps:
                    print(f'\n\nresp: {resp["text"]}\n\n\n')
                
                for resp in resps:
                    if resp['finish_reason'] == 'length':
                        continue
                    for task in parse_input(resp['text'], parser=json_parser):
                        most_similar, score = records[sampled_intent_id].update(task['query'], arg.similarity_bound)
                        if score > arg.similarity_bound:
                            print(f"similarity score: {score}\n task: {task}\n most similar: {most_similar}\n")
                            continue
                        
                        f.write(json.dumps(task, ensure_ascii=False) + '\n')
                        f.flush()
                        pbar.update(1)
                        num_generated += 1
                    
            
    
