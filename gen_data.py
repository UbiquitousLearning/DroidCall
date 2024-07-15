import argparse
import os
import random
import json
from tqdm import tqdm 
from utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    parser = argparse.ArgumentParser(description='args for gen_data.py')
    parser.add_argument('--seed', type=str, help='seed file', default='seeds.jsonl')
    parser.add_argument('--output', type=str, help='output file', default='machine_generated.jsonl')
    parser.add_argument('--model_path', type=str, help='tokenizer and model path', default='hfl/chinese-alpaca-2-7b')
    parser.add_argument('--similarity_bound', type=float, help='similarity bound to filter prompts', default=0.7)
    parser.add_argument('--num_data', type=int, help='number of data to generate', default=5000)
    parser.add_argument('--batch_size', type=int, help='batch size', default=1)
    parser.add_argument('--num_tasks', type=int, help='number of tasks used in prompt', default=3)

    arg = parser.parse_args()
    
    print(f'started to load model from {arg.model_path}')
    huggingface_tokenizer = AutoTokenizer.from_pretrained(arg.model_path)
    tokenizer = HuggingFaceTokenizer(huggingface_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(arg.model_path, device_map="auto")
    print(f"model loaded to {model.device}")
    
    record = SimilarityRecord(tokenizer)
    generate_response = GenerateResponse(huggingface_tokenizer, model)
    
    all_tasks = []
    with open(arg.seed, 'r') as f:
        for line in f:
            j = json.loads(line)
            all_tasks.append(j)
            record.add(j['input'])
            
    print(f"loaded {len(all_tasks)} seed tasks")
    
    num_generated = 0
    try:
        with open(arg.output, 'r') as f:
            for line in f:
                j = json.loads(line)
                all_tasks.append(j)
                num_generated += 1
                record.add(j['input'])
        print(f"loaded {num_generated} generated tasks")
    except FileNotFoundError:
        print(f"{arg.output} not found, creating a new file")
                        
    with tqdm(total=arg.num_data) as pbar:
        pbar.update(num_generated)
        with open(arg.output, 'a') as f:
            while num_generated < arg.num_data:
                prompts = list(generate_prompts_(all_tasks, arg.batch_size, arg.num_tasks))
                resps = generate_response('', prompts,
                                          max_new_tokens=2049, do_sample=True, top_k=50, top_p=0.95, temperature=0.9, num_return_sequences=1, pad_token_id=tokenizer.pad_token_id
                                            , eos_token_id=tokenizer.eos_token_id)
                
                for resp in resps:
                    for task in parse_input(**resp):
                        most_similar, score = record.update(task['input'], arg.similarity_bound)
                        if score > arg.similarity_bound:
                            print(f"similarity score: {score}\n task: {task}\n most similar: {most_similar}\n")
                            continue
                        
                        f.write(json.dumps(task, ensure_ascii=False) + '\n')
                        f.flush()
                        pbar.update(1)
                        num_generated += 1
                    
            
    
