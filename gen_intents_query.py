import argparse
import random
import json
from tqdm import tqdm 
from utils import *
from transformers import AutoTokenizer
from openai import OpenAI
from string import Template


system_prompt = """
You are an Android development expert, skilled in understanding user needs and translating them into appropriate Android intent usage. Your task is to generate realistic user queries and provide solutions using Android intents.
"""

prompt_template = Template("""
Please help me come up with a set of 20 diverse user queries related to Android intents. For each query, you should provide a solution using the given intent information.

Here are the requirements:

1. Create diverse user queries that could be solved using the provided Android intent. The user query can be long, short, complex, or concise.
2. I want you to generate query only in English. Combine questions with statements of user needs or problems.
3. Represent realistic user scenarios or requests that can be addressed using the Android intent system.
4. Remenber to make the query more diverse, including different lengths and complexity. Some queries should be simple and direct, while others can be more detailed and elaborate.
5. When generating queries, consider the user's perspective and the context in which they would use the Android device. The queries should reflect the user's needs and preferences. And you should
   check if the query is reasonable and can be solved by the given intent.
6. For each query, provide a solution that use the given intent to address the user's need. The solution should include:
   - The intent action to be used
   - Any necessary URI, MIME type, or extras that should be included
   - A brief explanation of how this intent solves the user's query and what it does
7. Keep the solution concise and clear.
8. Include the intent data uri, mime, and extras if needed to solve the user's query.
9. Use realistic and diverse values for the extras fields that reflect the purpose of the intent.
10. Occasionally omit certain extras to show flexibility in intent usage, if applicable.
11. Include some edge cases or unusual requests to test the limits of the intent's functionality.
12. Format the result in JSON. Example: {"query":"...", "intent": "...", "mime": "...", "uri": "...", "extras": {...}, "explanation": "..."}
13. Keep in mind that the user queries should be varied and cover a wide range of scenarios that can be addressed using the Android intent system.
    And you must make different lengths and complexity of the queries, some should be simple and direct, while others can be more detailed and elaborate.
    The query can describe why the user needs to perform the action, what they want to achieve, or what problem they are facing.

Here's the intent information you should use:
$intent_info

Please provide a list of 20 user queries and their corresponding solutions using this intent. 

$examples

Now please generate the user queries and solutions.
""")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='args for gen_intents_query.py')
    parser.add_argument('--intents_file', type=str, help='file that stores the intents info', default='intents.jsonl')
    parser.add_argument('--seed', type=str, help='seed file', default='seeds.jsonl')
    parser.add_argument('--output', type=str, help='output file', default='machine_generated.jsonl')
    parser.add_argument('--model_path', type=str, help='tokenizer and model path', default='./tokenizer_qwen2')
    parser.add_argument('--similarity_bound', type=float, help='similarity bound to filter prompts', default=0.6)
    parser.add_argument('--num_data', type=int, help='number of data to generate', default=100)
    parser.add_argument('--num_tasks', type=int, help='number of tasks used in prompt', default=5)
    parser.add_argument('--id', type=int, help='the intent id to generate query', default=1)

    arg = parser.parse_args()
    
    print(f'started to load model from {arg.model_path}')
    huggingface_tokenizer = AutoTokenizer.from_pretrained(arg.model_path)
    tokenizer = HuggingFaceTokenizer(huggingface_tokenizer)
    
    intents_info = {}
    with open(arg.intents_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            j = json.loads(line)
            intents_info[j['id']] = j
            
    assert arg.id in intents_info, f"intent id {arg.id} not found in {arg.intents_file}"
    
    records = SimilarityRecord(tokenizer)
    client = OpenAI()
    generate_response = OpenAiGenerateResponse(client=client, model="gpt-4o-mini", system_prompt=system_prompt)
    
    intents_queries = []
    with open(arg.seed, 'r') as f:
        for line in f:
            j = json.loads(line)
            id = j['id']
            if id == arg.id:
                intents_queries.append(j)
                records.add(j['query'])
            
    print(f"loaded seed tasks")
    
    num_generated = 0
    try:
        with open(arg.output, 'r') as f:
            for line in f:
                j = json.loads(line)
                id = j['id']
                if id == arg.id:
                    intents_queries.append(j)
                    records.add(j['query'])
                    num_generated += 1
        print(f"loaded {num_generated} generated tasks")
    except FileNotFoundError:
        print(f"{arg.output} not found, creating a new file")
                        
    json_parser = JsonParser()
    
    with open('prompt.txt', 'r') as f:
        prompt = f.read()
    
    intent_desc = json.dumps(intents_info[arg.id], ensure_ascii=False, indent=2)
    with tqdm(total=arg.num_data) as pbar:
        pbar.update(num_generated)
        with open(arg.output, 'a') as f:
            while num_generated < arg.num_data:
                sampled_intents = random.sample(intents_queries, arg.num_tasks)
                
                prompt = prompt_template.substitute(intent_info=intent_desc, examples=json.dumps(sampled_intents, ensure_ascii=False, indent=2))
                print(f"prompt: {prompt}")
                resps = generate_response('', [prompt])
                
                # print all responses
                for resp in resps:
                    print(f'\n\nreason: {resp["finish_reason"]}\nresp: {resp["text"]}\n\n\n')
                
                for resp in resps:
                    if resp['finish_reason'] == 'length':
                        continue
                    for task in parse_input(resp['text'], parser=json_parser):
                        most_similar, score = records.update(task['query'], arg.similarity_bound)
                        if score > arg.similarity_bound:
                            print(f"similarity score: {score}\n task: {task}\n most similar: {most_similar}\n")
                            continue
                        task['id'] = arg.id
                        f.write(json.dumps(task, ensure_ascii=False) + '\n')
                        f.flush()
                        pbar.update(1)
                        num_generated += 1
                    
            
    
