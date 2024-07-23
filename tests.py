from transformers import AutoTokenizer
from openai import OpenAI
from utils import *

model_path = "./tokenizer_qwen2"

def test_similarity_record():
    sentences = [
        "I like to eat apples.",
        "每一天都是美好的一天呀",
        "I really think 每天 is 美好的一天",
        "I do not like to eat apples.",
        "我不喜欢吃apples"
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hug_tokenizer = HuggingFaceTokenizer(tokenizer)
    
    r = SimilarityRecord(hug_tokenizer)
    print(f"num proc: {r.num_processes}")
    
    for sentence in sentences:
        print(f'updating: {sentence}')
        print(r.update(sentence))
        print(r.sentences)
        print("--------------------")

if __name__ == '__main__':
    # test_similarity_record()
    
    # client = OpenAI()
    # gen = OpenAiGenerateResponse(client=client, model="gpt-3.5-turbo", system_prompt="You are a kind and helpful person, always ready to help others.")
    # res = gen("Hello,", ["How are you?", "What are you doing?"])
    # print(res)
    
    with open('tmp.json', 'r') as f:
        j = json.load(f)
    
    keys = ['id', 'query', 'intent', 'mime', 'uri', 'extras']
    with open('seeds.jsonl', 'a') as f:
        for item in j:
            item = {k: item[k] for k in keys}
            f.write(json.dumps(item) + '\n')
    
    # id = 1
    # with open('intents-common.jsonl', 'r') as f:
    #     with open('intents.jsonl', 'w') as f2:
    #         for line in f:
    #             j = json.loads(line)
    #             j['id'] = id
    #             f2.write(json.dumps(j) + '\n')
    #             id += 1
            
            
        
        

