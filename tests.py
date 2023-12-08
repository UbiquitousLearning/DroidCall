from transformers import AutoTokenizer
from utils import *

model_path = "/data/shrelic/data/hfl/chinese-alpaca-2-7b"

def test_similarity_record():
    sentences = [
        "I like to eat apples.",
        "每一天都是美好的一天呀",
        "I really think 每天 is 美好的一天",
        "I do not like to eat apples.",
        "我不喜欢吃apples"
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    r = SimilarityRecord(tokenizer)
    print(f"num proc: {r.num_processes}")
    
    for sentence in sentences:
        print(r.update(sentence))
        print(r.sentences)
        print("--------------------")

if __name__ == '__main__':
    test_similarity_record()

