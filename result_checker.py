import json
import argparse
from bert_score import score
import os

parser = argparse.ArgumentParser(description='check result of generation of GPT')
parser.add_argument("--input", type=str, default="./results/openai_gpt-4o-mini_result.jsonl", help="Path to the input file")
parser.add_argument("--answer", type=str, default="./data/annotation_data.jsonl", help="Path to the answer file")
parser.add_argument("--output", type=str, default="./results/accuracy.json", help="Path to the output accuracy file")
parser.add_argument("--model_name", type=str, default="openai_gpt-4o-mini", help="Model name")
parser.add_argument("--task_name", type=str, default="", help="Task name")
arg = parser.parse_args()

def semantic_compare(a: str, b: str, threshold: float = 0.85):
    # compare two str using bert score
    cands = [a]
    refs = [b]
    _, _, F1 = score(cands, refs, lang="en")
    if F1[0] > threshold:
        return True
    return False


def is_field_none(obj):
    if not obj:
        return True

    if isinstance(obj, str) and obj.strip().lower() == "none":
        return True
    
    return False
    

def deep_compare(obj1, obj2, ftype="strict"):
    if ftype == "ignore":
        return True
    
    if is_field_none(obj1) and is_field_none(obj2):
        return True
    
    if type(obj1) != type(obj2):
        return False
    
    if isinstance(obj1, dict) and isinstance(obj2, dict):
        if len(obj1) != len(obj2):
            return False
        for key in obj1:
            if key not in obj2 or not deep_compare(obj1[key], obj2[key], ftype=ftype):
                return False
        return True
    
    elif isinstance(obj1, list) and isinstance(obj2, list):
        if len(obj1) != len(obj2):
            return False
        for item1, item2 in zip(obj1, obj2):
            if not deep_compare(item1, item2, ftype=ftype):
                return False
        return True
    
    elif isinstance(obj1, str) and isinstance(obj2, str):
        if ftype == "strict":
            return obj1.strip().lower() == obj2.strip().lower()
        else:
            return semantic_compare(obj1, obj2)
    
    elif isinstance(obj1, int) and isinstance(obj2, int):
        return obj1 == obj2
    
    return False


def check_with_type(candidate, ref, l, field_type):
    for key in l:
        if key not in candidate:
            return False
        if not deep_compare(candidate[key], ref[key], field_type[key]["type"]):
            return False
    
    return True


def check(candidate, ref):
    if "intent" not in candidate:
        return False
    
    if candidate["intent"] != ref["intent"]:
        return False
    
    # check uri and mime
    if not check_with_type(candidate, ref, ["uri", "mime"], ref["field_type"]):
        return False
    
    # check extras
    if "extras" not in candidate:
        return False
    
    if not check_with_type(candidate["extras"], ref["extras"], ref["extras"].keys(), ref["field_type"]["extras"]):
        return False
    
    return True
    
    
def add_suffix(filename, suffix):
    # 分离文件名和扩展名
    name_part, ext_part = filename.rsplit('.', 1)
    
    # 在文件名部分添加后缀，并重新拼接扩展名
    new_filename = f"{name_part}_{suffix}.{ext_part}"
    
    return new_filename


def main():
    with open(arg.input, "r") as fin:
        gpt_result = [json.loads(line) for line in fin]
    
    with open(arg.answer, "r") as fin:
        answer = [json.loads(line) for line in fin]
    
    correct_num = 0
    
    pass_output = add_suffix(arg.input, "pass")
    fail_output = add_suffix(arg.input, "fail")
    
    with open(pass_output, "w") as pass_fout, open(fail_output, "w") as fail_fout:
        for gpt_item, answer_item in zip(gpt_result, answer):
            assert gpt_item["query"] == answer_item["query"]
            answer_item.pop("query")
            curr_item = {
                "query": gpt_item["query"],
                "generate": gpt_item["response"],
                "answer": answer_item,
            }
            
            if is_field_none(gpt_item["response"]):
                fail_fout.write(json.dumps(curr_item) + "\n")
                continue
            
            if check(gpt_item["response"], answer_item):
                correct_num += 1
                pass_fout.write(json.dumps(curr_item) + "\n")
            else:
                fail_fout.write(json.dumps(curr_item) + "\n")
    
    print(f"Accuracy: {correct_num / len(gpt_result)}")
    
    if os.path.exists(arg.output): 
        with open(arg.output, "r") as fin:
            acc = json.load(fin)
    else:
        acc = {}
    
    acc[f"{arg.model_name}_{arg.task_name}"] = correct_num / len(gpt_result)
    
    with open(arg.output, "w") as fout:
        json.dump(acc, fout, indent=4)
    
        
    


if __name__ == "__main__":
    main()

