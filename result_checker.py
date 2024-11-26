import json
import argparse
from bert_score import score
import os

parser = argparse.ArgumentParser(description='check result of generation of GPT')
parser.add_argument("--input", type=str, default="./results/openai_gpt-4o-mini_result.jsonl", help="Path to the input file")
parser.add_argument("--api", type=str, default="./data/annotated_api.jsonl", help="Path to the answer file")
parser.add_argument("--output", type=str, default="./results/accuracy.json", help="Path to the output accuracy file")
parser.add_argument("--model_name", type=str, default="openai_gpt-4o-mini", help="Model name")
parser.add_argument("--task_name", type=str, default="", help="Task name")
parser.add_argument("--table_prefix", type=str, default="", help="Table prefix")
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
    if obj == None:
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
        # for item1, item2 in zip(obj1, obj2):
        #     if not deep_compare(item1, item2, ftype=ftype):
        #         return False
        
        # no need to follow the order
        # here [1, 2, 2] [1, 2, 3] will return True
        # so we need to check double direction
        for item1 in obj1:
            found = False
            for item2 in obj2:
                if deep_compare(item1, item2, ftype=ftype):
                    found = True
                    break
            if not found:
                return False
        
        for item2 in obj2:
            found = False
            for item1 in obj1:
                if deep_compare(item1, item2, ftype=ftype):
                    found = True
                    break
            if not found:
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
    
    
def add_suffix(filename, suffix):
    # 分离文件名和扩展名
    name_part, ext_part = filename.rsplit('.', 1)
    
    # 在文件名部分添加后缀，并重新拼接扩展名
    new_filename = f"{name_part}_{suffix}.{ext_part}"
    
    return new_filename

def check_result(resp, answer, apis_info):
    correct_num = 0
    total_num = 0
    # check if answers is in response
    resp_map = {item.get("name", ""): item for item in resp if isinstance(item, dict) 
                and isinstance(item.get("name", ""), str)}
    for ans in answer:
        func_name = ans["name"]
        api_info = apis_info[func_name]
        if func_name not in resp_map:
            total_num += len(api_info["arguments"])
            continue
        
        resp_ans = resp_map[func_name]
        for arg_name, arg_value in api_info["arguments"].items():
            if arg_name not in ans["arguments"] and arg_name not in resp_ans["arguments"]:
                correct_num += 1
                total_num += 1
            else:
                if arg_value["required"] and arg_name not in ans["arguments"]:
                    total_num += 1
                    continue
                arg_default = arg_value.get("default", None)
                resp_arg_value = resp_ans["arguments"].get(arg_name, arg_default)
                ans_arg_value = ans["arguments"].get(arg_name, arg_default)
                # print(f"{arg_value}")
                check_type = arg_value.get("match_type", "strict")
                if deep_compare(ans_arg_value, resp_arg_value, ftype=check_type):
                    correct_num += 1
                    total_num += 1
                else:
                    total_num += 1
    return correct_num, total_num
    

from recorder import update    

def main():
    results = []
    with open(arg.input, "r") as fin:
        for line in fin:
            try:
                results.append(json.loads(line))
            except:
                pass
            
    
    apis_info = {}
    with open(arg.api, "r") as fin:
        for line in fin:
            item = json.loads(line)
            apis_info[item["name"]] = item
            
    
    pass_output = add_suffix(arg.input, "pass")
    fail_output = add_suffix(arg.input, "fail")
    
    
    with open(pass_output, "w") as pass_fout, open(fail_output, "w") as fail_fout:
        score = 0
        total_correct_num = 0
        for result in results:
            resp, answer = result["response"], result["answers"]
            correct_num, total_num = check_result(resp, answer, apis_info)
            if total_num == 0:
                delta_score = 1.0
            else:
                delta_score = correct_num / total_num
            if abs(delta_score - 1) < 1e-6:
                total_correct_num += 1
                pass_fout.write(json.dumps(result) + "\n")
            else:
                fail_fout.write(json.dumps(result) + "\n")
            score += delta_score
        
        accuracy = score / len(results)
        total_correct_accuracy = total_correct_num / len(results)
    
    if os.path.exists(arg.output): 
        with open(arg.output, "r") as fin:
            acc = json.load(fin)
    else:
        acc = {}
    
    if arg.model_name not in acc:
        acc[arg.model_name] = {}
    
    acc[arg.model_name][arg.task_name] = {
        "soft_accuracy": accuracy,
        "accuracy": total_correct_accuracy,
    }
    
    with open(arg.output, "w") as fout:
        json.dump(acc, fout, indent=4)
        
    update(f"{arg.table_prefix}-accuracy", arg.model_name, arg.task_name, total_correct_accuracy)
    update(f"{arg.table_prefix}-soft_accuracy", arg.model_name, arg.task_name, accuracy)


if __name__ == "__main__":
    main()

