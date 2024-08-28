import json
import random

if __name__ == "__main__":
    instructions = {}
    with open("data/instructions.jsonl", "r") as f:
        for line in f:
            inst = json.loads(line)
            func_name = inst["answers"][0]["name"]
            if func_name not in instructions:
                instructions[func_name] = []
            instructions[func_name].append(inst)
            
    with open("data/instructions_train.jsonl", "w") as train_f, open("data/instructions_test.jsonl", "w") as test_f:
        for func_name, insts in instructions.items():
            random.shuffle(insts)
            test_num = 8
            train_insts = insts[:-test_num]
            test_insts = insts[-test_num:]
            
            for inst in train_insts:
                train_f.write(json.dumps(inst, ensure_ascii=False) + "\n")
            
            for inst in test_insts:
                test_f.write(json.dumps(inst, ensure_ascii=False) + "\n")
    
