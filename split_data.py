import json
import random
import argparse
from itertools import chain

parser = argparse.ArgumentParser(description='Process chat instructions')

parser.add_argument("--files", nargs="+", action="append", help="Input files")
parser.add_argument("--train_file", type=str, default="data/DroidCall_train.jsonl", help="Output train file")
parser.add_argument("--test_file", type=str, default="data/DroidCall_test.jsonl", help="Output test file")
parser.add_argument("--num_test", type=int, default=200, help="Number of test instructions")
args = parser.parse_args()

if __name__ == "__main__":
    files = list(chain(*args.files))
    all_lines = []
    for file in files:
        with open(file, "r") as f:
            all_lines.extend(f.readlines())
            
    # shuffle the lines
    # and put the first num_test lines into test file
    # and the rest into train file
    random.shuffle(all_lines)
    with open(args.train_file, "w") as f:
        for line in all_lines[args.num_test:]:
            f.write(line)
            
    with open(args.test_file, "w") as f:
        for line in all_lines[:args.num_test]:
            f.write(line)
            
    
