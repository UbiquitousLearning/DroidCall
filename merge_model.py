from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="/data/share/Qwen2.5-1.5B-Instruct", help="Base model")
parser.add_argument("--adapter", type=str, default="", help="Adapter")
parser.add_argument("--output", type=str, default="checkpoint/merged", help="Output model")
args = parser.parse_args()

import requests

if __name__ == "__main__":
    # load the base model and the adapter
    # merge them into a new model
    # save the new model to the output path
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base_model, args.adapter)
    merge_model = model.merge_and_unload()
    print(f"merge model: {merge_model}")
    merge_model.save_pretrained(args.output)
    
