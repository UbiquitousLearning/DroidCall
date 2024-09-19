from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
import torch
import wandb
import argparse
import random

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="/data/share/Qwen2-1.5B-Instruct", help="Path to the model")
parser.add_argument("--model_name", type=str, default="Qwen2-1.5B-Instruct", help="Model name")
parser.add_argument("--data_path", type=str, default="data/function_call/instruction_train.jsonl", help="Path to the data")
parser.add_argument("--eval_steps", type=int, default=200, help="Number of steps to evaluate")
parser.add_argument("--use_wandb", action="store_true", help="use wandb to log or not")
arg = parser.parse_args()

DATA_PATH = arg.data_path # "data/function_call/instruction_train.jsonl" # "data/function_call/glaive_train.jsonl" 
MODEL_PATH = arg.model_path # "/data/share/Qwen2-1.5B-Instruct" # "model/HuggingFaceTB/SmolLM"
OUTPUT_DIR = "checkpoint"

LEARNING_RATE = 1.41e-5
EPOCHS = 3

LORA_R = 4
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
PER_DEVICE_TRAIN_BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 64

if arg.use_wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="function calling",
        
        entity="shrelic",

        # track hyperparameters and run metadata
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": arg.model_name,
            "epochs": EPOCHS,
            "lora_r": LORA_R,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        }
    )

if __name__ == "__main__":
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    train_size = int(0.96 * len(dataset))  
    eval_size = len(dataset) - train_size

    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, train_size + eval_size))
    # print(dataset)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
                                                 device_map="auto", 
                                                #  attn_implementation="flash_attention_2",
                                                 torch_dtype=torch.bfloat16)
    
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    peft_model = get_peft_model(model, peft_config)
    
    sft_config = SFTConfig(
        output_dir=f"{OUTPUT_DIR}/{arg.model_name}",
        report_to="all" if arg.use_wandb else "tensorboard",
        logging_dir="log",
        packing=False,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        logging_steps=1,
        save_steps=arg.eval_steps,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=arg.eval_steps,
        save_total_limit=3,
    )
    
    trainer = SFTTrainer(
        peft_model, 
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
        # peft_config=peft_config,
        tokenizer=tokenizer,
    )
    
    
    
    
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/{arg.model_name}")
