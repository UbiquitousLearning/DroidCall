from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
from peft import PeftModel
import torch
import wandb
import argparse
import random

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="path/to/model", help="Path to the model")
parser.add_argument("--model_name", type=str, default="Qwen2.5-1.5B-Instruct", help="Model name")
parser.add_argument("--data_path", type=str, default="data/finetune/DroidCall_train.jsonl", help="Path to the data")
parser.add_argument("--eval_steps", type=int, default=200, help="Number of steps to evaluate")
parser.add_argument("--use_wandb", action="store_true", help="use wandb to log or not")
parser.add_argument("--lora_r", type=int, default=8, help="Lora r")
parser.add_argument("--lora_alpha", type=int, default=32, help="Lora alpha")
parser.add_argument("--lora_dropout", type=float, default=0.08, help="Lora dropout")
parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Per device train batch size")
parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Gradient accumulation steps")
parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="Learning rate")
parser.add_argument("--epochs", type=int, default=24, help="Number of epochs")
parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="Per device eval batch size")
parser.add_argument("--additional_lora", type=str, default="", help="merge additional lora adapter")
arg = parser.parse_args()

DATA_PATH = arg.data_path 
MODEL_PATH = arg.model_path 
OUTPUT_DIR = "checkpoint"

LEARNING_RATE = arg.learning_rate
EPOCHS = arg.epochs

LORA_R = arg.lora_r
LORA_ALPHA = arg.lora_alpha
LORA_DROPOUT = arg.lora_dropout
PER_DEVICE_TRAIN_BATCH_SIZE = arg.per_device_train_batch_size
PER_DEVICE_EVAL_BATCH_SIZE = arg.per_device_eval_batch_size
GRADIENT_ACCUMULATION_STEPS = arg.gradient_accumulation_steps

LORA_ADAPTER_PATH = arg.additional_lora

if arg.use_wandb:
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="function calling",
        
        entity="huggingface",

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
    
# module to finetune
TARGET_MODULES_MAP = {
    "Qwen": "all-linear",
    "llama": "all-linear",
    "phi": "all-linear",
    "minicpm": "all-linear",
    "gemma-2-2b-it": "all-linear",
}


def remove_system_prompt(example):
    messages = example["messages"]
    messages = [m for m in messages if m["role"] != "system"]
    return {"messages": messages}

PROCESS_MAP = {
    # "gemma-2-2b-it": remove_system_prompt, # we modified prompt template of gemma-2-2b-it so no need to remove system prompt
}

ATTN_MAP = {
    "gemma-2-2b-it": "eager",
}


if __name__ == "__main__":
    dataset = load_dataset("json", data_files=DATA_PATH)["train"]
    
    train_size = int(0.96 * len(dataset))  
    eval_size = len(dataset) - train_size

    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, train_size + eval_size))
    lower_name = arg.model_name.lower()
    for name in PROCESS_MAP:
        if name in lower_name:
            train_dataset = train_dataset.map(PROCESS_MAP[name])
            eval_dataset = eval_dataset.map(PROCESS_MAP[name])
            break
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, 
                                                 device_map="cuda", 
                                                 attn_implementation=ATTN_MAP.get(lower_name, "flash_attention_2"),
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    
    print(f"model: {model}")
    
    if LORA_ADAPTER_PATH:
        model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)
        model = model.merge_and_unload()
        print(f"merge lora adapter: {LORA_ADAPTER_PATH}\nmodel:{model}")
    
    target_modules = None
    
    for k, v in TARGET_MODULES_MAP.items():
        if k.lower() in arg.model_name.lower():
            target_modules = v
            break
    
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    
    peft_model = get_peft_model(model, peft_config)
    
    # print(f"target modules: {peft_model.targeted_module_names}")
    
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
        max_seq_length=8192,
    )
    
    
    
    
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/{arg.model_name}")
