from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
import torch
import wandb

DATA_PATH = "data/train_dataset.jsonl" # "data/finetune_dataset.jsonl"
MODEL_PATH =  "/data/share/Qwen2-1.5B-Instruct" # "model/HuggingFaceTB/SmolLM"
OUTPUT_DIR = "checkpoint"

LEARNING_RATE = 1.41e-5
EPOCHS = 3

LORA_R = 4
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
PER_DEVICE_TRAIN_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 16

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="function calling",
    
    entity="shrelic",

    # track hyperparameters and run metadata
    config={
        "learning_rate": LEARNING_RATE,
        "architecture": "SmolLM",
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
    train_size = int(0.95 * len(dataset))  
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
        output_dir=f"{OUTPUT_DIR}/Qwen2-1.5B-Instruct",
        report_to="all",
        logging_dir="log",
        packing=False,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=EPOCHS,
        logging_steps=1,
        save_steps=20,
        load_best_model_at_end=True,
        eval_strategy="steps",
        eval_steps=20,
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
    trainer.save_model(f"{OUTPUT_DIR}/Qwen2-1.5B-Instruct")
