# DroidCall

`DroidCall` is the first training and testing dataset for accurate Android Intent invocation constructed by a highly flexible and reusable data generation pipeline.

## Function Predefinition

The first step of the `DroidCall` workflow is to predefine the function we want models to learn to use. We have done that in the [`api.py`](api.py), we just use the way python define functions to predefine functions we need, and use google style docstring to describe it.

Once finishing predefinition, extract the `json` format description with the following command
```bash
python extract.py 
```
That would extract the functions description from [`api.py`](api.py) and generate [`api.jsonl`](data/api.jsonl) in [`data`](data) directory.

> note that we already have the result file in this repo, this command will append json at the tail of [`api.jsonl`](data/api.jsonl). So if you want to define your own function, just delete the original one and generate it yourself.

## Dataset Construction

### Download [`xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)

When generating seed data, we will use data points in [`xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) as examples to introduce the ICL capability of GPT-4, making it to generate better seed data. Use the following command to download the dataset
```bash
cd data
mkdir function_call
huggingface-cli download --repo-type dataset\
  Salesforce/xlam-function-calling-60k xlam_function_calling_60k.json\
  --local-dir . --local-dir-use-symlinks False
```
you will download the `xlam_function_calling_60k.json` and save it to `DroidCall/data/function_call`.

Then you need to process the [`xlam-function-calling-60k`](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k) dataset to the same format we use in our work, use the following command
```bash
# make sure you are in DroidCall dir
python extract.py --handler xlam \
  --input data/function_call/xlam_function_calling_60k.json \
  --output data/function_call/processed_xlam.jsonl
```

### Data Generation
> **Note:** If you don't want to generate data by yourself and want to use our generated data, you can download the data from [DroidCall](https://huggingface.co/datasets/mllmTeam/DroidCall). Put `DroidCall_*.jsonl` in the `data` directory.

Once you've done the above steps, you can use the following command to generate simple data
```bash
python gen_instructions.py --tokenizer_path path/to/tokenizer \  # we use qwen2 tokenizer
--num_generate 300 \  # the minimum data points to generate for a single function
--similarity_threshold 0.75 \  # rouge score to filter out similar data
--sample_num 8 \  # examples in prompt to guide LLM to generate
--model_class gpt \  # currently only gpt and deepseek available
--model_name gpt-4-turbo  # model name
```
This command will generate a file named `instructions.jsonl` in the `data` directory.

Use the following command to generate complex data
```bash
python gen_complex_instructions.py --tokenizer_path path/to/tokenizer \  # we use qwen2 tokenizer
--num_generate 300 \  # the minimum data points to generate for a single function
--similarity_threshold 0.75 \  # rouge score to filter out similar data
--sample_num 8 \  # examples in prompt to guide LLM to generate
--model_class gpt \  # currently only gpt and deepseek available
--model_name gpt-4-turbo  # model name
```
This will generate a file named `instructions_complex.jsonl` in the data directory.

Next use the tool script [`split_data.py`](scripts/split_data.py) to combine the above two file and shuffle and split into train and test split.
```bash
python scripts/split_data.py --files data/instructions.jsonl data/instructions_complex.jsonl --num_test 200  # the number of samples in test set
```

After that you will see `DroidCall_train.jsonl` and `DroidCall_test.jsonl` in the data directory.

>tokenizer is used to tokenize text so that we can calculate rouge score

## Finetuning with DroidCall
### Preparation for Finetuning
Use following command to produce chat format data for fine-tuning
```bash
python scripts/create_finetune_dataset.py data/DroidCall_train.jsonl data/finetune/DroidCall_train.jsonl --format code_short
```
format can be one of:
- code_short
- code
- json_short
- json
details can be found in out paper.

### Finetune SLMs
We provide a simple training script [`finetune_llm.py`](scripts/finetune_llm.py). You can use the following command to start training
```bash
CUDA_VISIBLE_DEVICES=... accelerate launch scripts/finetune_llm.py --model_path path/to/model --model_name model_name
```

Checkpoints and the saved model can be found in `checkpoint/model_name`.

You can review the file to see how to adjust the training hyperparameters.

### Merge
We use lora to finetune SLMs. Sometimes we need to merge the lora adapter into the original model. Use the following command to merge:
```bash
python scripts/merge_model.py --base_model path/to/base_model --adapter path/to/adapter --output output_path
```

>Note: the original prompt template of gemme does not support system prompt, so we ajust its prompt template.

## evaluation
> **Note:** If you don't want to generate `annotated_api.jsonl` yourself, you can just download it from [DroidCall](https://huggingface.co/datasets/mllmTeam/DroidCall) and put it in `data` directory.

When compare the parameters with the ground truth, some parameters are considered to be correct if they are semantically similar (e.g. title, query). So when conducting evaluations, we need to know whether a certain parameter should be compared precisely or semantically. We use LLM to annotate every parameters, use the following command to generate `annotated_api.jsonl` in `data` directory:
```bash
python scripts/annotate.py
```

We write a simple program to record evaluation result.
```bash
python recorder/server.py
```
This will start a server listening 8989 port.  The server will receive result and write in csv file in `table` directly.

Then we can use [`evaluate.sh`](evaluate.sh) to evaluate SLMs.Below are some configurations you need to fill out
```bash
declare -a model_names=(
    "gpt-4o"
    "gpt-4o-mini"
)

declare -a model_paths=(
    "path/to/Model-A"
    "path/to/Model-B"
)

declare -a task_names=(
    "task-A"
    "task-B"
)

declare -a adapter_paths=(
    "adapter-A"
    "adapter-B"
)

# the number of function docs to retrieve
RETRIEVE_DOC_NUM=4 

# this can be one of
#  - openai: use openai api
#  - deepseek: use deepseek api
#  - hf_causal_lm: use huggingface transformers
#  - lora_causal_lm: use huggingface transformers with a lora adapter
HANDLER=openai

# few-shot or not
ADD_EXAMPLES=false

# table name to record result
# this determines the CSV file name where server.py stores the results.
TABLE_PREFIX="naive"

# this can be one of
# - json: this is good when testing zero-shot accuracy
# - json_short
# - code
# - code_short: default format for finetuning
FORMAT_TYPE="json"
```
Use following command to start evaluation:
```bash
CUDA_VISIBLE_DEVICES=... ./evaluate.sh
```

Result can be found at `results` and `table` folder.

For example, if you want to test the Zero-shot `gemma-2-2b-it` and `gemma-3-4b-it`, you can modify the `evaluate.sh` as following:
```bash
declare -a model_names=(
    "gemma-2-2b-it"  # just give a model name you want
    "gemma-3-4b-it"  
)

declare -a model_paths=(
    "path/to/gemma-2-2b-it" # path to model
    "path/to/gemma-3-4b-it"
)

declare -a task_names=(
    "zero-shot"     # just give a task name yourself
    "zero-shot"
)

declare -a adapter_paths=(
    "adapter-A"  # if no lora adapter, this will be ignored
    "adapter-B"
)

# the number of function docs to retrieve
RETRIEVE_DOC_NUM=4 

# this can be one of
#  - openai: use openai api
#  - deepseek: use deepseek api
#  - hf_causal_lm: use huggingface transformers
#  - lora_causal_lm: use huggingface transformers with a lora adapter
HANDLER=hf_causal_lm

# few-shot or not
ADD_EXAMPLES=false # this is zero-shot

# table name to record result
# this determines the CSV file name where server.py stores the results.
TABLE_PREFIX="naive" # just give a name

# this can be one of
# - json: this is good when testing zero-shot accuracy
# - json_short
# - code
# - code_short: default format for finetuning
FORMAT_TYPE="json"
```
