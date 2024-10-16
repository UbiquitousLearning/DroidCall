#!/bin/bash

# 定义实验参数
declare -a model_names=("Qwen2.5-1.5B-Instruct")
declare -a model_paths=("/data/share/Qwen2.5-1.5B-Instruct")  # 单一元素数组
declare -a task_names=(
    "DroidCall-checkpoint-200" 
    "DroidCall-checkpoint-250"
    "DroidCall-checkpoint-300"
    "DroidCall-checkpoint-350"
)
declare -a adapter_paths=(
    "../other/LLaMA-Factory/saves/Qwen2.5-1.5B-Instruct_DroidCall/checkpoint-200"
    "../other/LLaMA-Factory/saves/Qwen2.5-1.5B-Instruct_DroidCall/checkpoint-250"
    "../other/LLaMA-Factory/saves/Qwen2.5-1.5B-Instruct_DroidCall/checkpoint-300"
    "../other/LLaMA-Factory/saves/Qwen2.5-1.5B-Instruct_DroidCall/checkpoint-350"
)

# 找到最长的数组长度
max_length=0
for array in model_names task_names model_paths adapter_paths; do
    eval "length=\${#$array[@]}"
    if (( length > max_length )); then
        max_length=$length
    fi
done

# 扩展单一元素的数组
for array in model_names task_names model_paths adapter_paths; do
    eval "length=\${#$array[@]}"
    if (( length == 1 && length < max_length )); then
        eval "single_value=\${$array[0]}"
        eval "$array=()"
        for ((i=0; i<max_length; i++)); do
            eval "$array+=(\"$single_value\")"
        done
    fi
done

# 共通参数
RETRIEVE_DOC_NUM=4
HANDLER=lora_causal_lm
IS_NESTED=true
ADD_EXAMPLES=false
TABLE_PREFIX="llama_factory"

# 遍历所有实验参数组合
for ((i=0; i<max_length; i++)); do
    MODEL_NAME=${model_names[i]}
    TASK_NAME=${task_names[i]}
    MODEL_PATH=${model_paths[i]}
    ADAPTER_PATH=${adapter_paths[i]}

    # 设置嵌套和添加示例的标志
    NEST_FLAG=""
    FEW_SHOT_FLAG=""

    if [ "$IS_NESTED" = true ]; then
        NEST_FLAG="--is_nested"
    fi

    if [ "$ADD_EXAMPLES" = true ]; then
        FEW_SHOT_FLAG="--add_examples"
    fi

    # 运行 gen_solution.py
    python gen_solution.py \
        --retrieve_doc_num "$RETRIEVE_DOC_NUM" \
        --model_name "$MODEL_NAME" \
        --handler "$HANDLER" \
        --path "$MODEL_PATH" \
        --adapter_path "$ADAPTER_PATH" \
        --task_name "$TASK_NAME" \
        $NEST_FLAG $FEW_SHOT_FLAG

    # 运行 result_checker.py
    python result_checker.py \
        --input "results/${HANDLER}_${MODEL_NAME}_${TASK_NAME}_result.jsonl" \
        --model_name "$MODEL_NAME" \
        --task_name "$TASK_NAME" \
        --table_prefix "$TABLE_PREFIX"
done
