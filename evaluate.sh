#!/bin/bash

declare -a model_names=(
    "gpt-4-turbo"
)

declare -a model_paths=(
    ""
)

declare -a task_names=(
    "few-shot"
)

declare -a adapter_paths=(
    "adapter"
)

max_length=0
for array in model_names task_names model_paths adapter_paths; do
    eval "length=\${#$array[@]}"
    if (( length > max_length )); then
        max_length=$length
    fi
done

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

RETRIEVE_DOC_NUM=4
HANDLER=openai
IS_NESTED=true
ADD_EXAMPLES=true
TABLE_PREFIX="naive"
FORMAT_TYPE="json"
SEP_START="$"
SEP_END="$"

# 确保 results 目录存在
mkdir -p results

for ((i=0; i<max_length; i++)); do
    MODEL_NAME=${model_names[i]}
    TASK_NAME=${task_names[i]}
    MODEL_PATH=${model_paths[i]}
    ADAPTER_PATH=${adapter_paths[i]}

    NEST_FLAG=""
    FEW_SHOT_FLAG=""

    if [ "$IS_NESTED" = true ]; then
        NEST_FLAG="--is_nested"
    fi

    if [ "$ADD_EXAMPLES" = true ]; then
        FEW_SHOT_FLAG="--add_examples"
    fi

    # 定义相关文件路径
    OUTPUT_FILE="results/${HANDLER}_${MODEL_NAME}_${TASK_NAME}_result.jsonl"
    PASS_FILE="results/${HANDLER}_${MODEL_NAME}_${TASK_NAME}_result_pass.jsonl"
    FAIL_FILE="results/${HANDLER}_${MODEL_NAME}_${TASK_NAME}_result_fail.jsonl"

    # 检查是否需要运行 gen_solution.py
    if [ ! -f "$OUTPUT_FILE" ]; then
        echo "Running gen_solution.py for ${MODEL_NAME} ${TASK_NAME}..."
        python gen_solution.py \
            --retrieve_doc_num "$RETRIEVE_DOC_NUM" \
            --model_name "$MODEL_NAME" \
            --handler "$HANDLER" \
            --path "$MODEL_PATH" \
            --adapter_path "$ADAPTER_PATH" \
            --task_name "$TASK_NAME" \
            --format_type "$FORMAT_TYPE" \
            --sep_start "$SEP_START" \
            --sep_end "$SEP_END" \
            $NEST_FLAG $FEW_SHOT_FLAG
    else
        echo "Skipping gen_solution.py for ${MODEL_NAME} ${TASK_NAME}, output file already exists."
    fi

    # 检查是否需要运行 result_checker.py
    if [ ! -f "$PASS_FILE" ] || [ ! -f "$FAIL_FILE" ]; then
        echo "Running result_checker.py for ${MODEL_NAME} ${TASK_NAME}..."
        python result_checker.py \
            --input "$OUTPUT_FILE" \
            --model_name "$MODEL_NAME" \
            --task_name "$TASK_NAME" \
            --table_prefix "$TABLE_PREFIX"
    else
        echo "Skipping result_checker.py for ${MODEL_NAME} ${TASK_NAME}, result files already exist."
    fi
done
