#!/bin/bash

declare -a model_names=(
    "Model-A"
    "Model-B"
)

declare -a model_paths=(
    "path/to/Model-A"
    "path/to/Model-B"
)

declare -a task_names=(
    "task1"
    "task2"
)

declare -a adapter_paths=(
    "adapter1"
    "adapter2"
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
HANDLER=hf_causal_lm
IS_NESTED=true
ADD_EXAMPLES=true
TABLE_PREFIX="naive"
FORMAT_TYPE="json"
SEP_START="$"
SEP_END="$"


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

    # run gen_solution.py
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

    # run result_checker.py
    python result_checker.py \
        --input "results/${HANDLER}_${MODEL_NAME}_${TASK_NAME}_result.jsonl" \
        --model_name "$MODEL_NAME" \
        --task_name "$TASK_NAME" \
        --table_prefix "$TABLE_PREFIX"
done
