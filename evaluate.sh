#!/bin/bash

export CUDA_VISIBLE_DEVICES=5

RETRIEVE_DOC_NUM=2
MODEL_NAME=QWen2.5-coder
HANDLER=hf_causal_lm
MODEL_PATH=/data/share/QWen2.5-coder
ADAPTER_PATH=
TASK_NAME=instruct
IS_NESTED=false
ADD_EXAMPLES=false

NEST_FLAG=""
FEW_SHOT_FLAG=""

if [ $IS_NESTED = true ]; then
    NEST_FLAG="--is_nested"
fi

if [ $ADD_EXAMPLES = true ]; then
    FEW_SHOT_FLAG="--add_examples"
fi

# python gen_solution.py \
#     --retrieve_doc_num "$RETRIEVE_DOC_NUM" \
#     --model_name "$MODEL_NAME" \
#     --handler "$HANDLER" \
#     --path "$MODEL_PATH" \
#     --adapter_path "$ADAPTER_PATH" \
#     --task_name "$TASK_NAME" \
#     $NEST_FLAG $FEW_SHOT_FLAG

python result_checker.py \
    --input "results/${HANDLER}_${MODEL_NAME}_${TASK_NAME}_result.jsonl" \
    --model_name $MODEL_NAME \
    --task_name $TASK_NAME
    
