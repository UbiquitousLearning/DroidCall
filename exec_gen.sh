#!/bin/bash

# 开始ID
start_id=3
# 结束ID
end_id=45

# 循环从 start_id 到 end_id
for id in $(seq $start_id $end_id)
do
    echo "Running python script with --id $id"
    TOKENIZERS_PARALLELISM=false python gen_intents_query.py --model_path ../xLLM/tokenizer_qwen2/ --id $id
done
