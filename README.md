# LLM for Android Function calling

## 数据生成

暂略

## evaluation

### 使用LLM生成答案

用如下命令使用LLM对qeury数据生成相应答案
```bash
python gen_solution.py --input path/to/query_data\ # query文件路径，默认为data/filtered_data.jsonl一般使用默认的即可
    --retrieve_doc_num number\ # RAG中检索文档数量，默认为2
    --model_name model_name\ # 模型名称
    --handler handler_name\ # 选择handler，目前可选的有hf_causal_lm和openai，hf_causal_lm使用transformers来使用LLM生成输出，openai调用openai的api生成输出
    --path model_path\ # handler为hf_causal_lm时，该参数指定本地模型路径
    --retriever retriever_name\ # 可选fake或chromadb，fake一定会给出正确的文档，默认为fake
    --task_name name # 控制输出文件名称
```
以上在`results`目录下生成名称为`{HANDLER}_{MODEL_NAME}_{arg.task_name}_result.jsonl`的文件记录模型处理query的结果。

### 测试生成答案准确率

使用如下命令测试模型生成答案准确率
```bash
python result_checker.py --input input_file_path\ # 输入文件，选中上一节中的输出文件即可
    --answer answer_file_path\ # 选择答案文件，默认为./data/annotation_data.jsonl，一般选择默认的即可
    --output output_file\ # 输出记录文件默认为results/accuracy.json，一般选择默认的即可
    --model_name model_name\  # 影响记录名称
    --task_name task_name # 影响记录名称
```

以上命令会在`output`中生成一条名称为`model_name_task_name`这样的记录，值为模型答案正确率。

