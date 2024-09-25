# LLM for Android Function calling

## 数据生成

### api定义

在`api.py`中定义可供调用的`function`，包含每个`function`的描述信息，可参考已有的`api.py`。

定义好`api.py`后，通过如下命令来提取可用函数的结构化信息
```bash
python extract.py --output path/to/output_file
```
若不指定`--output`，默认输出为`data/api.jsonl`，每一行内容形式如下
```json
{
  "name": "ACTION_INSERT_EVENT",
  "description": "Add a new event to the user's calendar.\n",
  "arguments": {
    "TITLE": {
      "description": "The event title.",
      "type": "str",
      "required": true
    },
    "DESCRIPTION": {
      "description": "The event description.",
      "type": "str",
      "required": true
    },
    ...
  }
}
```

### 准备xlam_function_calling数据集

在生成数据时，需要采样一些示例样例给LLM作参考，以此激活LLM的ICL能力。自行编写示例工作量大，因此在此利用已有的工作生成的数据作为样例给到LLM。此处选择了APIGen生成的`xlam_function_calling_60k`数据集。在使用前，需要将其整理成本工作相同的格式，使用如下命令
```bash
python extract.py --handler xlam --input xlam_function_calling_60k --output data/processed_xlam.jsonl
```
这将在`data`下生成`processed_xlam.jsonl`数据文件，用于后续采样样例时使用。

### 生成数据集
准备好`api.jsonl`和`processed_xlam.jsonl`后可以用如下命令生成数据集
```bash
python gen_instructions.py --output data/instructions.jsonl \
    --num_generate <num_data_per_function> \
    --sample_num <num_sampler_for_icl> \
    --model_class gpt \
    --model_name gpt-4o \
    --similarity_threshold 0.75
```
每个选项解释如下:
- output: 输出数据文件位置
- num_generate: 为每一个function产生多少条instruction数据
- sample_num: 采样条数
- model_class: 使用什么模型，目前有deepseek和gpt
- model_name: 模型名称
- similarity_threshold: 去除重复数据的阈值（rouge_score）



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

