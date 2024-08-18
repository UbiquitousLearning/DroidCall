from string import Template
import json
from utils import SimilarityRecord, OpenAiGenerateResponse, extract_and_parse_jsons
from transformers import AutoTokenizer
import random
from openai import OpenAI

INIT_PROMPT = Template("""
我需要你帮我生成一些function calling的数据集，我会给你一个tool的描述，你需要根据这个tool生成一些query以及对应的answer，即调用tool的答案来
解决用户的query。下面是我的一些要求：
1. 对于query，尽可能的使用不同的词汇、句法来保证query的多样性。query可以长、可以短、可以复杂也可以简洁，总之尽量不要生成类似的query，我希望能保证query的多样性。
2. query的语言尽可能的保持多样性，也就是说这个query可以是一个命令、可以是一个问题、也可以带有详细描述的请求等等。
3. 生成的query要尽可能的覆盖tool的所有可能的用法，即各个参数覆盖性要全面，保证能使用tool来完成各种形式的工作。
4. 生成的query要可以使用给出的tools来解决。
5. 对于你生成的query，你要给出使用tool解决的answer，即给出使用的tool和对应的各个参数的值。
6. 给出参数时，若一个参数required=False，则可以不给出相应的值。
7. query answer对要尽可能的覆盖tool的可能的所有用法。
8. 生成的数据一定要按照我样例中的格式给出。

下面是一些样例:
tool:
{
    "name": "live_giveaways_by_type",
    "description": "Retrieve live giveaways from the GamerPower API based on the specified type.",
    "arguments": {
        "type": {
            "description": "The type of giveaways to retrieve (e.g., game, loot, beta).",
            "required": true,
            "type": "str",
            "default": "game"
        }
    }
}
response: 
{
  "query": "Where can I find live giveaways for beta access and games?",
  "answers": [
    {
      "name": "live_giveaways_by_type",
      "arguments": {
        "type": "beta"
      }
    }]
}

接下来我将给你一个tool，你帮我生成10条query answer对
tool: $tool
""")

INIT_PROMPT = Template("""
I need your help to generate some function calling datasets. I will provide you with a tool description, and you need to generate queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, you may omit its value.
7. The query-answer pairs should cover as many possible uses of the tool as possible.
8. The generated data must be presented in the format given in my example.

following are some examples:
$examples

Now I will give you a tool, and you help me generate 10 query-answer pairs.
REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE
tool: $tool
""")

def format_example(example):
    tool = example["tools"][0]
    resp = {
        "query": example["query"],
        "answers": example["answers"]
    }
    return f'tool: {json.dumps(tool, indent=2)}\nresponse: {json.dumps(resp, indent=2)}'


if __name__ == "__main__":
    all_examples = []
    with open("data/processed_xlam.jsonl", "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            if example["tools_num"] == 1 and example["answers_num"] == 1:
                all_examples.append(example)
        
    path = "../xLLM/tokenizer_qwen2"
    tokenizer = AutoTokenizer.from_pretrained(path)

    records = SimilarityRecord(tokenizer)
    client = OpenAI()
    generate_response = OpenAiGenerateResponse(client=client, model="gpt-4o-mini", system_prompt="")

    with open("data/api.jsonl") as f:
        all_tools = [json.loads(line) for line in f.readlines()]

    examples = random.sample(all_examples, 2)
    examples_text = "\n".join([format_example(example) for example in examples])
    tool = random.choice(all_tools)
    tool_text = json.dumps(tool, indent=4)
    prompt_text = INIT_PROMPT.substitute(examples=examples_text, tool=tool_text)
    print(prompt_text)
    print("\n\n")
    
    resps = generate_response('', [prompt_text])
    
    for resp in resps:
        if resp["finish_reason"] == "stop":
            print(f'text: {resp["text"]}\n\n')
            for j in extract_and_parse_jsons(resp["text"]):
                print(json.dumps(j, indent=2, ensure_ascii=False))
    
