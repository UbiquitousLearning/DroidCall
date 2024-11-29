import json
from string import Template
from transformers import AutoTokenizer
import argparse
import os
os.sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../")
from utils.formatter import MessageTemplate
import random

parser = argparse.ArgumentParser(description='Process chat instructions')
parser.add_argument('input_file', type=str, help='Input file path')
parser.add_argument('output_file', type=str, help='Output file path')
parser.add_argument('--tokenizer', type=str, default=None, help='Path to tokenizer')
parser.add_argument('--handler', type=str, default='DroidCall', choices=["xlam", "glaive", "DroidCall"], help='Handler for formatting the instructions')
parser.add_argument("--api_num", type=int, default=4, help="Number of API to retrieve in a query")
parser.add_argument("--format", type=str, default="code_short", choices=["json", "code", "code_short", "json_short"], help="Format of the output")
parser.add_argument("--sep_start", type=str, default="$", help="Start separator for function call")
parser.add_argument("--sep_end", type=str, default="$", help="End separator for function call")

args = parser.parse_args()

from utils.prompt import SYSTEM_PROMPT_FOR_FUNCTION_CALLING, JSON_NESTED_CALLING_PROMT, FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL
import random

SYSTEM_PROMPT = SYSTEM_PROMPT_FOR_FUNCTION_CALLING
NEST_PROMPT = JSON_NESTED_CALLING_PROMT
PROMPT_FOR_CHATMODEL = Template(FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL)

GLAIVE_SYSTEM_PROMPT = Template("""
You are a helpful assistant with access to the following functions. Use them if required - $functions
""")

def format_assistant_response(answers):
    return json.dumps(answers, indent=2, ensure_ascii=False)

def format_glaive_instruction(instruction):
    messages = instruction["messages"]
    chat = [
        {
            "role": "system", 
            "content": GLAIVE_SYSTEM_PROMPT.substitute(functions=json.dumps(instruction["tools"], indent=2, ensure_ascii=False))
        },
    ]
    for message in messages:
        if not isinstance(message["content"], str):
            str_content = json.dumps(message["content"], indent=2, ensure_ascii=False)
        else:
            str_content = message["content"]
        chat.append({
            "role": message["role"],
            "content": str_content
        })
        
    return chat

def xlam_wrapper(type:str="json", **kwargs):
    message_template = MessageTemplate.get_message_template(type)
    sep_start = kwargs.get("sep_start", "")
    sep_end = kwargs.get("sep_end", "")
    message_template.set_function_call_sep(sep_start, sep_end)
    
    def format_xlam_instruction(instruction):
        chat = message_template.format(instruction)["message"]
        return chat
    
    return format_xlam_instruction

def DroidCall_wrapper(type:str="json", api_file: str="data/api.jsonl", **kwargs):
    message_template = MessageTemplate.get_message_template(type)
    sep_start = kwargs.get("sep_start", "")
    sep_end = kwargs.get("sep_end", "")
    message_template.set_function_call_sep(sep_start, sep_end)
    
    name2api = {}
    with open(api_file, 'r', encoding='utf-8') as f:
        for line in f:
            api = json.loads(line)
            name2api[api["name"]] = api
            
    n_api = args.api_num
    
    def format_DroidCall_instruction(instruction):
        tools = instruction["tools"]
        used_names = [tool["name"] for tool in tools]
        if len(tools) < n_api:
            available_api_names = [api for api in name2api.keys() if api not in used_names]
            additional_api_names = random.sample(available_api_names, min(n_api - len(tools), len(available_api_names)))
            tools.extend([name2api[api_name] for api_name in additional_api_names])
        
        random.shuffle(tools)
        instruction["tools"] = tools
        chat = message_template.format(instruction)["message"]
        return chat
    
    return format_DroidCall_instruction
        
HANDLER_MAP = {
    "xlam": xlam_wrapper(args.format, sep_start=args.sep_start, sep_end=args.sep_end),
    "glaive": format_glaive_instruction,
    "DroidCall": DroidCall_wrapper(args.format, sep_start=args.sep_start, sep_end=args.sep_end),
}

def process_instructions(input_file, output_file, format_instruction, tokenizer_path=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if tokenizer_path else None
    # if directory of output file does not exist, create it
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        instructions = []
        _, ext = os.path.splitext(input_file)
        # print(f"ext: {ext}")
        if ext == ".jsonl": 
            for line in infile:
                instruction = json.loads(line)
                instructions.append(instruction)
        else:
            all = json.load(infile)
            instructions = all
            
        output_is_jsonl = output_file.endswith('.jsonl')
        formatted_instructions = []

        total_num = 0
        total_token_num = 0
        for instruction in instructions:
            # Format the instruction into a chat
            total_num += 1
            chat = format_instruction(instruction)
            
            if tokenizer:
                text = tokenizer.apply_chat_template(chat, tokenize=False)
                total_token_num += len(tokenizer(text)["input_ids"])
                formatted_instruction = {"text": text, "messages": chat}
            else:
                formatted_instruction = {"messages": chat}
            
            if output_is_jsonl:
                outfile.write(json.dumps(formatted_instruction, ensure_ascii=False) + "\n")
            else:
                formatted_instructions.append(formatted_instruction)
        
        if not output_is_jsonl:
            json.dump(formatted_instructions, outfile, ensure_ascii=False, indent=2)
        
        print(f"Total number of instructions: {total_num}")
        if tokenizer:
            print(f"Average number of tokens: {total_token_num/total_num}")
            

if __name__ == "__main__":
    # Fetch the appropriate handler function based on the command line argument
    selected_handler = HANDLER_MAP[args.handler]
    
    # Process the instructions with the selected handler and tokenizer
    process_instructions(args.input_file, args.output_file, selected_handler, args.tokenizer)
    