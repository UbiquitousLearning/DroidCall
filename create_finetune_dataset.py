import json
from string import Template
from transformers import AutoTokenizer
import argparse
import os

from utils.prompt import SYSTEM_PROMPT_FOR_FUNCTION_CALLING, NESTED_CALLING_PROMT, FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL

SYSTEM_PROMPT = SYSTEM_PROMPT_FOR_FUNCTION_CALLING
NEST_PROMPT = NESTED_CALLING_PROMT
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

def format_xlam_instruction(instruction):
    functions = "\n".join([json.dumps(func, indent=2, ensure_ascii=False) for func in instruction["tools"]])
    user_message = PROMPT_FOR_CHATMODEL.substitute(
        user_query=instruction['query'],
        functions=functions,
        nest_prompt=NEST_PROMPT,
        example=""
    )
    chat = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": user_message
        },
        {
            "role": "assistant",
            "content": format_assistant_response(instruction['answers'])
        }
    ]
    return chat
        
HANDLER_MAP = {
    "xlam": format_xlam_instruction,
    "glaive": format_glaive_instruction,
    "DroidCall": format_xlam_instruction
}

def process_instructions(input_file, output_file, format_instruction=format_xlam_instruction, tokenizer_path=None):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path) if tokenizer_path else None
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
            
        for instruction in instructions:
            # Format the instruction into a chat
            chat = format_instruction(instruction)
            
            if tokenizer:
                text = tokenizer.apply_chat_template(chat, tokenize=False)
            
                # Write the chat to the output file
                outfile.write(json.dumps({"text": text, "messages": chat}, ensure_ascii=False) + "\n")
            else:
                outfile.write(json.dumps({"messages": chat}, ensure_ascii=False) + "\n")
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process chat instructions')
    parser.add_argument('input_file', type=str, help='Input file path')
    parser.add_argument('output_file', type=str, help='Output file path')
    parser.add_argument('--tokenizer', type=str, default=None, help='Path to tokenizer')
    parser.add_argument('--handler', type=str, default='xlam', choices=HANDLER_MAP.keys(), help='Handler for formatting the instructions')
    
    args = parser.parse_args()
    
    # Fetch the appropriate handler function based on the command line argument
    selected_handler = HANDLER_MAP[args.handler]
    
    # Process the instructions with the selected handler and tokenizer
    process_instructions(args.input_file, args.output_file, selected_handler, args.tokenizer)
    