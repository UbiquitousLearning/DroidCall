import json
from string import Template
from transformers import AutoTokenizer

SYSTEM_PROMPT = """
You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. You should only return the function call in tools call sections.
"""

PROMPT_FOR_CHATMODEL = Template("""
Here is a list of functions in JSON format that you can invoke:
$functions

Should you decide to return the function call(s), Put it in the format of 
<function_call> {
    "name": "func1",
    "arguments": {
        "arg1": "value1",
        "arg2": "value2",
        ...
    }
}
<function_call> {
    "name": "func2",
    "arguments": {
        "arg1": "value1",
        "arg2": "value2",
        ...
    }
}
...
If an argument is a response from a previous function call, you can reference it in the following way:
<function_call> {
    "name": "func3",
    "arguments": {
        "arg1": "value1",
        "arg2": "<function_response>func1",
        ...
    }
}
NO other text MUST be included. 

Now my query is: $user_query
""")

def process_instructions(input_file, output_file, tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            instruction = json.loads(line)
            
            # Format the functions
            functions = "\n".join([json.dumps(func, indent=2, ensure_ascii=False) for func in 
                                   instruction["tools"]])
            
            # Create the user message using the template
            user_message = PROMPT_FOR_CHATMODEL.substitute(
                user_query=instruction['query'],
                functions=functions
            )
            
            # Create the chat format
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
            
            text = tokenizer.apply_chat_template(chat, tokenize=False)
            
            # Write the chat to the output file
            outfile.write(json.dumps({"text": text, "chat": chat}, ensure_ascii=False) + "\n")
            

def format_assistant_response(answers):
    return "\n".join([f"<function_call> {json.dumps(answer, indent=2, ensure_ascii=False)}" for answer in answers])

if __name__ == "__main__":
    input_file = 'data/processed_xlam.jsonl'
    output_file = 'xlam_function_calling.jsonl'
    tokenizer_path = "../xLLM/tokenizer_smollm"
    process_instructions(input_file, output_file, tokenizer_path)
    