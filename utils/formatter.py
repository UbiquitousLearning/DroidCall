from typing import Dict
from string import Template
import json
from .prompt import *

class Formatter:
    def __init__(self, template: str, **kwarg) -> None:
        self.sub_formatters: Dict[str, Formatter]= {}
        self.template= Template(template)
        self.sub_formatters = {}
        for key, value in kwarg.items():
            if isinstance(value, Formatter):
                self.sub_formatters[key]= value
    
    def _get_sub_formatters_result(self, **kwargs)->Dict[str, str]:
        result= {}
        for key, sub_formatter in self.sub_formatters.items():
            result[key]= sub_formatter.format(**kwargs)
        return result

    def __getattr__(self, name):
        if name in self.sub_formatters:
            return self.sub_formatters[name]
        raise AttributeError(f"'Formatter' object has no attribute '{name}'")
    
    def format(self, **kwargs)->str:
        format_res = self._get_sub_formatters_result(**kwargs)
        # for k, v in format_res.items():
        #     print(f"{k}: {v}\n\n")
        # print(f"template: {self.template}")
        return self.template.substitute(format_res)
    

class ConstantFormatter(Formatter):
    def __init__(self, constant: str) -> None:
        super().__init__("")
        self.constant= constant
    
    def format(self, **kwargs)->str:
        return self.constant
    
class FieldFormatter(Formatter):
    def __init__(self, field_name: str) -> None:
        super().__init__("")
        self.field_name= field_name
    
    def format(self, **kwargs)->str:
        res = kwargs.get(self.field_name, "")
        assert isinstance(res, str)
        return res
    
class FunctionCallingFormatter(Formatter):
    def __init__(self, sep_start: str="", sep_end="") -> None:
        super().__init__("")
        self.sep_start = sep_start
        self.sep_end = sep_end
    
    def set_sep(self, sep_start: str, sep_end: str):
        self.sep_start = sep_start
        self.sep_end = sep_end
        
class ConstantCallingFormatter(FunctionCallingFormatter):
    def __init__(self, constant: str, sep_start: str="", sep_end="") -> None:
        super().__init__(sep_start, sep_end)
        self.constant = constant
    
    def format(self, **kwargs) -> str:
        return self.sep_start + self.constant + self.sep_end
    
class JsonFunctionCallingFormatter(FunctionCallingFormatter):
    def format(self, **kwargs) -> str:
        calls = kwargs.get('calls', [])
        return self.sep_start + json.dumps(calls, ensure_ascii=False, indent=2) + self.sep_end
    
from typing import Any

class CodeFunctionCallingFormatter(FunctionCallingFormatter):
    def _format_call(self, call: Dict[str, Any])->str:
        call_id = call.get('id', 0)
        function_name = call.get('name', '')
        arguments = call.get('arguments', {})

        # Helper function to format each argument based on its type
        def format_arg(value):
            if isinstance(value, str):
            # 检查字符串是否以'#'开头，并且后面是数字
                if value.startswith("#") and value[1:].strip().isdigit():
                    return f'result{value[1:].strip()}'  # 返回格式为"resultn"的字符串
                else:
                    return f'"{value}"'  # 对其他字符串添加引号
            elif isinstance(value, list):
                return json.dumps(value, ensure_ascii=True)  # Format lists to string representation
            elif isinstance(value, dict):
                return json.dumps(value, ensure_ascii=True)
                
            return str(value)  # Use str for other data types

        # Format all arguments using format_arg function
        argument_str = ', '.join(f'{key}={format_arg(value)}' for key, value in arguments.items())
        result_str = f'result{call_id} = {function_name}({argument_str})'
        
        return result_str
    
    def format(self, **kwargs) -> str:
        calls = kwargs.get('calls', [])
        return self.sep_start + '\n'.join(self._format_call(call) for call in calls) + self.sep_end
        

class GetFunctionExampleFormatter(Formatter):
    def __init__(self, examples_file:str, call_formatter: FunctionCallingFormatter) -> None:
        super().__init__("")
        self.examples = []
        with open(examples_file) as f:
            for line in f:
                item = json.loads(line)
                if len(item["answers"]) == 1:
                    self.examples.append(item)
        
        self.call_formatter = call_formatter
    
    def format(self, **kwargs)->str:
        result_text = "Following are some examples:\n"
        
        for tool in kwargs["tools"]:
            for example in self.examples:
                ok = False
                for ans in example["answers"]:
                    if ans["name"] == tool["name"]:
                        ok = True
                        break
                if ok:
                    result_text += f"query: {example['query']}\nanswers: \n{self.call_formatter.format(calls=example['answers'])}\n\n"
                    break
        return result_text
    
class FunctionFormatter(Formatter):
    def __init__(self, format_type: str = "json"):
        super().__init__("")
        self.format_type = format_type
        if self.format_type not in self._format_methods:
            raise ValueError(f"Unsupported format_type {self.format_type}")

    @staticmethod
    def _format_json(tools: dict) -> str:
        return json.dumps(tools, ensure_ascii=False, indent=2)

    @staticmethod
    def _format_code_single_tool(tool) -> str:
        # Initialize an empty list to hold each line of the formatted description
        formatted_lines = []

        # Extracting and formatting the basic information
        assert "name" in tool, "The tool must have a name"
        assert "description" in tool, "The tool must have a description"
        name = tool["name"]
        formatted_lines.append(f"Name:\n    {name}")
        
        description = tool.get('description', '').strip()
        formatted_lines.append(f"Description:\n    {description}")

        # Formatting the arguments section if 'arguments' key exists
        if 'arguments' in tool:
            args_description = "Args:"
            for arg, details in tool['arguments'].items():
                arg_type = details.get('type', 'Unknown Type')
                arg_description = details.get('description', '').strip()
                args_description += f"\n    {arg} ({arg_type}): {arg_description}"
            formatted_lines.append(args_description)
        else:
            formatted_lines.append("Args:\n    None")

        # Formatting the returns section if 'returns' key exists
        if 'returns' in tool:
            return_type = tool['returns'].get('type', 'Unknown Return Type')
            return_description = tool['returns'].get('description', '').strip()
            returns_description = f"Returns:\n    {return_type}: {return_description}"
            formatted_lines.append(returns_description)
        else:
            formatted_lines.append("Returns:\n    None")

        # Formatting the example section if 'examples' key exists
        if 'examples' in tool:
            example_text = "Example:\n"
            for example in tool['examples']:
                example_text += f"    {example.strip()}\n"
            formatted_lines.append(example_text.strip())

        # Joining all parts to form the full description
        full_description = "\n".join(formatted_lines)
        return full_description
        

    @staticmethod
    def _format_code(tools: dict) -> str:
        tools_description = [FunctionFormatter._format_code_single_tool(tool) for tool in tools]
        sep = "\n" + "="*50 + "\n"
        return sep.join(tools_description)

    _format_methods = {
        'json': _format_json,
        'code': _format_code
    }

    def format(self, **kwargs) -> str:
        formatter = self._format_methods[self.format_type]
        return formatter(kwargs.get("tools", {}))
        


class MessageTemplate:
    SYSTEM_MESSAGE_MAP: Dict[str, Formatter] = {
        "json_with_examples": ConstantFormatter(SYSTEM_PROMPT_FOR_FUNCTION_CALLING),
        "code_with_examples": ConstantFormatter(SYSTEM_PROMPT_FOR_FUNCTION_CALLING),
        "json": ConstantFormatter(SYSTEM_PROMPT_FOR_FUNCTION_CALLING),
        "code": ConstantFormatter(SYSTEM_PROMPT_FOR_FUNCTION_CALLING),
        "code_short": ConstantFormatter(SHORT_SYSTEM_PROMPT_FOR_FUNCTION_CALLING),
        "json_short": ConstantFormatter(SHORT_SYSTEM_PROMPT_FOR_FUNCTION_CALLING)
    }
    
    USER_MESSAGE_MAP: Dict[str, Formatter] = {
        "json_with_examples": Formatter(
            FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL,
            functions=FunctionFormatter(format_type="json"),
            call_format=ConstantCallingFormatter(JSON_CALL_FORMAT),
            nest_prompt=ConstantFormatter(JSON_NESTED_CALLING_PROMT),
            example=GetFunctionExampleFormatter("data/DroidCall_train.jsonl", JsonFunctionCallingFormatter()),
            user_query=FieldFormatter("query"),
        ),
        "code_with_examples": Formatter(
            FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL,
            functions=FunctionFormatter(format_type="code"),
            call_format=ConstantCallingFormatter(CODE_CALL_FORMAT),
            nest_prompt=ConstantFormatter(CODE_NESTED_CALLING_PROMPT),
            example=GetFunctionExampleFormatter("data/DroidCall_train.jsonl", JsonFunctionCallingFormatter()),
            user_query=FieldFormatter("query"),
        ),
        "json": Formatter(
            FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL,
            functions=FunctionFormatter(format_type="json"),
            call_format=ConstantCallingFormatter(JSON_CALL_FORMAT),
            nest_prompt=ConstantFormatter(JSON_NESTED_CALLING_PROMT),
            example=ConstantFormatter(""),
            user_query=FieldFormatter("query"),
        ),
        "code": Formatter(
            FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL,
            functions=FunctionFormatter(format_type="code"),
            call_format=ConstantCallingFormatter(CODE_CALL_FORMAT),
            nest_prompt=ConstantFormatter(CODE_NESTED_CALLING_PROMPT),
            example=ConstantFormatter(""),
            user_query=FieldFormatter("query"),
        ),
        "code_short": Formatter(
            SHORT_FUNCTION_CALLING_PROMPT,
            functions=FunctionFormatter(format_type="code"),
            call_format=ConstantFormatter(""),
            nest_prompt=ConstantFormatter(""),
            example=ConstantFormatter(""),
            user_query=FieldFormatter("query"),
        ),
        "json_short": Formatter(
            SHORT_FUNCTION_CALLING_PROMPT,
            functions=FunctionFormatter(format_type="json"),
            call_format=ConstantFormatter(""),
            nest_prompt=ConstantFormatter(""),
            example=ConstantFormatter(""),
            user_query=FieldFormatter("query"),
        )
    }
    
    ASSISTANT_MESSAGE_MAP: Dict[str, Formatter] = {
        "json_with_examples": JsonFunctionCallingFormatter(),
        "code_with_examples": CodeFunctionCallingFormatter(),
        "json": JsonFunctionCallingFormatter(),
        "code": CodeFunctionCallingFormatter(),
        "code_short": CodeFunctionCallingFormatter(),
        "json_short": JsonFunctionCallingFormatter()
    }
    
    @staticmethod
    def get_message_template(type: str)->"MessageTemplate":
        system_message_formatter = MessageTemplate.SYSTEM_MESSAGE_MAP[type]
        user_message_formatter = MessageTemplate.USER_MESSAGE_MAP[type]
        assistant_message_formatter = MessageTemplate.ASSISTANT_MESSAGE_MAP[type]
        return MessageTemplate(system_message_formatter, user_message_formatter, assistant_message_formatter)
    
    def __init__(self, system_formatter: Formatter, user_formatter: Formatter, assistant_formatter: Formatter) -> None:
        self.system_formatter = system_formatter
        self.user_formatter = user_formatter
        self.assistant_formatter = assistant_formatter
    
    def set_function_call_sep(self, sep_start: str, sep_end: str):
        self.assistant_formatter.set_sep(sep_start, sep_end)
        
        if isinstance(self.user_formatter.call_format, FunctionCallingFormatter):
            self.user_formatter.call_format.set_sep(sep_start, sep_end)
        # self.user_formatter.nest_prompt.set_sep(sep_start, sep_end)
        if isinstance(self.user_formatter.example, GetFunctionExampleFormatter):
            self.user_formatter.example.call_formatter.set_sep(sep_start, sep_end)
        
        
    def format(self, data: Dict[str, str], no_assistant: bool=False)->Dict[str, str]:
        if no_assistant:
            return {
                "message": [
                    {
                        "role": "system",
                        "content": self.system_formatter.format(**data)
                    },
                    {
                        "role": "user",
                        "content": self.user_formatter.format(**data)
                    }
                ]
            }
            
        return {
            "message": [
                {
                    "role": "system",
                    "content": self.system_formatter.format(**data)
                },
                {
                    "role": "user",
                    "content": self.user_formatter.format(**data)
                },
                {
                    "role": "assistant",
                    "content": self.assistant_formatter.format(calls=data.get("answers", []))
                }
            ]
        }
        
from transformers import AutoTokenizer

if __name__ == "__main__":
    # message_template = MessageTemplate.get_message_template("code_short")
    # message_template.set_function_call_sep("$", "$")
    
    # with open("data/DroidCall_train.jsonl") as f:
    #     all_data = [json.loads(line) for line in f]
    
    # import random
    # random.seed(55)
    # data = random.choice(all_data)
    # print(f"data: {json.dumps(data, ensure_ascii=False, indent=2)}")
    
    # print("*"*50)
    
    # # tokenizer = AutoTokenizer.from_pretrained("/data/share/Qwen2.5-1.5B-Instruct")
    # tokenizer = AutoTokenizer.from_pretrained("/data/shrelic/data/xllm-best_ckpt/xllm_sft")
    # message = message_template.format(data)
    # text = tokenizer.apply_chat_template(message["message"], tokenize=False, add_generation_prompt=False)
    # token_ids = tokenizer.tokenize(text)
    # print(f"text: {text}")
    
    # print(f"len: {len(token_ids)}")
    
    func_formatter = FunctionFormatter(format_type="code")
    all_api = {}
    with open("data/api.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            api = json.loads(line)
            all_api[api["name"]] = api
    
    print(func_formatter.format(tools=[all_api["ACTION_SET_ALARM"], all_api["web_search"], all_api["send_email"], all_api["dial"]]))
    
    