import inspect
import importlib
from typing import Optional, Dict, Any
from docstring_parser import parse
import json
import argparse
from typing import Generator


def get_json_description(func):
    """
    Get the JSON schema for the given function.

    Args:
        func (Callable): The function to get the JSON schema for.

    Returns:
        Dict[str, Any]: A dictionary representing the JSON schema for the function.
    """
    # {
    #     "name": "get_json_description",
    #     "description": "Get the JSON schema for the given function.\n",
    #     "arguments": {
    #         "func": {
    #             "description": "The function to get the JSON schema for.",
    #             "type": "Callable",
    #             "required": true # when no default value is provided
    #         }
    #     },
    #     "returns": {
    #         "description": "A dictionary representing the JSON schema for the function.",
    #         "type": "Dict[str, Any]"
    #     }
    # }
    
    doc = inspect.getdoc(func)
    
    parsed_doc = parse(doc)
    
    desc = {
        "name": func.__name__,
        "description": parsed_doc.description,
        "arguments": {}
    }
    
    signature = inspect.signature(func)
    
    for param in parsed_doc.params:
        item = {
            "description": param.description,
            "type": param.type_name,
            "required": False
        }
        
        for p in signature.parameters.values():
            if p.name == param.arg_name:
                if p.default is inspect.Parameter.empty:
                    item["required"] = True
                else:
                    item["default"] = p.default
        
        desc["arguments"][param.arg_name] = item
        
    if parsed_doc.returns and parsed_doc.returns.type_name and parsed_doc.returns.description.strip().lower() != "none":
        desc["returns"] = {
            "description": parsed_doc.returns.description,
            "type": parsed_doc.returns.type_name
        }
        
    if parsed_doc.examples:
        desc["examples"] = [
            example.description for example in parsed_doc.examples
        ]
        
    return desc
        

parser = argparse.ArgumentParser(description="Extract API documentation from a Python module.")
parser.add_argument("--handler", type=str, default="py", help="specify the handler function to extract the API documentation from",
                    choices=["py", "xlam"])
parser.add_argument("--input", type=str, default="api", help="input to extract from")
parser.add_argument("--output", type=str, default="data/api.jsonl", help="output file to store extracted apis")

args = parser.parse_args()

def py_extract(input: str)->Generator[dict[str, Any], None, None]:
    module_name = input
    module = importlib.import_module(module_name)
    functions = inspect.getmembers(module, inspect.isfunction)
    for _, function in functions:
        # get the function that is defined in the module
        if function.__module__ == module_name:
            yield get_json_description(function)


def convert_xlam_func_to_desc(func_obj):
    desc = {
        "name": func_obj["name"],
        "description": func_obj["description"],
        "arguments": {},
    }
    for key, value in func_obj["parameters"].items():
        item = {
            "description": value["description"],
            "required": True
        }
        
        tmp = value["type"].split(",")
        if len(tmp) <= 1:
            tmp = tmp + [""]
        
        if len(tmp) > 2:
            # print(f"warning: {key} has more than 2 values in type")
            # print(tmp)
            # print(json.dumps(func_obj, indent=2))
            tmp = tmp[:2]
            
        type_name, is_optional = tmp
        item["type"] = type_name
        if is_optional == "optional":
            item["required"] = False
        
        if "default" in value:
            item["default"] = value["default"]
        
        desc["arguments"][key] = item
        
    return desc


def xlam_extract(input: str)->Generator[dict[str, Any], None, None]:
    with open(input, "r") as f:
        data = json.load(f)
    
    for item in data:
        query = item["query"]
        answers = json.loads(item["answers"])
        for i in range(len(answers)):
            answers[i]["id"] = i
        # print(f"answers: {answers}")
        
        tools = json.loads(item["tools"])
        
        yield {
            "query": query,
            "answers": answers,
            "tools": [
                    convert_xlam_func_to_desc(tool) for tool in tools
                ],
            "tools_num": len(tools),
            "answers_num": len(answers),
        }
    

def glaive_extract(input: str)->Generator[dict[str, Any], None, None]:
    pass


HANDLER_MAP =  {
    "py": py_extract,
    "xlam": xlam_extract
}

if __name__ == "__main__":
    extractor = HANDLER_MAP[args.handler]
    with open(args.output, "a") as f:
        for api in extractor(args.input):
            # print(f"{api["answers"]}")
            f.write(json.dumps(api, ensure_ascii=False) + "\n")
    
