from dataclasses import dataclass, field
import requests

@dataclass
class Call:
    name: str
    arguments: dict = field(default_factory=dict)
    
@dataclass
class Result:
    state: str
    message: str
    return_type: str
    return_value: any = None

class Executor:
    def __init__(self, url: str, verbose: bool = False):
        self.url = url
        self.functions = {}
        self.verbose = verbose
        
    def register(self, func):
        self.functions[func.__name__] = func
    
    def execute(self, call: Call)->Result:
        if self.verbose:
            print(f"executing call: {call}")
            
        if call.name in self.functions:
            func = self.functions[call.name]
            try:
                # 将参数字典展开为关键字参数
                result_value = func(**call.arguments)
                return Result(state="Success", message="Function executed successfully.", return_type=str(type(result_value)), return_value=result_value)
            except TypeError as e:
                return Result(state="Error", message=f"Argument mismatch: {e}", return_type="None")
            except Exception as e:
                return Result(state="Error", message=str(e), return_type="None")
        else:
            # sent request to the server
            response = requests.post(self.url, json={"name": call.name, "arguments": call.arguments})
            if response.status_code != 200:
                return Result(state="Error", message=f"Request failed with status code {response.status_code}", return_type="None")
            
            result = response.json()
            return Result(**result)
