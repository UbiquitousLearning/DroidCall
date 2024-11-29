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
        pass # TODO: Implement this
