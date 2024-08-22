import json

from utils import extract_and_parse_jsons

response_text = """
```json
I have two cats
[
    {"name": "fluffy", "age": 3},
    {"name": "whiskers", "age": 4}
]

and I also have a dog
{"name": "fido", "age": 7}
```"""

from utils import JsonExtractor, DataFilter
from typing import Dict

def check_format(data):
    if "query" not in data or "answers" not in data:
        return False
    if not isinstance(data["query"], str):
        return False
    if not isinstance(data["answers"], list):
        return False
    for ans in data["answers"]:
        if not isinstance(ans, dict):
            return False
        if "name" not in ans or "arguments" not in ans:
            return False
        if not isinstance(ans["arguments"], dict):
            return False
    return True

class FormatFilter(DataFilter):
    def validate(self, data: Dict[str, str]) -> bool:
        return check_format(data)

if __name__ == "__main__":
    filters = [JsonExtractor()]
    resps = [{"text": response_text, "finish_reason": "stop"}]
    for filter in filters:
        resps = filter.filter(resps)
    for resp in resps:
        print(resp)
    
