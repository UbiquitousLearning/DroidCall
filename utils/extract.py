import json
import re

import pyparsing as pp
from pyparsing import pyparsing_common as ppc

def convert_value(val):
    match = re.match(r'^result(\d+)$', val)
    if match:
        return f"#{match.group(1)}"
    
    if isinstance(val, str):
        if val.lower() in ["none", "null"]:
            return None
        if val.lower() == "true":
            return True
        if val.lower() == "false":
            return False

    try:
        return json.loads(val)
    except json.JSONDecodeError:
        if val.isdigit():
            return int(val)
        try:
            return float(val)
        except ValueError:
            return val

def extract_calls(calls_str):
    pattern = r'result(\d+) = (\w+)\((.*?)\)'
    matches = re.finditer(pattern, calls_str)
    
    for match in matches:
        call_id, function_name, arguments_str = match.groups()
        
        args_pattern = r'(\w+)=((?:\[.*?\]|{.*?}|".*?"|[^,]+))'
        arguments = {}

        for arg_name, arg_val in re.findall(args_pattern, arguments_str):
            arg_val = arg_val.strip()
            if arg_val.endswith(','):  
                arg_val = arg_val[:-1].strip()
            arguments[arg_name] = convert_value(arg_val)
        
        yield {
            "id": int(call_id),
            "name": function_name,
            "arguments": arguments
        }


def get_json_obj(text: str):
    def make_keyword(kwd_str, kwd_value):
        return pp.Keyword(kwd_str).setParseAction(pp.replaceWith(kwd_value))

    if not hasattr(get_json_obj, "jsonDoc"):
        # set to False to return ParseResults
        RETURN_PYTHON_COLLECTIONS = True

        TRUE = make_keyword("true", True)
        FALSE = make_keyword("false", False)
        NULL = make_keyword("null", None)

        LBRACK, RBRACK, LBRACE, RBRACE, COLON = map(pp.Suppress, "[]{}:")

        jsonString = pp.dblQuotedString().setParseAction(pp.removeQuotes)
        jsonNumber = ppc.number().setName("jsonNumber")

        jsonObject = pp.Forward().setName("jsonObject")
        jsonValue = pp.Forward().setName("jsonValue")

        jsonElements = pp.delimitedList(jsonValue).setName(None)

        jsonArray = pp.Group(
            LBRACK + pp.Optional(jsonElements) + RBRACK, aslist=RETURN_PYTHON_COLLECTIONS
        ).setName("jsonArray")

        jsonValue << (jsonString | jsonNumber | jsonObject | jsonArray | TRUE | FALSE | NULL)

        memberDef = pp.Group(
            jsonString + COLON + jsonValue, aslist=RETURN_PYTHON_COLLECTIONS
        ).setName("jsonMember")

        jsonMembers = pp.delimitedList(memberDef).setName(None)
        jsonObject << pp.Dict(
            LBRACE + pp.Optional(jsonMembers) + RBRACE, asdict=RETURN_PYTHON_COLLECTIONS
        )

        jsonComment = pp.cppStyleComment
        jsonObject.ignore(jsonComment)
        jsonDoc = jsonObject | jsonArray
        get_json_obj.jsonDoc = jsonDoc
    for _, l, r in get_json_obj.jsonDoc.scanString(text):
        json_string = text[l:r]
        try:
            parsed_data = json.loads(json_string)
            return parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")


def extract_and_parse_jsons(text):
    def make_keyword(kwd_str, kwd_value):
        return pp.Keyword(kwd_str).setParseAction(pp.replaceWith(kwd_value))

    if not hasattr(extract_and_parse_jsons, "jsonDoc"):
        # set to False to return ParseResults
        RETURN_PYTHON_COLLECTIONS = True

        TRUE = make_keyword("true", True)
        FALSE = make_keyword("false", False)
        NULL = make_keyword("null", None)

        LBRACK, RBRACK, LBRACE, RBRACE, COLON = map(pp.Suppress, "[]{}:")

        jsonString = pp.dblQuotedString().setParseAction(pp.removeQuotes)
        jsonNumber = ppc.number().setName("jsonNumber")

        jsonObject = pp.Forward().setName("jsonObject")
        jsonValue = pp.Forward().setName("jsonValue")

        jsonElements = pp.delimitedList(jsonValue).setName(None)

        jsonArray = pp.Group(
            LBRACK + pp.Optional(jsonElements) + RBRACK, aslist=RETURN_PYTHON_COLLECTIONS
        ).setName("jsonArray")

        jsonValue << (jsonString | jsonNumber | jsonObject | jsonArray | TRUE | FALSE | NULL)

        memberDef = pp.Group(
            jsonString + COLON + jsonValue, aslist=RETURN_PYTHON_COLLECTIONS
        ).setName("jsonMember")

        jsonMembers = pp.delimitedList(memberDef).setName(None)
        jsonObject << pp.Dict(
            LBRACE + pp.Optional(jsonMembers) + RBRACE, asdict=RETURN_PYTHON_COLLECTIONS
        )

        jsonComment = pp.cppStyleComment
        jsonObject.ignore(jsonComment)
        jsonDoc = jsonObject | jsonArray
        extract_and_parse_jsons.jsonDoc = jsonDoc
    for _, l, r in extract_and_parse_jsons.jsonDoc.scanString(text):
        json_string = text[l:r]
        try:
            parsed_data = json.loads(json_string)

            if isinstance(parsed_data, list):
                for item in parsed_data:
                    yield item
            else:
                yield parsed_data
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
      
from abc import ABC, abstractmethod

class CallExtractor(ABC):
    @abstractmethod
    def extract(self, text: str):
        pass
    
    @staticmethod
    def get_extractor(extractor_type: str):
        if extractor_type == "json":
            return JsonCallExtractor()
        elif extractor_type == "code":
            return CodeCallExtractor()
        else:
            raise ValueError(f"Unsupported extractor type {extractor_type}")
    
class JsonCallExtractor(CallExtractor):
    def extract(self, text: str):
        return extract_and_parse_jsons(text)
    
class CodeCallExtractor(CallExtractor):
    def extract(self, text: str):
        return extract_calls(text)
            
            
if __name__ == "__main__":
    text = """
    <tool_call>
    result0 = get_contact_info(name="Benjamin", key="email")
    result1 = web_search(query="Benjamin latest paper on economics", engine="google")
    result2 = add(a=1, b=2.45, c=result0, query="111,111")
    </tool_call><|im_end|>
    """
    extractor = CallExtractor.get_extractor("code")
    
    for call in extractor.extract(text):
        print(call)
