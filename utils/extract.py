import json
import re

import pyparsing as pp
from pyparsing import pyparsing_common as ppc

def convert_value(value):
    # 尝试将值转换为浮点数，如果实际上是整数，则转换为整数
    # print(f"value: {value}")
    try:
        float_value = float(value)
        if float_value.is_integer():
            return int(float_value)
        return float_value
    except ValueError:
        match = re.match(r'result(\d+)', value)
        if match:
            return f"#{match.group(1)}"
        return value

def extract_calls(calls_str):
    # 使用正则表达式来匹配字符串中的多个调用部分
    # 每个调用的格式为: result{id} = {function_name}({arguments})
    pattern = r'result(\d+) = (\w+)\((.*?)\)'
    matches = re.finditer(pattern, calls_str)
    
    # 遍历匹配结果
    for match in matches:
        call_id, function_name, arguments_str = match.groups()
        
        # 处理参数字符串，转换成字典
        # 更新正则表达式，以匹配带引号的字符串或不带引号的数字/文本，包括浮点数
        args_pattern = r'(\w+)=(".*?"|[\w.]+)'
        arguments = {
            arg_name: convert_value(arg_val.strip('"')) for arg_name, arg_val in re.findall(args_pattern, arguments_str)
        }
        
        # 构建并返回结果字典
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
            # 尝试解析找到的JSON字符串
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
            # 尝试解析找到的JSON字符串
            parsed_data = json.loads(json_string)

            # 如果解析结果是列表，则返回列表中的每个项
            if isinstance(parsed_data, list):
                for item in parsed_data:
                    yield item
            else:
                # 如果不是列表，则直接返回解析结果
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
    result2 = add(a=1, b=2.45, c=result0)
    </tool_call><|im_end|>
    """
    extractor = CallExtractor.get_extractor("code")
    
    for call in extractor.extract(text):
        print(call)
