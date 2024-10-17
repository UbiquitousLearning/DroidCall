import json
import re

import pyparsing as pp
from pyparsing import pyparsing_common as ppc

def extract_calls(calls_str):
    # 使用正则表达式来匹配字符串中的多个调用部分
    # 每个调用的格式为 result{id} = {function_name}({arguments})
    pattern = r'result(\d+) = (\w+)\((.*?)\)'
    matches = re.finditer(pattern, calls_str)
    
    # 初始化一个列表来存储所有解析出的调用
    extracted_calls = []
    
    # 遍历匹配结果
    for match in matches:
        call_id, function_name, arguments_str = match.groups()
        
        # 处理参数字符串，转换成字典
        args_pattern = r'(\w+)="([^"]*)"'
        arguments = dict(re.findall(args_pattern, arguments_str))
        
        # 构建并添加结果字典
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
