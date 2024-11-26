COMPLEX_INSTRUCTION_SEED_PROMPT = """
I need your help to generate some function calling datasets. I will provide you with a tool description, and you need to generate queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, you may omit its value.
7. The generated data must be presented in the format given in my example.
8. THE PARAMETER VALUES GENERATED WITH FUNCTION CALL GENERATED MUST BE VALUES THAT CAN BE INFERRED FROM THE USER'S QUERY; YOU CANNOT FABRICATE PARAMETERS THAT CANNOT BE OBTAINED FROM THE USER'S REQUEST.
9. THE GENERATED QUERY SHOULD CONTAIN ENOUGH INFOMATION SO THAT YOU COULD CORRECTLY GENERATE PARAMETER USED BY THE TOOLS. THIS IS
 ALSO TO GUARANTEE THAT YOU DON'T FABRICATE PARAMETERS.
10. You should use all the tools I provided to generate the query and answer. It means that you should generate a query that needs to use all the tools I provided to solve, and remember to provider an answer that uses all the tools to solve the query.
11. You can use the same tool multiple times in a single query to ensure the query diversity.
12. Attach each answer with an id starting from 0. And if a tool should use the respone from another tool, you can reference it using #id, where id is the id of the tool.
13. Generate data of nested function calls if possible. i.e., the argument of a function call is the response of another function call.

following are some examples:
$examples
tools: [
  {
    "name": "send_email",
    "description": "Send an email to a certain mailbox.",
    "arguments": {
       "to": {
         "description": "email address to send to",
         "type": "str",
         "required": true
       }
    }
  },
  {
    "name": "get_email_by_name",
    "description": "get email by the name searching contacts and return the email",
    "arguments": {
      "name": {
        "description": "name to find the email",
        "type": "str",
        "required": true
      }
    },
    "returns": {
      "decription": "Return the email address of the contact person corresponding to the name."
      "type": "str"
    }
  }
]
response: 
{
  "query": "Help me send an email to Tom",
  "answers": [
    {
      "id": 0,
      "name": "get_email_by_name",
      "arguments": {
        "name": "Tom",
      }
    }, 
    {
      "id": 1,
      "name": "send_email",
      "arguments": {
        "to": "#1"
      }
    }
  ]  
}

tools: [
  {
    "name": "dial",
    "description": "Opens the dialer with a specified number in a phone app for user.\n\nThis function helps user to start a phone call process. It can open\nthe dialer with a pre-filled number. User can then choose to dial the number.",
    "arguments": {
      "phone_number": {
        "description": "The phone number to dial. This should be a valid\ntelephone number as defined in IETF RFC 3966. Examples include:\n\"2125551212\" or \"(212) 555 1212\".",
        "type": "str",
        "required": true
      }
    },
    "examples": [
      "# Open dialer with a number\ndial(\"2125551212\")"
    ]
  },
  {
    "name": "get_contact_info",
    "description": "Get the contact information based on the contact name and the key.\n",
    "arguments": {
      "name": {
        "description": "The name of the contact.",
        "type": "str",
        "required": true
      },
      "key": {
        "description": "The key to get the information of the contact.\ncan be one of the following: \"email\", \"phone\", \"address\" \"uri\"\nif key is \"uri\", this function will return the uri of the contact that can be \nused to edit the contact.",
        "type": "str",
        "required": true
      }
    },
    "returns": {
      "description": "The information of the contact based on the key.",
      "type": "str"
    },
    "examples": [
      "get_contact_info(\"John Doe\", \"email\")\nthis will return the email of the contact named \"John Doe\""
    ]
  }
]
responese:
{
  "query": "I want to call my friend Victor",
  "answers": [
    {
      "id": 0,
      "name": "get_contact_info",
      "arguments": {
        "name": "Victor",
        "key": "phone"
      }
    },
    {
      "id": 1,
      "name": "dial",
      "arguments": {
        "phone_number": "#0"
      }
    }
  ]
}

Now I will give you a tool, and you help me generate 15 query-answer pairs.
REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE AND PUT IT IN A JSON LIST.
REMEMBER YOU SHOULD USE ALL THE TOOLS AT ONE QUERY AND SOLVE IT WITH ALL TOOLS, AND GENERATE NESTED CALL IF POSSIBLE.
REMEMBER NOT TO FABRICATE PARAMETERS FOR TOOLS. PARAMETERS SHOULD BE INFERED FROM USER QUERY.
tools: 
$tools
"""

COMPLEX_INSTRUCTION_GEN_PROMPT = """
I need your help to generate some function calling datasets. I will provide you with a tool description, and you need to generate queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, you may omit its value.
7. The generated data must be presented in the format given in my example.
8. THE PARAMETER VALUES GENERATED WITH FUNCTION CALL GENERATED MUST BE VALUES THAT CAN BE INFERRED FROM THE USER'S QUERY; YOU CANNOT FABRICATE PARAMETERS THAT CANNOT BE OBTAINED FROM THE USER'S REQUEST.
9. THE GENERATED QUERY SHOULD CONTAIN ENOUGH INFOMATION SO THAT YOU COULD CORRECTLY GENERATE PARAMETER USED BY THE TOOLS. THIS IS
 ALSO TO GUARANTEE THAT YOU DON'T FABRICATE PARAMETERS.
10. You should use all the tools I provided to generate the query and answer. It means that you should generate a query that needs to use all the tools I provided to solve, and remember to provider an answer that uses all the tools to solve the query.
11. You can use the same tool multiple times in a single query to ensure the query diversity.
12. Attach each answer with an id starting from 0. And if a tool should use the respone from another tool, you can reference it using #id, where id is the id of the tool.
13. Generate data of nested function calls if possible. i.e., the argument of a function call is the response of another function call.


Now I will give you some tools and some example data of query-answer pairs using these tools. 
Please help me generate 40 query-answer pairs.
tools: $tools
examples: $examples

REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE AND PUT IT IN A JSON LIST.
REMEMBER YOU SHOULD USE ALL THE TOOLS AT ONE QUERY AND SOLVE IT WITH ALL TOOLS, AND GENERATE NESTED CALL IF POSSIBLE.
REMEMBER NOT TO FABRICATE PARAMETERS FOR TOOLS. PARAMETERS SHOULD BE INFERED FROM USER QUERY.
"""

SEED_GENERATION_PROMPT = """
I need your help to generate some function calling datasets. I will provide you with a tool description, and you need to generate queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, you may omit its value.
7. The generated data must be presented in the format given in my example.
8. The parameter values generated with function call generated must be values that can be inferred from the user's query; YOU CANNOT FABRICATE PARAMETERS THAT CANNOT BE OBTAINED FROM THE USER'S REQUEST.
9. Attach each answer with an id starting from 0. And if a tool should use the respone from another tool, you can reference it using #id, where id is the id of the tool.

following are some examples:
$examples

Now I will give you a tool, and you help me generate 15 query-answer pairs.
REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE
REMEMBER NOT TO FABRICATE PARAMETERS FOR TOOLS. PARAMETERS SHOULD BE INFERED FROM USER QUERY.
tool: $tool
"""

DATA_GENERATION_PROMPT = """
I need your help to generate some function calling datasets. I will provide you with a tool description and some example data for you. 
You need to generate queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, it is not necessary to provide its value.
7. The query-answer pairs should cover as many possible uses of the tool as possible.
8. The generated data must be presented in the format given in my example.
9. The parameter values generated with function call generated must be values that can be inferred from the user's query; YOU CANNOT FABRICATE PARAMETERS THAT CANNOT BE OBTAINED FROM THE USER'S REQUEST.

following are tool I provided and some examples of query-answer pairs:
tool: $tool
examples: $examples

Now please help me generate 40 query-answer pairs.
REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE
REMEMBER NOT TO FABRICATE PARAMETERS FOR TOOLS. PARAMETERS SHOULD BE INFERED FROM USER QUERY.
"""

SYSTEM_PROMPT_FOR_FUNCTION_CALLING = """
You are an expert in composing functions. You are given a query and a set of possible functions. 
Based on the query, you will need to make one or more function calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. Remember you should not use functions that is not suitable for the query and only return the function call in tools call sections. 
"""

SHORT_SYSTEM_PROMPT_FOR_FUNCTION_CALLING = """
You are an expert in composing functions.
"""

JSON_NESTED_CALLING_PROMT = """
If an argument is a response from a previous function call, you can reference it in the following way like the argument value of arg2 in func1:
[
    {
      "id": 0,
      "name": "func0",
      "arguments": {
          "arg1": "value1",
          "arg2": "value2",
          ...
      }
    },
    {
      "id": 1,
      "name": "func1",
      "arguments": {
          "arg1": "value1",
          "arg2": "#0",
          ...
      }
    },
    ...
]
This means that the value of arg2 in func1 is the return value from func0 (#0 means the response from the function call with id 0).
"""

CODE_NESTED_CALLING_PROMPT = """
You can do nested function calling in the following way:
result1 = func0(arg1="value1", arg2="value2", ...)
result2 = func1(arg1="value1", arg2=result1, ...)
...
This means that the value of arg2 in func1 is the return value from func0.
"""

JSON_CALL_FORMAT = """
[
    {
      "id": 0,
      "name": "func0",
      "arguments": {
          "arg1": "value1",
          "arg2": "value2",
          ...
      }
    },
    {
      "id": 1,
      "name": "func1",
      "arguments": {
          "arg1": "value1",
          "arg2": "value2",
          ...
      }
    },
    ...
]
"""

CODE_CALL_FORMAT = """
result1 = func0(arg1="value1", arg2="value2", ...)
result2 = func1(arg1="value1", arg2=result1, ...)
...
"""

FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL = """
Here is a list of functions that you can invoke:
$functions

Should you decide to return the function call(s), Put it in the format of 
$call_format

$nest_prompt

$example

If there is a way to achieve the purpose using the given functions, please provide the function call(s) in the above format.
REMEMBER TO ONLY RETURN THE FUNCTION CALLS LIKE THE EXAMPLE ABOVE, NO OTHER INFORMATION SHOULD BE RETURNED.

Now my query is: $user_query
"""

SHORT_FUNCTION_CALLING_PROMPT = """
Here is a list of functions:
$functions

$call_format

$nest_prompt

$example

Now my query is: $user_query
"""

ANNOTATION_PROMPT = """
Now I have some functions described in JSON format, and I have some query and solution that using these functions.
I use GPT to generate solution for the query, and I need to validate the solution with the given solution.
Some arguments in the functinos should be strictly matched with the given solution, while others should be semantically matched.
I need to annotate the field type of each argument in the functions.
Blow is an example function
{
	"name": "ACTION_INSERT_EVENT",
	"description": "Add a new event to the user's calendar.\n",
	"arguments": {
		"TITLE": {
			"description": "The event title.",
			"type": "str",
			"required": true
		},
		"DESCRIPTION": {
			"description": "The event description.",
			"type": "str",
			"required": true
		},
		"EVENT_LOCATION": {
			"description": "The event location.",
			"type": "str",
			"required": true
		},
		"EXTRA_EVENT_ALL_DAY": {
			"description": "A boolean specifying whether this is an all-day event. Default is False.",
			"type": "bool",
			"required": false,
			"default": false
		},
		"EXTRA_EVENT_BEGIN_TIME": {
			"description": "The start time of the event in ISO 8601 format. Default is None.",
			"type": "str",
			"required": false,
			"default": null
		},
		"EXTRA_EVENT_END_TIME": {
			"description": "The end time of the event in ISO 8601 format. Default is None.",
			"type": "str",
			"required": false,
			"default": null
		},
		"EXTRA_EMAIL": {
			"description": "A list of email addresses that specify the invitees. Default is None.",
			"type": "List[str]",
			"required": false,
			"default": null
		}
	}
}

{
    "TITLE": {
        "reason": "It is no need to generate a title strictly matched with the given solution, so it should be semantically matched.",
        "match_type": "semantic",
    },
    "DESCRIPTION": {
        "reason": "The description is subjective and is uncertain. So it should be semantically matched.",
        "match_type": "semantic",
    },
    "EVENT_LOCATION": {
        "reason": "Location is a common field, so it should be strictly matched.",
        "match_type": "strict",
    },
    "EXTRA_EVENT_ALL_DAY": {
        "reason": "The field is specified by query, so it should be strictly matched.",
        "match_type": "strict",
    },
    "EXTRA_EVENT_BEGIN_TIME": {
        "reason": "The field is specified by query, so it should be strictly matched.",
        "match_type": "strict",
    },
    "EXTRA_EVENT_END_TIME": {
        "reason": "The field is specified by query, so it should be strictly matched.",
        "match_type": "strict",
    },
    "EXTRA_EMAIL": {
        "reason": "The field is specified by query, so it should be strictly matched.",
        "match_type": "strict",
    }
}


You should only annotate the arguments, not other fields. If arguments is an empty dict {} you can just annotate
it with an empty dict {}.

Here is the function I give you:
$function

Please annotate the argument field type of the function JUST LIKE THE EXAMPLE ABOVE.
"""
