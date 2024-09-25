
SEED_GENERATION_PROMPT = """
I need your help to generate some function calling datasets. I will provide you with a tool description, and you need to generate queries and corresponding answers based on this tool, i.e., the answers that call the tool to resolve the user's query. Here are my requirements:

1. For queries, try to use different vocabulary and syntax to ensure query diversity. Queries can be long or short, complex or concise. In short, try not to generate similar queries; I want to ensure query diversity.
2. The language of the queries should be as diverse as possible. This means a query can be a command, a question, or a request with detailed descriptions, etc.
3. The generated queries should cover all possible uses of the tool as much as possible, meaning the coverage of various parameters should be comprehensive, ensuring the tool can be used to complete various forms of work.
4. The generated queries should be solvable using the given tools.
5. For the queries you generate, you should provide answers using the tool, i.e., give the tool used and the values for each parameter.
6. When providing parameters, if a parameter has required=False, you may omit its value.
7. The query-answer pairs should cover as many possible uses of the tool as possible.
8. The generated data must be presented in the format given in my example.
9. The parameter values generated with function call generated must be values that can be inferred from the user's query; you cannot fabricate a value out of thin air.

following are some examples:
$examples

Now I will give you a tool, and you help me generate 40 query-answer pairs.
REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE
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
9. The parameter values generated with function call generated must be values that can be inferred from the user's query; you cannot fabricate a value out of thin air.

following are tool I provided and some examples of query-answer pairs:
tool: $tool
examples: $examples

Now please help me generate 40 query-answer pairs.
REMEMBER TO GENERATE THE RESULT IN JSON FORMAT LIKE THE EXAMPLE ABOVE
"""

SYSTEM_PROMPT_FOR_FUNCTION_CALLING = """
You are an expert in composing functions. You are given a question and a set of possible functions. 
Based on the question, you will need to make one or more function/tool calls to achieve the purpose. 
If none of the function can be used, point it out. If the given question lacks the parameters required by the function,
also point it out. You should only return the function call in tools call sections.
"""

NESTED_CALLING_PROMT = """
If an argument is a response from a previous function call, you can reference it in the following way like the argument value of arg2 in func3:
{
    "name": "func3",
    "arguments": {
        "arg1": "value1",
        "arg2": "@func2",
        ...
    }
}
This means that the value of arg2 in func3 is the response from func2.
"""

FUNCTION_CALLING_PROMPT_FOR_CHAT_MODEL = """
Here is a list of functions in JSON format that you can invoke:
$functions

Should you decide to return the function call(s), Put it in the format of 
[
    {
        "name": "func1",
        "arguments": {
            "arg1": "value1",
            "arg2": "value2",
            ...
        }
    },
    {
        "name": "func2",
        "arguments": {
            "arg1": "value1",
            "arg2": "value2",
            ...
        }
    },
    ...
]

$nest_prompt

$example

If there is a way to achieve the purpose using the given functions, please provide the function call(s) in the above format.
REMEMBER TO ONLY RETURN THE FUNCTION CALLS LIKE THE EXAMPLE ABOVE, NO OTHER INFORMATION SHOULD BE RETURNED.

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
