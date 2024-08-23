from string import Template
from utils import get_json_obj
import json
from openai import OpenAI
from tqdm import tqdm

PROMPT = Template("""
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
""")

API_FILE = "data/api.jsonl"
OUTPUT_FILE = "data/annotated_api.jsonl"

from utils import JsonlSampler, JsonExtractor, OpenAiGenerateResponse, LLMDataCollector
from typing import List, Dict

class JsonFormatSampler(JsonlSampler):
    def format(self, samples: List[Dict[str, str]])->Dict[str, str]:
        self.last_sample = samples[0]
        return {"function": json.dumps(samples[0], indent=2, ensure_ascii=False)}

if __name__ == "__main__":
    sampler = JsonFormatSampler(API_FILE)
    extractor = JsonExtractor()
    
    client = OpenAI()
    generate_response = OpenAiGenerateResponse(client=client, model="gpt-4-turbo", system_prompt="")
    
    collector = LLMDataCollector(PROMPT, sampler, [extractor], generate_response)
    
    with open(OUTPUT_FILE, "w") as out_f:
        for annotation in collector.collect(1000):
            api = sampler.last_sample
            print(f"api: {json.dumps(api, indent=2, ensure_ascii=False)}\n")
            print(f"annotation: {json.dumps(annotation, indent=2, ensure_ascii=False)}")
            for k in api["arguments"]:
                if k in annotation:
                    api["arguments"][k].update(annotation[k])
            out_f.write(json.dumps(api, ensure_ascii=False)+"\n")
    
    
    
    