import json
from string import Template

query_template = Template("""
Below is the query from the user, you need to provide the intent and provide the solution to the user query.
user_query: $user_query
""")

response_template = Template("""
Solution: $solution

Below is the intents info and an explanation of the solution:

intent: $intents_info

explanation: $explanation
""")


RECODES_FILE = "machine_generated.jsonl"
INTENTS_FILE = "intents.jsonl"

OUTPUT_FILE = "finetune_dataset.jsonl"


if __name__ == "__main__":
    intents_info = {}
    with open(INTENTS_FILE, 'r') as f:
        for line in f:
            j = json.loads(line)
            intents_info[j['id']] = j
            
    output = open(OUTPUT_FILE, 'w')
    
    with open(RECODES_FILE, 'r') as f:
        for line in f:
            recode = json.loads(line)
            query = recode.pop('query')
            id = recode.pop('id')
            explanation = recode.pop('explanation', "")
            intent_info_text = json.dumps(intents_info[id], indent=2)
            solution_text = json.dumps(recode, indent=2)
            
            query_text = query_template.substitute(user_query=query)
            response_text = response_template.substitute(solution=solution_text, intents_info=intent_info_text, explanation=explanation)
            output.write(json.dumps(
                {"messages": [
                    {"role": "user", "content": query_text},
                    {"role": "assistant", "content": response_text}
                ]}
            )+'\n')
        output.flush()
    
    output.close()
