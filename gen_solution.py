import chromadb
import json
from typing import List
from openai import OpenAI
from string import Template
from tqdm import tqdm
import os
from utils import get_json_obj
import argparse

parser = argparse.ArgumentParser(description='Generate solution for the task')
parser.add_argument('--input', type=str, default='./data/filtered_data.jsonl', help='Path to the input file')
parser.add_argument('--retrieve_doc_num', type=int, default=3, help='Number of documents to retrieve')
arg = parser.parse_args()

client = chromadb.PersistentClient(path="./chromaDB")

collection = client.get_or_create_collection('intents')



PROMPT_FOR_CHATMODEL = Template("""
You are an Android development expert, skilled in understanding user needs and you are very familiar with Android intents.
Now you will be given a user query and a list of relevant intents that may be related to the query. You need to 
Based on the user query, you should select proper intent that can achieve the user's goal. If none of the intents can be used,
point it out. You should only return the intent and the solution to the user query. You should not provide any additional information.
                                
If you can use one of the intent to solve the user query, please provide the intent and the solution in the following format:
{
    "intent": ... ,
    "uri": ... ,
    "mime": ... ,
    "extras": {
        ...
    },
}

Below is an example:

User query: I need to set an alarm for 10:30 PM every Sunday to remember to prepare for next week's work.
{
    "intent": "ACTION_SET_ALARM",
    "uri": "",
    "mime": "",
    "extras": {
        "EXTRA_HOUR": 22,
        "EXTRA_MINUTES": 30,
        "EXTRA_MESSAGE": "Prepare for Work",
        "EXTRA_DAYS": [
            1
        ],
        "EXTRA_RINGTONE": "",
        "EXTRA_VIBRATE": true
    },
}
If some field is not applicable, you can leave it empty. Remember to only provide the intent and the solution in the specific format. Do not provide any additional information.
                                
Now I will give you a user query and a list of intents. You should provide the intent and the solution to the user query.

User query: $user_query
Here are the intents you can use to solve the user query:
$intents_info
""")


class Handler:
    model_name: str

    def __init__(self, model_name, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def inference(self, user_query: str, documents: List[str]) -> str:
        # This method is used to retrive model response for each model.
        pass
    

class OpenAIHandler(Handler):
    def __init__(self, model_name, temperature=0.7, top_p=1, max_tokens=1000) -> None:
        super().__init__(model_name, temperature, top_p, max_tokens)
        self.client = OpenAI()

    def inference(self, user_query: str, documents: List[str]) -> str:
        # This method is used to retrive model response for each model.
        message =[
            {
                "role": "user",
                "content": PROMPT_FOR_CHATMODEL.substitute(user_query=user_query, intents_info="\n".join(documents))
            },
        ]
        # print(message)
        response = self.client.chat.completions.create(
            messages=message,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        return response.choices[0].message.content


HANDLER_MAP = {
    "openai": OpenAIHandler
}

HANDLER = "openai"
MODEL_NAME = "gpt-4o-mini"


if __name__ == '__main__':
    handler = HANDLER_MAP[HANDLER](MODEL_NAME)
    
    queries = []
    with open(arg.input, "r") as f:
        for line in f:
            j = json.loads(line)
            user_query = j["query"]
            queries.append(user_query)
    
    # create output directory if not exists
    if not os.path.exists("./results"):
        os.makedirs("./results")
    output_file = open(f"./results/{HANDLER}_{MODEL_NAME}_result.jsonl", "w")
    for query in tqdm(queries):
        results = collection.query(
            query_texts=[query],
            n_results=arg.retrieve_doc_num,
        )
        docs = results['documents'][0]
        documents = [
            json.dumps(json.loads(doc), indent=2, ensure_ascii=False)
            for doc in docs
        ]
        
        response = handler.inference(query, documents)
        # print(f"User Query: {query}")
        # print(f"Intents: {documents}")
        # print(f"Response: {response}")
        res = get_json_obj(response)
        output_file.write(json.dumps({"query": query, "response": res}, ensure_ascii=False) + "\n")
        output_file.flush()
        
    output_file.close()
