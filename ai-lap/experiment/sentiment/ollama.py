from langchain_community.llms import Ollama
import os
import json

os.environ["LANGCHAIN_TRACING_V2"] = "false"

ollama = Ollama(base_url='http://10.173.65.105:8080', model="tinyllama")
#ollama = Ollama(base_url='http://127.0.0.1:11434', model="tinyllama")

def file_read(dir_name):
    res = []

    for file_name in os.listdir(dir_name):
        with open(dir_name + "/" + file_name) as f:
            #Content_list is the list that contains the read lines.     
            for line in f:
                line_json = json.loads(line)
                res.append(line_json['message'])
   
    return res

messages = file_read("ai-lap/examples/logs/out/messages")
messages = messages[:500]
print(len(messages))

sentiment = {
    "positive": 0,
    "negative": 0,
    "neutral": 0,
}

counter = 0;
for query in messages:
    counter += 1
    if counter % 10 == 0:
        print("iteration:" + str(counter))

    res = ollama("Answer only POSITIVE or NEGATIVE or NEUTRAL without any other words: " + query[:2048])

    if 'POSITIVE' in res:
        sentiment['positive'] += 1
    
    if 'NEGATIVE' in res:
        sentiment['negative'] += 1
        print("negative result for: " + query[:2048])

    if 'NEUTRAL' in res:
        sentiment['neutral'] += 1

print('\n')
print(sentiment)
