import torch
import json
import os

from transformers import LlamaTokenizer, LlamaForCausalLM

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

sentiment = {
    "positive": 0,
    "negative": 0,
    "neutral": 0,
}

device = 'cpu'

from transformers import pipeline

sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english", device_map='auto')

print('start')
counter = 0
for query in messages:
    counter += 1
    if counter % 10 == 0:
        print("iteration:" + str(counter) + ". stats: " + str(sentiment))
        

    res = sentiment_analysis(query[:1024])
    res = res[0]['label']
    
    if 'POSITIVE' in res:
        sentiment['positive'] += 1
    
    if 'NEGATIVE' in res:
        sentiment['negative'] += 1
        #print("!!!my output! negative result for query: " + query[:1024])

    if 'NEUTRAL' in res:
        sentiment['neutral'] += 1

print('\n')
print(sentiment)