import torch
import json
import os

from transformers import LlamaTokenizer, LlamaForCausalLM
from accelerate import Accelerator

def file_read(dir_name):
    res = []

    for file_name in os.listdir(dir_name):
        with open(dir_name + "/" + file_name) as f:
            #Content_list is the list that contains the read lines.     
            for line in f:
                line_json = json.loads(line)
                res.append(line_json['message'])
   
    return res

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = file_read("ai-lap/examples/logs/out/messages")
messages = messages[:500]

sentiment = {
    "positive": 0,
    "negative": 0,
    "neutral": 0,
}

device = 'cpu'

model_path = 'openlm-research/open_llama_3b'
tokenizer = LlamaTokenizer.from_pretrained(model_path, device_map='auto')
model = LlamaForCausalLM.from_pretrained(model_path, device_map='auto')


print('start')
counter = 0;
for query in messages:
    counter += 1
    if counter % 10 == 0:
        print("iteration:" + str(counter))

    prompt = "Answer only POSITIVE or NEGATIVE or NEUTRAL without any other additional explanation. Here is the text: " + query[:2048]
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generation_output = model.generate(input_ids=input_ids, max_new_tokens=128)
    res = tokenizer.decode(generation_output[0])

    if 'POSITIVE' in res:
        sentiment['positive'] += 1
    
    if 'NEGATIVE' in res:
        sentiment['negative'] += 1
        print("negative result for: " + query[:2048])

    if 'NEUTRAL' in res:
        sentiment['neutral'] += 1

print('\n')
print(sentiment)