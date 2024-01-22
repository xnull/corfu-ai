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

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = file_read("ai-lap/examples/logs/out/messages")
messages = messages[:500]

sentiment = {
    "positive": 0,
    "negative": 0,
    "neutral": 0,
}

device = 'cpu'

from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", device_map="mps")

print('start')
counter = 0;
for query in messages:
    counter += 1
    if counter % 3 == 0:
        print("iteration:" + str(counter) + ". stats: " + str(sentiment))

    messages = [
        {
            "role": "system",
            "content": "You have to answer only: POSITIVE or NEGATIVE or NEUTRAL. Format of you answer must be: {label: POSITIVE}, or {label: NEGATIVE}, or {label: NEUTRAL}",
        },
        {
            "role": "user", 
            "content": "Evaluate the message if it positive or negative or neutral: " + query[:2048]
        },
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    outputs = pipe(prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    res = outputs[0]["generated_text"].split('<|assistant|>')[1]
    print(res)
    print(outputs[0]["generated_text"])

    if 'POSITIVE' in res:
        sentiment['positive'] += 1
    
    if 'NEGATIVE' in res:
        sentiment['negative'] += 1

    if 'NEUTRAL' in res:
        sentiment['neutral'] += 1

print('\n')
print(sentiment)