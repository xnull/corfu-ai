import os
import json
import random

def file_read(dir_name):
    res = []

    for file_name in os.listdir(dir_name):
        with open(dir_name + "/" + file_name) as f:
            #Content_list is the list that contains the read lines.     
            for line in f:
                line_json = json.loads(line)

                label = 1
                label_str = line_json['logLevel'].strip()
                if label_str == 'DEBUG':
                    label = 1
                elif label_str == 'INFO':
                    label = 1
                else:
                    label = 0

                data = {
                    "label": label,
                    "text": line_json['message'],
                }
                res.append(data)
   
    return res

messages = file_read("ai-lap/examples/logs/out/messages")

with open('ai-lap/examples/logs/dataset/train.json', 'a') as f:
    for message in messages:
        f.write(json.dumps(message) + '\n')

with open('ai-lap/examples/logs/dataset/test.json', 'a') as f:
    for i in range(5000):
        message = random.choice(messages)
        f.write(json.dumps(message) + '\n')

