import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os;
import json;

def file_read(dir_name):
    res = []

    for file_name in os.listdir(dir_name):
        with open(dir_name + "/" + file_name) as f:
            #Content_list is the list that contains the read lines.     
            for line in f:
                line_json = json.loads(line)
                res.append(line_json['message'])
   
    return res

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

messages = file_read("ai-lap/examples/logs/out/messages")
messages = messages[:500]
print(len(messages))

sentiment = {
    "positive": 0,
    "negative": 0,
}

counter = 0;
for query in messages:
    counter += 1
    if counter % 100 == 0:
        print("iteration:" + str(counter))

    inputs = tokenizer(query[:1000], return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    res = model.config.id2label[predicted_class_id]

    if res == 'POSITIVE':
        sentiment['positive'] += 1
    else:
        sentiment['negative'] += 1

print('\n')
print(sentiment)