import os

from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import pandas as pd

import json
import time;


#https://huggingface.co/thenlper/gte-large

#hf embeddings: https://huggingface.co/blog/getting-started-with-embeddings#2-host-embeddings-for-free-on-the-hugging-face-hub

print('Start embeddings')

def file_read(dir_name):
    res = []

    for file_name in os.listdir(dir_name):
        with open(dir_name + "/" + file_name) as f:
            #Content_list is the list that contains the read lines.     
            for line in f:
                line_json = json.loads(line)
                res.append(line_json['message'])
   
    return res

texts = file_read("logs/out/messages")


print('load sentence transformer')
#model = SentenceTransformer('all-MiniLM-L6-v2', device='mps')
#model = SentenceTransformer('thenlper/gte-large', device='mps')
model = SentenceTransformer('distilbert-base-uncased', device='mps')
#model = SentenceTransformer('BAAI/bge-large-en-v1.5', device='mps')
#model = SentenceTransformer('TinyLlama/TinyLlama-1.1B-Chat-v1.0', device='cpu')

#model = SentenceTransformer('openlm-research/open_llama_3b', device='mps')

os.makedirs("logs/embeddings/", exist_ok = False) 

print('start dataset embeddings')
start = time.time()
dataset_embeddings = model.encode(texts)

pd_embeddings = pd.DataFrame(dataset_embeddings)
pd_embeddings.to_csv("logs/embeddings/embeddings.csv", index=False)

print('finish, time: ', time.time() - start)