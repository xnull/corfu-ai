from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer

from datasets import load_dataset

import torch
import linecache
import json

# Prompt engineering!!!

print('semantic search')

model = SentenceTransformer('distilbert-base-uncased', device='mps')

query = [
    #'Find messages which changes the state, and sort by time'
    #'Find wrong epoch exceptions and failure detector errors'
    #'Find all fluctuations in the system'
    'Find SequenceServer changes'
]
query_embeddings = model.encode(query)

dataset = load_dataset('csv', data_files=['logs/embeddings/embeddings.csv'])
dataset_embeddings = torch.from_numpy(dataset["train"].to_pandas().to_numpy()).to(torch.float)
hits = semantic_search(query_embeddings, dataset_embeddings, top_k=50)

print("\n\n")

ids = []
for i in range(len(hits[0])):
    #print(texts[hits[0][i]['corpus_id']])
    ids.append(hits[0][i]['corpus_id'])
print(ids)

lines = []
for id in ids:
    source_data = linecache.getline('logs/out/messages/filebeat-20240116.ndjson', id)
    source_data_json = json.loads(source_data)
    lines.append(source_data_json)

print("lines: " + str(len(lines)))

for line in lines:
    print(line['message'] + "\n")
