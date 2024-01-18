from sentence_transformers.util import semantic_search
from sentence_transformers import SentenceTransformer

from datasets import load_dataset

from transformers import BertTokenizer

import torch
import linecache
import json

from transformers import LlamaTokenizer, LlamaForCausalLM

# Prompt engineering!!!

model = SentenceTransformer('distilbert-base-uncased', device='cpu')
dataset = load_dataset('csv', data_files=['logs/embeddings/embeddings.csv'])
dataset_embeddings = torch.from_numpy(dataset["train"].to_pandas().to_numpy()).to(torch.float)

model_path = 'openlm-research/open_llama_3b'
tokenizer = LlamaTokenizer.from_pretrained(model_path, low_cpu_mem_usage=True)
llama_model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
)

def retrieve(query):
    print('semantic search')
    
    query_embeddings = model.encode(query)
    hits = semantic_search(query_embeddings, dataset_embeddings, top_k=50)

    print("\n\n")

    ids = []
    for i in range(len(hits[0])):
        #print(texts[hits[0][i]['corpus_id']])
        ids.append(hits[0][i]['corpus_id'])
    print(ids)

    lines = []
    for id in ids:
        source_data = linecache.getline('logs/out/messages/filebeat.json', id)
        source_data_json = json.loads(source_data)
        lines.append(source_data_json)

    result= []
    for line in lines:
        result.append(line['message'])

    return result

def run_llm(input_data, question):
    result_question = question
    for event in input_data:
        result_question + "\n" + event
    input_ids = tokenizer(result_question, return_tensors="pt").input_ids
    generation_output = llama_model.generate(input_ids=input_ids, max_new_tokens=128)
    return tokenizer.decode(generation_output[0])

def run_ai(query, question):
    retrieval = retrieve(query)
    ### llm: check all the components and servers and find the most critical errors
    #knowledge_base = {
    #    servers: ['SequenserServer', 'LogUnitServer', 'ManagementServer', 'LayoutServer'],
    #    components: ['failure detector', 'strem log']
    #}
    llm_response = run_llm(retrieval, question)

    return {
        "retrieval": retrieval,
        "llm": llm_response
    }


#query = [
        #Find messages which changes the state, and sort by time'
        #'Find wrong epoch exceptions and failure detector errors'
        #'Find all fluctuations in the system'
        #'when SequenceServer bootstrapped'
        #'when Sequence server had a problem?'
#        'when SequenceServer bootstrapped and sort by time'
#    ]
#run_ai(query)