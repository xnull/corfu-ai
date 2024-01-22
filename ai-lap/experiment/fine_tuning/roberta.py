import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification

data_files = {
    "train": "ai-lap/examples/logs/dataset/train.json",
    "test": "ai-lap/examples/logs/dataset/test.json",
}
dataset = load_dataset("json", data_files=data_files)
print(dataset)

model_path = "siebert/sentiment-roberta-large-english"
device = 'cpu'

# Preprocessing
tokenizer = AutoTokenizer.from_pretrained(model_path, device_map=device)

# This function tokenizes the input text using the RoBERTa tokenizer. 
# It applies padding and truncation to ensure that all sequences have the same length (256 tokens).
def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize, batched=True)

small_train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2, device_map=device)

print("INIT TRAINER")
training_args = TrainingArguments(
    output_dir="test_trainer", 
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    #fp16=True
    use_cpu=True
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    #compute_metrics=compute_metrics
)

print("TRAIN")
trainer.train()