"""
Fine-tune a pretrained model

Adapted from https://huggingface.co/docs/transformers/training
"""

import evaluate
import torch

from datasets import load_dataset

from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer
)

# Hyperparameters
BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 5e-5
DATASET_SIZE = 1000
NUM_LABELS = 5

# Specify the device to use
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load 'Yelp Reviews' data
# https://huggingface.co/datasets/yelp_review_full
dataset = load_dataset("yelp_review_full")

# Tokenize dataset
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenized_datasets = dataset.map(
	lambda examples : tokenizer(examples["text"], padding="max_length", truncation=True), batched=True)

# Remove the text column as the model does not accept raw text as an input
tokenized_datasets = tokenized_datasets.remove_columns(["text"])

# Rename the label column to labels because the model expects the argument to be named labels
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# Set the format of the dataset to return PyTorch tensors instead of lists
tokenized_datasets.set_format("torch")

# Create a smaller subset of the full dataset to speed up the fine-tuning
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(DATASET_SIZE))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(DATASET_SIZE))

# Create data loaders for the train and test datasets
train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=BATCH_SIZE)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=BATCH_SIZE)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=NUM_LABELS)
model.to(device)

# Create an optimizer and learning rate scheduler to fine-tune the model
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Add a progress bar over the number of training steps
num_training_steps = NUM_EPOCHS * len(train_dataloader)
progress_bar = tqdm(range(num_training_steps))

# Training loop
model.train()
for epoch in range(NUM_EPOCHS):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

# Evaluate loop
metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

print(metric.compute())
