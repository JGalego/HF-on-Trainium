# pylint: disable=import-error

"""
Fine-tune a pretrained model on AWS Trainium with distributed training

Adapted from
https://huggingface.co/docs/transformers/training
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/programming-guide/training/pytorch-neuron-programming-guide.html#pytorch-neuron-multi-worker-data-parallel-training-using-torchrun
"""

import os
import evaluate
import torch

from datasets import load_dataset

from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend

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
device = "xla"

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

# Initialize distributed training
torch.manual_seed(0)
torch.distributed.init_process_group(device)
world_size = xm.xrt_world_size()

# Create a (distributed) data loader for the train dataset
train_sampler = None  #pylint: disable=invalid-name
if world_size > 1:
    train_sampler = DistributedSampler(
        small_train_dataset,
        num_replicas=world_size,
        rank=xm.get_ordinal(),
        shuffle=True
    )

train_dataloader = DataLoader(
    small_train_dataset,
    shuffle=not train_sampler,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
)

# Load the training data onto each device
train_device_loader = pl.MpDeviceLoader(train_dataloader, device)

# Create a data loader for the test dataset
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
    for batch in train_device_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        # Consolidates the gradients between cores and issues the XLA device step computation
        xm.optimizer_step(optimizer)

        optimizer.zero_grad()
        progress_bar.update(1)

# Save model checkpoint
os.makedirs("checkpoints", exist_ok=True)
ckpt = {"state_dict": model.state_dict()}
xm.save(ckpt, "checkpoints/checkpoint.pt")

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
