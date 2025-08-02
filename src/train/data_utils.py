import os
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from collections import Counter

def load_dataset(dataset_path):
    return load_from_disk(dataset_path)

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

def encode_dataset(ds, tokenizer, label2id, max_len):
    def tokenize_and_encode(batch):
        encodings = tokenizer(
            batch["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
        encodings["label"] = [label2id[label] for label in batch["label"]]
        return encodings

    ds = ds.map(tokenize_and_encode, batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

def compute_class_weights(dataset, label2id, label_pad_token_id=-100):
    label_counts = Counter()
    for item in dataset:
        labels = item['label']
        label_counts.update([l for l in labels if l != label_pad_token_id])

    total = sum(label_counts.values())
    num_labels = len(label2id)

    weights = []
    for label_id in range(num_labels):
        count = label_counts.get(label_id, 1)
        weight = total / count
        weights.append(weight)

    return torch.tensor(weights, dtype=torch.float)
