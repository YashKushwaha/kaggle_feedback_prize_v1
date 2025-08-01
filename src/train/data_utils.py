import os
from datasets import load_from_disk
from transformers import AutoTokenizer

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
