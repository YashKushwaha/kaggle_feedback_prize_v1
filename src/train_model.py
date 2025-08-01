# train_frozen_bert_classifier.py

import torch
from torch import nn
from transformers import BertModel, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_from_disk, DatasetDict
import numpy as np
import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.config import DATASET_PATH, MODEL_NAME, EPOCHS, BATCH_SIZE, MAX_LEN, CHECKPOINT_DIR
from src.model_class import BertWithLinearClassifier
# --- 2. Load Dataset ---
print("📦 Loading dataset...")


import transformers
print(transformers.__version__)

dataset_path = os.path.join(DATASET_PATH, 'sentence_level_feedback_dataset')
ds: DatasetDict = load_from_disk(dataset_path)
#ds = ds['train']
from transformers import TrainerCallback

class SaveBestLinearHeadCallback(TrainerCallback):
    def __init__(self, save_dir="./checkpoints"):
        self.best_loss = float("inf")
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def on_evaluate(self, args, state, control, **kwargs):
        eval_loss = kwargs["metrics"]["eval_loss"]
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            save_path = os.path.join(self.save_dir, "best_linear_head.pt")
            torch.save(kwargs["model"].classifier.state_dict(), save_path)
            print(f"📦 Saved better linear head with loss {eval_loss:.4f}")


# --- 3. Tokenizer & Label Encoding ---
print("🔠 Tokenizing and encoding labels...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
label_list = ds["train"].unique("label")
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

print('label2id -> ', label2id)

def tokenize_and_encode(batch):
    encodings = tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN
    )
    encodings["label"] = [label2id[label] for label in batch["label"]]
    return encodings

ds = ds.map(tokenize_and_encode, batched=True)
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

# --- 5. Training Setup ---
model = BertWithLinearClassifier(MODEL_NAME, num_labels=len(label_list))

output_dir = os.path.join(CHECKPOINT_DIR, 'results')
logging_dir = os.path.join(CHECKPOINT_DIR, 'logging')
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    #save_strategy="epoch",
    save_strategy="no",  # Don't save model checkpoints automatically
    logging_dir=logging_dir,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    load_best_model_at_end=False,
    report_to="none"
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


# Use smaller subsets for testing
train_dataset = ds["train"].select(range(1000))        # or fewer
eval_dataset   = ds["validation"].select(range(200))
test_subset  = ds["test"].select(range(200))

# --- 6. Trainer ---
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[SaveBestLinearHeadCallback(save_dir = CHECKPOINT_DIR)]
)

# --- 7. Train ---
print("🚀 Training...")
trainer.train()

# --- 8. Evaluate ---
print("📊 Final Evaluation on Test Set:")
results = trainer.evaluate(test_subset)
print(results)
