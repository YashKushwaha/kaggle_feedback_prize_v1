import os
import json
import warnings
import datetime
from pathlib import Path
import torch

from transformers import Trainer, TrainingArguments

from src.train.config import *
from src.train.model_class import BertWithLinearClassifier
from src.train.data_utils import load_dataset, get_tokenizer, encode_dataset
from src.train.trainer_utils import SaveBestLinearHeadCallback, compute_metrics

warnings.simplefilter(action='ignore', category=FutureWarning)

print("📦 Loading dataset...")
ds = load_dataset(os.path.join(DATASET_PATH, 'sentence_level_feedback_dataset'))

print("🔠 Tokenizing and encoding labels...")
tokenizer = get_tokenizer(MODEL_NAME)
label_list = ds["train"].unique("label")
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

ds = encode_dataset(ds, tokenizer, label2id, MAX_LEN)

print("🧠 Building model...")
model = BertWithLinearClassifier(MODEL_NAME, num_labels=len(label_list))

TEST_MODE = False

if TEST_MODE:
    train_dataset = ds["train"].select(range(1000))
    eval_dataset = ds["validation"].select(range(200))
    test_subset = ds["test"].select(range(200))
else:
    train_dataset = ds["train"]#.select(range(1000))
    eval_dataset = ds["validation"]#.select(range(200))
    test_subset = ds["test"]#.select(range(200))

now =datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

training_args = TrainingArguments(
    output_dir=os.path.join(CHECKPOINT_DIR, "results"),
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir=os.path.join(CHECKPOINT_DIR, "logs", now),
    logging_strategy="epoch",  # 🆕 log at each eval
    logging_steps=1,           # 🆕 if using 'steps' strategy
    report_to="tensorboard",   # 🆕 enable tensorboard
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    load_best_model_at_end=True,
    dataloader_num_workers=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[SaveBestLinearHeadCallback(save_dir=CHECKPOINT_DIR)]
)

print("🚀 Training...")
if Path(training_args.output_dir).exists() and any(Path(training_args.output_dir).glob("checkpoint-*")):
    print("🔁 Resuming from last checkpoint...")
    trainer.train(resume_from_checkpoint=True)
else:
    print("🚀 Starting fresh training...")
    trainer.train()


print("📊 Final Evaluation on Test Set:")
results = trainer.evaluate(test_subset)
print(results)

print("💾 Saving config...")
config_data = {
    "num_labels": len(label_list),
    "label2id": label2id,
    "id2label": id2label,
    "max_len": MAX_LEN
}
with open(os.path.join(CHECKPOINT_DIR, CONFIG_FILENAME), "w") as f:
    json.dump(config_data, f)
