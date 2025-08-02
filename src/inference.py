

from src.config import DATASET_PATH, MODEL_NAME, EPOCHS, BATCH_SIZE, MAX_LEN, CHECKPOINT_DIR, CONFIG_FILENAME, DEVICE
from src.model_class import BertWithLinearClassifier
from nltk.tokenize import sent_tokenize
import torch.nn as nn
import torch
import os
import json 
import torch.nn.functional as F
from src.train.data_utils import get_tokenizer

classifier_weights = os.path.join(CHECKPOINT_DIR, "best_linear_head.pt")
with open(CONFIG_FILENAME, 'r') as f:
    model_config = json.load(f)

NUM_LABELS = model_config['num_labels']
model = BertWithLinearClassifier(model_name = MODEL_NAME, num_labels=NUM_LABELS)
model.classifier.load_state_dict(torch.load(classifier_weights))
model.to(DEVICE)
model.eval()

tokenizer = get_tokenizer(MODEL_NAME)

def preprocess(text):
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)

def predict(sentences):
    inputs = preprocess(sentences)  # assumes batch of strings
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]  # shape: [batch_size, num_labels]
        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)  # shape: [batch_size]

    return pred_labels.cpu(), probs.cpu()

example = "In today's world of technology, not everything made can be trusted. Therefore verify everything with your own senses."
sentences  = sent_tokenize(example)
print('sentences  -> ', sentences )

id2label = model_config['id2label'] 
id2label = {int(i):j for i,j in id2label.items()}

all_preds, all_probs = predict(sentences)

for i, sent in enumerate(sentences):
    label_id = all_preds[i].item()
    confidence = all_probs[i][label_id].item()
    label = id2label[label_id]
    print(f"Sentence: {sent}")
    print(f"Predicted Label: {label} ({confidence:.4f} confidence)")
    print("=" * 40)