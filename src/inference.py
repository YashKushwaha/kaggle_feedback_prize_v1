

from src.config import DATASET_PATH, MODEL_NAME, EPOCHS, BATCH_SIZE, MAX_LEN, CHECKPOINT_DIR, CONFIG_FILENAME
from src.model_class import BertWithLinearClassifier

from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch
import os
import json 
import torch.nn.functional as F

classifier_weights = os.path.join(CHECKPOINT_DIR, "best_linear_head.pt")
with open(CONFIG_FILENAME, 'r') as f:
    model_config = json.load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_LABELS = model_config['num_labels']  # or however many labels you have
model = BertWithLinearClassifier(model_name = MODEL_NAME, num_labels=NUM_LABELS)
model.classifier.load_state_dict(torch.load(classifier_weights))
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(text):
    return tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)

def predict(text):
    inputs = preprocess(text)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        probs = F.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
    
    return pred_label, probs.cpu().numpy()

example = "In today's world of technology, not everything made can be trusted."
label_id, confidence = predict(example)
print('label_id -> ', label_id)
print('confidence -> ', confidence)

# Optional: convert label ID to name
id2label = model_config['id2label'] 
id2label = {int(i):j for i,j in id2label.items()}
print('id2label -> ', id2label)

label = id2label[label_id]
_ = confidence[0][label_id]
print(f"Predicted label: {label} ({_:.4f} confidence)")
