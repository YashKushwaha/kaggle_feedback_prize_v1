import os
import numpy as np
import torch
from transformers import TrainerCallback

class SaveBestLinearHeadCallback(TrainerCallback):
    def __init__(self, save_dir):
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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}