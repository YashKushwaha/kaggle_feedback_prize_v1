import os
import numpy as np
import torch
from transformers import TrainerCallback

from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

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


def get_compute_metrics_fn(id2label):
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)

        def decode(preds, labels):
            pred_labels = [
                [id2label[p] for (p, l) in zip(pred_seq, label_seq) if l != -100]
                for pred_seq, label_seq in zip(predictions, labels)
            ]
            true_labels = [
                [id2label[l] for (p, l) in zip(pred_seq, label_seq) if l != -100]
                for pred_seq, label_seq in zip(predictions, labels)
            ]
            return pred_labels, true_labels

        pred_labels, true_labels = decode(predictions, labels)

        return {
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1": f1_score(true_labels, pred_labels),
            "accuracy": accuracy_score(true_labels, pred_labels),
        }

    return compute_metrics