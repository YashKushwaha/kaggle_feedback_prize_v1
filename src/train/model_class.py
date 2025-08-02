import torch.nn as nn
from transformers import BertModel

class BertWithLinearClassifier(nn.Module):
    def __init__(self, model_name, num_labels, loss_fn=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False  # Freeze BERT

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.bert.config.hidden_size, num_labels)
        )
        self.loss_fn = loss_fn

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)

        loss = None
        if labels is not None and self.loss_fn is not None:
            loss = self.loss_fn(logits, labels)

        return {"logits": logits, "loss": loss} if loss is not None else {"logits": logits}
