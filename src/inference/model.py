import torch
import torch.nn.functional as F
from src.model_class import BertWithLinearClassifier
from src.train.data_utils import get_tokenizer
from src.config import MODEL_NAME, MAX_LEN, CHECKPOINT_DIR, CONFIG_FILENAME
import os
import json
from nltk.tokenize import sent_tokenize

class SentenceClassifier:
    def __init__(self, config_path=CONFIG_FILENAME, checkpoint_dir=CHECKPOINT_DIR, model_name=MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        with open(config_path, 'r') as f:
            self.model_config = json.load(f)

        self.id2label = {int(i): j for i, j in self.model_config['id2label'].items()}
        self.tokenizer = get_tokenizer(model_name)

        # Initialize model
        self.model = BertWithLinearClassifier(
            model_name=model_name,
            num_labels=self.model_config['num_labels']
        )
        classifier_weights = os.path.join(checkpoint_dir, "best_linear_head.pt")
        self.model.classifier.load_state_dict(torch.load(classifier_weights, map_location=self.device))

        self.model.to(self.device)
        self.model.eval()

    def preprocess(self, texts):
        return self.tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=MAX_LEN)

    def predict(self, sentences):
        inputs = self.preprocess(sentences)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k in ["input_ids", "attention_mask"]}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs["logits"]
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        results = []
        for i, sent in enumerate(sentences):
            pred_id = preds[i].item()
            conf = probs[i][pred_id].item()
            results.append({
                "sentence": sent,
                "label": self.id2label[pred_id],
                "confidence": conf
            })
        return results
    
if __name__ == '__main__':
    classifier = SentenceClassifier()
    sentences = """Imagine being in a classroom full of loud students. You wouldnt get much done. With that being said dont you think that giving students the opportunity to take their classes at home? Although some students need "hands on learning it should still be an opportunity. It sounds like a fantastic idea to me because some students work better alone and around no one, and it could also help students stay caught up and work at their own pace, and it would show how advance some of the students really are. To begin with, some students work better by alone and around no one. Some times being in a classroom can be difficult for students. Most of the time its really loud and no one is really paying attention which can be hard for the students really trying to just get their work done and pass the class. Not saying that it has to be completely silent in a classroom for everyone to focus, but imagine being in a class where the students talk the entire period or continue to get in trouble, it would be hard to stay focused and get everything done right? When students get around other students they tend to do things to empress one another which can make it very hard for other students to get their stuff done. Meanwhile students could be at home doing their work. Just picture how much you could get done in a shorter time span at home by your self! A lot right? The only way you could get distracted at home while your doing your work would be if you distract yourself."""
    sentences = sent_tokenize(sentences)
    predictions  = classifier.predict(sentences)

    with open('output.json', 'w') as f:
        json.dump(predictions, f)
