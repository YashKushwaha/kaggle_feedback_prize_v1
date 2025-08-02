import os
import nltk
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(PROJECT_ROOT, 'local_only', 'dataset', 'feedback-prize-2021')
HF_HOME = os.path.join(PROJECT_ROOT, 'local_only', 'HF_HOME')
os.environ['HF_HOME'] = HF_HOME

NLTK_HOME = os.path.join(PROJECT_ROOT, 'local_only', 'NLTK_HOME')
nltk.data.path.append(NLTK_HOME)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'local_only', 'checkpoints')
CONFIG_FILENAME = os.path.join(CHECKPOINT_DIR, 'model_config.json')


TOKENIZER_DIR = os.path.join(PROJECT_ROOT, 'local_only', 'tokenizer')
CHECKPOINT_NAME = "classifier_weights.pth"

MAX_LEN = 128
#MODEL_NAME = "deberta-v3-large"
MODEL_NAME = "bert-base-uncased"

MODEL_DIR = os.path.join(PROJECT_ROOT, 'local_only', 'models')

EPOCHS = 5
BATCH_SIZE = 128
LR = 5e-4
OVERWRITE_EXISTING_FILE = False

#os.environ['TRANSFORMERS_CACHE'] = MODEL_DIR


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")