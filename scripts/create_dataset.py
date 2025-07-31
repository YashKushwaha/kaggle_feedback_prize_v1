import os
import pandas as pd
from nltk.tokenize import sent_tokenize
import nltk
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from nltk.data import find
from src.config import NLTK_HOME, DATASET_PATH

# Check if 'punkt' is already downloaded
try:
    x = find('tokenizers/punkt')
    print(x)
    print("Punkt tokenizer is already available.")
except LookupError:
    print("Punkt not found. Downloading now...")
    nltk.download('punkt', download_dir=NLTK_HOME)

try:
    x = find('tokenizers/punkt_tab')
    print(x)
    print("Punkt tokenizer is already available.")
except LookupError:
    print("Punkt not found. Downloading now...")
    nltk.download('punkt_tab', download_dir=NLTK_HOME)

 # directory with .txt essay files
essay_dir = os.path.join(DATASET_PATH, "train")
train_csv_path = os.path.join(DATASET_PATH, "train.csv")

# Load CSV
df = pd.read_csv(train_csv_path)
annotations_by_id = df.groupby("id")

# IoU function
def iou(span1, span2_str):
    set1 = set(range(*span1))
    set2 = set(map(int, span2_str.split()))
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0

# Process all essays
data = []
essay_ids = df["id"].unique()

print('df.shape -> ', df.shape)
print('len(essay_ids) -> ', len(essay_ids))

for essay_id in essay_ids:
    essay_path = os.path.join(essay_dir, f"{essay_id}.txt")
    if not os.path.exists(essay_path):
        continue

    with open(essay_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    sentences = sent_tokenize(full_text)
    words = full_text.split()

    # Map sentence to word indices
    sentence_word_ranges = []
    start = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        end = start + len(sentence_words)
        sentence_word_ranges.append((start, end))
        start = end

    annots = annotations_by_id.get_group(essay_id)

    for i, (start, end) in enumerate(sentence_word_ranges):
        label = "O"
        best_iou = 0
        for _, row in annots.iterrows():
            score = iou((start, end), row["predictionstring"])
            if score > 0.5 and score > best_iou:
                label = row["discourse_type"]
                best_iou = score

        data.append({
            "essay_id": essay_id,
            "sentence": sentences[i],
            "label": label
        })

# Create DataFrame
df_sentences = pd.DataFrame(data)

# Split into train/val/test (80/10/10 stratified)
train_val, test = train_test_split(
    df_sentences, test_size=0.10, stratify=df_sentences["label"], random_state=42
)
train, val = train_test_split(
    train_val, test_size=0.1111, stratify=train_val["label"], random_state=42
)

# Convert to Hugging Face Datasets
ds = DatasetDict({
    "train": Dataset.from_pandas(train.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val.reset_index(drop=True)),
    "test": Dataset.from_pandas(test.reset_index(drop=True)),
})

# Save to disk (optional)
save_name = "sentence_level_feedback_dataset"
save_name = os.path.join(DATASET_PATH, save_name)
ds.save_to_disk(save_name)

metadata_file = os.path.join(save_name, 'metadata.txt')
with open(metadata_file, 'w', encoding="utf-8") as f:
    f.write(str(ds))

# Number of examples to save from each split
N = 5

samples_file = os.path.join(save_name, "dataset_samples.txt")
# Save a few examples from each split to a text file
with open(samples_file, "w", encoding="utf-8") as f:
    for split in ["train", "validation", "test"]:
        f.write(f"\n=== {split.upper()} EXAMPLES ===\n")
        for example in ds[split].select(range(N)):
            f.write(f"Essay ID: {example['essay_id']}\n")
            f.write(f"Label: {example['label']}\n")
            f.write(f"Sentence: {example['sentence']}\n")
            f.write("-----\n")