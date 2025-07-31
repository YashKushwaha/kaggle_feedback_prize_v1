import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from src.config import DATASET_PATH

# Load train.csv

file_name = os.path.join(DATASET_PATH, "train.csv")
df = pd.read_csv(file_name, nrows=10)

# Pick one essay
essay_id = df['id'].iloc[0]  # or choose manually
train_folder = os.path.join(DATASET_PATH, "train")
essay_path = Path(train_folder) / f"{essay_id}.txt"

## Read essay text
with open(essay_path, 'r', encoding='utf-8') as f:
    full_text = f.read()

# Get annotations for the essay
essay_df = df[df['id'] == essay_id].copy()
words = full_text.strip().split()

# Discourse type to color map
color_map = {
    'Lead': '#AED6F1',
    'Position': '#A9DFBF',
    'Claim': '#F5B7B1',
    'Evidence': '#FCF3CF',
    'Counterclaim': '#FADBD8',
    'Rebuttal': '#D7DBDD',
    'Concluding Statement': '#D1F2EB'
}

# Initialize labels for each word
word_labels = [''] * len(words)

# Track the highest priority span per word (last wins)
for _, row in essay_df.iterrows():
    indices = list(map(int, row['predictionstring'].split()))
    for idx in indices:
        word_labels[idx] = row['discourse_type']

# Build HTML string
html_tokens = []
for idx, word in enumerate(words):
    label = word_labels[idx]
    if label:
        color = color_map.get(label, '#EEEEEE')
        html_tokens.append(f"<span style='background-color:{color}; padding:2px; margin:1px;' title='{label}'>{word}</span>")
    else:
        html_tokens.append(word)

html_output = f"""
<html>
<head><meta charset="UTF-8"><title>Discourse Highlight: {essay_id}</title></head>
<body style='font-family:Arial, sans-serif; line-height:1.6; font-size:16px;'>
<h2>Essay ID: {essay_id}</h2>
<p>{" ".join(html_tokens)}</p>
</body>
</html>
"""

# Save to HTML
output_path = f"highlighted_{essay_id}.html"
output_path = os.path.join(DATASET_PATH, output_path)
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_output)

print(f"✅ Highlighted HTML exported to: {output_path}")