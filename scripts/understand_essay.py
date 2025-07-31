import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
from src.config import DATASET_PATH

# Load train.csv

file_name = os.path.join(DATASET_PATH, "train.csv")
df = pd.read_csv(file_name, nrows=None)

# Pick one essay
essay_id = df['id'].iloc[0]  # or choose manually
train_folder = os.path.join(DATASET_PATH, "train")
essay_path = Path(train_folder) / f"{essay_id}.txt"

## Read essay text
with open(essay_path, 'r', encoding='utf-8') as f:
    full_text = f.read()

# Get annotations for the essay
# --- Annotations ---
essay_df = df[df['id'] == essay_id].copy()
words = full_text.strip().split()

# --- Color Mapping ---
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

for _, row in essay_df.iterrows():
    indices = list(map(int, row['predictionstring'].split()))
    for idx in indices:
        word_labels[idx] = row['discourse_type']

# --- Build HTML Tokens ---
html_tokens = []
for idx, word in enumerate(words):
    label = word_labels[idx]
    if label:
        color = color_map.get(label, '#EEEEEE')
        html_tokens.append(
            f"<span style='background-color:{color}; padding:2px; margin:1px;' title='{label}'>{word}</span>")
    else:
        html_tokens.append(word)

highlighted_text = " ".join(html_tokens)

# --- Build Legend ---
legend_items = []
for label, color in color_map.items():
    legend_items.append(
        f"<span style='background-color:{color}; padding:4px 8px; margin-right:8px; border-radius:4px;'>{label}</span>"
    )
legend_html = "<div style='margin-bottom:16px;'>" + " ".join(legend_items) + "</div>"

# --- Final HTML ---
html_output = f"""
<html>
<head>
    <meta charset="UTF-8">
    <title>Discourse Highlight: {essay_id}</title>
</head>
<body style='font-family:Arial, sans-serif; line-height:1.6; font-size:16px; padding:20px;'>
    <h2>Essay ID: {essay_id}</h2>
    {legend_html}
    <p>{highlighted_text}</p>
</body>
</html>
"""

# --- Export HTML ---
output_path = os.path.join(DATASET_PATH, f"highlighted_{essay_id}.html")
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(html_output)

print(f"✅ Highlighted HTML exported to: {output_path}")


import matplotlib.pyplot as plt

# Plot frequency of discourse types
plt.figure(figsize=(10, 5))
df['discourse_type'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Distribution of Discourse Types")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--')
plt.tight_layout()

# Save to file (e.g., PNG or PDF)
fig_name = "discourse_type_distribution.png"
fig_name = os.path.join(DATASET_PATH, fig_name)
plt.savefig(fig_name, dpi=300)  # or .pdf, .svg, etc.

# Optional: close the plot to free memory if running in a loop
plt.close()
