import os
import json
from datasets import load_dataset

#–– CONFIG ––#
RESULTS_DIR = "results"
OUTPUT_FILE = "MedQA_logits_responses_with_true_labels.json"
DATASET_ID = "GBaker/MedQA-USMLE-4-options"
TARGET_EPOCH = 5

#–– 1. Load train split and build true_labels ––#
# Map answer letters to integers
label_map = {"A": 0, "B": 1, "C": 2, "D": 3}

# Load and filter
ds = load_dataset(DATASET_ID)["train"]
ds = ds.filter(lambda ex: ex["answer_idx"] is not None and
                         ex["answer_idx"].strip().upper() in label_map)

# Extract numeric labels
true_labels = [
    label_map[ex["answer_idx"].strip().upper()]
    for ex in ds
]

#–– 2. Scan results folder and merge logits/responses ––#
merged = {}

for fname in sorted(os.listdir(RESULTS_DIR)):
    # Match files ending in _epochs_5.json
    if fname.endswith(f"_epochs_{TARGET_EPOCH}.json"):
        model_key = fname.split("_", 1)[0]  # 'albert_base_v2_...' → 'albert'
        path = os.path.join(RESULTS_DIR, fname)
        with open(path, "r") as f:
            data = json.load(f)
        logits = data.get("logits", [])
        responses = data.get("responses", [])

        if model_key in merged:
            merged[model_key]["logits"].extend(logits)
            merged[model_key]["responses"].extend(responses)
        else:
            merged[model_key] = {
                "logits": logits,
                "responses": responses
            }

#–– 3. Add true_labels and dump the final JSON ––#
merged["true_labels"] = true_labels

with open(OUTPUT_FILE, "w") as f:
    json.dump(merged, f, indent=2)

print(f"Merged results written to {OUTPUT_FILE}")
