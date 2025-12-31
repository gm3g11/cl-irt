import os
import json
import numpy as np
import random
import torch  # For random seed
from datasets import load_dataset  # To load ag_news

RESULTS_DIR = "ag_news_final_results"
OUTPUT_FILENAME = "merged_model_outputs_with_true_labels.json"  # Updated filename
DATASET_ID = "contemmcm/ag_news"
RANDOM_SEED = 63  # From your reference script

# --- Environment Setup (Minimal for this script's needs) ---
# Import paths from central config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME, GLUE_DIFFICULTY_DIR, MEDQA_DIFFICULTY_FILE  # From your reference
HF_DATASETS_CACHE = os.path.join(HF_HOME, "datasets")
os.environ["HF_HOME"] = HF_HOME  # Set HF_HOME for datasets to use
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.makedirs(HF_DATASETS_CACHE, exist_ok=True)


def set_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    print(f"Global random seed set to: {seed_val}")


def load_and_prepare_true_labels():
    """
    Loads the ag_news dataset, applies the same filtering and splitting
    as in the reference script, and returns the true labels from the
    'dataset_for_evaluation' split.
    """
    print(f"Loading dataset: {DATASET_ID} to extract true labels...")
    try:
        raw_dataset_full = load_dataset(DATASET_ID, cache_dir=HF_DATASETS_CACHE)
    except Exception as e:
        print(f"Error loading dataset {DATASET_ID}: {e}")
        return None

    if 'complete' not in raw_dataset_full:
        print(f"Error: Dataset {DATASET_ID} does not have a 'complete' split.")
        return None
    complete_dataset_view = raw_dataset_full['complete']

    label_feature = complete_dataset_view.features['label']
    num_labels_from_dataset = 0
    if hasattr(label_feature, 'num_classes'):
        num_labels_from_dataset = label_feature.num_classes
    elif hasattr(label_feature, 'names'):
        num_labels_from_dataset = len(label_feature.names)
    else:
        unique_labels = sorted(list(set(complete_dataset_view.unique("label")['label'])))
        if unique_labels and all(isinstance(l, int) for l in unique_labels):
            num_labels_from_dataset = len(unique_labels)
        else:
            print("Error: Could not dynamically determine number of labels for filtering.")
            return None

    print(f"Filtering 'complete' split for labels 0 to {num_labels_from_dataset - 1}...")
    complete_filtered = complete_dataset_view.filter(
        lambda example: 0 <= example['label'] < num_labels_from_dataset,
        num_proc=min(4, os.cpu_count() if os.cpu_count() else 1)  # Use num_proc for potentially large dataset
    )

    if len(complete_filtered) == 0:
        print("Error: Dataset is empty after filtering.")
        return None

    complete_filtered = complete_filtered.shuffle(seed=RANDOM_SEED)

    # Splitting: 80% for evaluation (where logits/responses were generated from)
    # and 20% for fine-tuning (not used here, but matches reference split)
    train_temp_split = complete_filtered.train_test_split(test_size=0.2, seed=RANDOM_SEED)
    dataset_for_evaluation = train_temp_split['train']  # This is the 80% split

    print(f"  'dataset_for_evaluation' (80% split) has {len(dataset_for_evaluation)} examples.")

    true_labels = dataset_for_evaluation['label']  # This is a list of integers

    if len(true_labels) != 995282:
        print(f"Warning: Extracted {len(true_labels)} true labels, but expected 995282. "
              "Ensure dataset processing matches the one used for generating logits/responses.")
    else:
        print(f"Successfully extracted {len(true_labels)} true labels from 'dataset_for_evaluation'.")

    return true_labels


def extract_model_name_from_filename(filename):
    return filename.split('_')[0]


def process_json_files(ground_truth_labels):
    print(f"\nStarting to process files in: {RESULTS_DIR}")

    if ground_truth_labels is None:
        print("Error: Ground truth labels not loaded. Cannot proceed with merging.")
        return

    if not os.path.isdir(RESULTS_DIR):
        print(f"Error: Directory '{RESULTS_DIR}' not found.")
        return

    merged_data = {}
    files_processed_count = 0
    found_target_files_count = 0

    for filename in os.listdir(RESULTS_DIR):
        if filename.endswith("epochs_5.json"):
            found_target_files_count += 1
            model_name_short = extract_model_name_from_filename(filename)
            file_path = os.path.join(RESULTS_DIR, filename)

            print(f"Processing file: {filename} (for model: {model_name_short})")

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                logits_key = "logits"
                responses_key = "responses"

                logits = data.get(logits_key)
                responses = data.get(responses_key)

                if logits is not None and responses is not None:
                    num_examples_logits = len(logits) if isinstance(logits, list) else 0
                    num_classes = len(logits[0]) if num_examples_logits > 0 and isinstance(logits[0], list) else 'N/A'
                    num_examples_responses = len(responses) if isinstance(responses, list) else 0
                    num_examples_ground_truth = len(ground_truth_labels)

                    print(f"  Data for {model_name_short}:")
                    print(f"    Logits extracted: {num_examples_logits} examples, {num_classes} classes per example.")
                    print(f"    Responses (predictions) extracted: {num_examples_responses} examples.")
                    print(f"    Ground Truth Labels available: {num_examples_ground_truth} examples.")

                    if not (num_examples_logits == num_examples_responses == num_examples_ground_truth):
                        print(f"    Warning for {model_name_short}: Length mismatch! "
                              f"Logits ({num_examples_logits}), Responses ({num_examples_responses}), Ground Truth Labels ({num_examples_ground_truth}). "
                              "Will not add ground truth labels for this model if counts don't match logits/responses.")
                        # Store only logits and responses if there's a mismatch with ground truth length for this specific file
                        merged_data[model_name_short] = {
                            "logits": logits,
                            "responses": responses
                        }
                        print(
                            f"    Stored logits and responses for {model_name_short}. Ground truth labels skipped due to length mismatch with this file's data.")

                    else:  # Lengths match
                        if num_examples_ground_truth == 995282:
                            print(
                                f"    Confirmed: Ground Truth Labels count ({num_examples_ground_truth}) matches expected 995282 and logits/responses.")
                        else:
                            print(
                                f"    Note: Ground Truth Labels count is {num_examples_ground_truth} (expected 995282). Storing anyway as it matches this file's logits/responses.")

                        merged_data[model_name_short] = {
                            "logits": logits,
                            "responses": responses,
                            "true_labels": ground_truth_labels  # Adding ground truth labels
                        }
                        print(
                            f"    Successfully extracted and stored data (including ground truth labels) for {model_name_short}.")

                    files_processed_count += 1  # Count as processed if logits & responses are found
                else:
                    missing_keys_info = []
                    if logits is None: missing_keys_info.append(f"'{logits_key}' (for logits)")
                    if responses is None: missing_keys_info.append(f"'{responses_key}' (for responses)")
                    print(
                        f"    Warning: Expected key(s) {', '.join(missing_keys_info)} not found or have null value in {filename}.")
                    print(f"    Available top-level keys in this file are: {list(data.keys())}")

            except json.JSONDecodeError:
                print(f"    Error: Could not decode JSON from {filename}.")
            except Exception as e:
                print(f"    Error processing file {filename}: {e}")
            print("-" * 30)

    if found_target_files_count == 0:
        print(f"\nNo files ending with 'epochs_5.json' found in '{RESULTS_DIR}'.")
        return

    if files_processed_count > 0:
        try:
            with open(OUTPUT_FILENAME, 'w') as outfile:
                json.dump(merged_data, outfile, indent=4)
            print(
                f"\nSuccessfully created merged file: {OUTPUT_FILENAME} with data from {files_processed_count} model(s).")
        except Exception as e:
            print(f"\nError writing merged file {OUTPUT_FILENAME}: {e}")
    else:
        print(
            f"\nNo data was successfully extracted from any matching files. Merged file '{OUTPUT_FILENAME}' not created.")


if __name__ == "__main__":
    set_random_seed(RANDOM_SEED)  # Set seed for dataset shuffling consistency
    true_labels_for_evaluation = load_and_prepare_true_labels()

    if true_labels_for_evaluation:
        process_json_files(true_labels_for_evaluation)
    else:
        print("Could not load ground truth labels. Aborting processing of JSON files.")