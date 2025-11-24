import os
import json
import numpy as np

RESULTS_DIR = "ag_news_final_results"
OUTPUT_FILENAME = "merged_model_outputs_with_true_labels.json"  # Changed output filename


def extract_model_name_from_filename(filename):
    """
    Extracts the base model name from the filename.
    e.g., 'albert_base_v2_ag_news_...json' -> 'albert'
    """
    return filename.split('_')[0]


def process_json_files():
    """
    Goes through specified JSON files, extracts logits, responses (predictions),
    and true labels, prints their lengths, and generates a merged JSON file.
    """
    print(f"Starting to process files in: {RESULTS_DIR}\n")

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

                # --- Expected key names ---
                # Please verify these key names against your actual JSON files.
                logits_key = "logits"
                responses_key = "responses"  # Based on your previous successful output
                true_labels_key = "labels_true"  # Common key name for true labels, adjust if different
                # Other possibilities: "true_labels", "ground_truth", "labels"

                logits = data.get(logits_key)
                responses = data.get(responses_key)
                true_labels = data.get(true_labels_key)

                if logits is not None and responses is not None and true_labels is not None:
                    num_examples_logits = len(logits) if isinstance(logits, list) else 0
                    num_classes = len(logits[0]) if num_examples_logits > 0 and isinstance(logits[0], list) else 'N/A'
                    num_examples_responses = len(responses) if isinstance(responses, list) else 0
                    num_examples_true_labels = len(true_labels) if isinstance(true_labels, list) else 0

                    print(f"  Data for {model_name_short}:")
                    print(f"    Logits extracted: {num_examples_logits} examples, {num_classes} classes per example.")
                    print(f"    Responses extracted: {num_examples_responses} examples.")
                    print(f"    True Labels extracted: {num_examples_true_labels} examples.")

                    if not (num_examples_logits == num_examples_responses == num_examples_true_labels):
                        print(f"    Warning for {model_name_short}: Length mismatch! "
                              f"Logits ({num_examples_logits}), Responses ({num_examples_responses}), True Labels ({num_examples_true_labels}).")

                    merged_data[model_name_short] = {
                        "logits": logits,
                        "responses": responses,
                        "true_labels": true_labels  # Adding true labels
                    }
                    files_processed_count += 1
                    print(f"    Successfully extracted and stored data (including true labels) for {model_name_short}.")
                else:
                    missing_keys_info = []
                    if logits is None:
                        missing_keys_info.append(f"'{logits_key}' (for logits)")
                    if responses is None:
                        missing_keys_info.append(f"'{responses_key}' (for responses)")
                    if true_labels is None:
                        missing_keys_info.append(f"'{true_labels_key}' (for true labels)")

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
    process_json_files()