import os
import json
import numpy as np  # Retained in case logits/responses are loaded in a way that benefits from numpy conversion later

RESULTS_DIR = "ag_news_final_results"
OUTPUT_FILENAME = "merged_model_outputs.json"


def extract_model_name_from_filename(filename):
    """
    Extracts the base model name from the filename.
    e.g., 'albert_base_v2_ag_news_...json' -> 'albert'
    """
    return filename.split('_')[0]


def process_json_files():
    """
    Goes through specified JSON files, extracts logits and responses,
    prints their lengths, prints available keys if expected ones are missing,
    and generates a merged JSON file.
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
                logits_key = "logits"
                responses_key = "responses"  # UPDATED based on your output

                logits = data.get(logits_key)
                responses = data.get(responses_key)

                if logits is not None and responses is not None:
                    num_examples_logits = len(logits) if isinstance(logits, list) else 0
                    num_classes = len(logits[0]) if num_examples_logits > 0 and isinstance(logits[0], list) else 'N/A'
                    num_examples_responses = len(responses) if isinstance(responses, list) else 0

                    print(f"  Data for {model_name_short}:")
                    print(f"    Logits extracted: {num_examples_logits} examples, {num_classes} classes per example.")
                    print(f"    Responses extracted: {num_examples_responses} examples.")

                    if num_examples_logits != num_examples_responses:
                        print(
                            f"    Warning for {model_name_short}: Length mismatch between logits ({num_examples_logits}) and responses ({num_examples_responses}).")

                    merged_data[model_name_short] = {
                        "logits": logits,
                        "responses": responses
                    }
                    files_processed_count += 1
                    print(f"    Successfully extracted and stored data for {model_name_short}.")
                else:
                    missing_keys_info = []
                    if logits is None:
                        missing_keys_info.append(f"'{logits_key}' (for logits)")
                    if responses is None:
                        missing_keys_info.append(f"'{responses_key}' (for responses)")

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