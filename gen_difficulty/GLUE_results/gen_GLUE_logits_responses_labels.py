import os
import json
import numpy as np
import random
import torch  # For random seed
from datasets import load_dataset

# --- Configuration ---
GLUE_TASKS = ["mrpc", "qnli", "qqp", "mnli", "rte", "sst2"]
# Base directory where task-specific folders (mrpc, qnli, etc.) are located.
# Assumes these folders are in the same directory as the script, or provide full path.
BASE_RESULTS_DIR = "."  # Current directory, change if task folders are elsewhere

# Output directory for the merged files
MERGED_OUTPUT_DIR = "./glue_merged_outputs"
os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)

HF_HOME_DEFAULT = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
RANDOM_SEED = 63


# --- HF Cache Setup ---
def setup_huggingface_cache(hf_home_path=None):
    if hf_home_path is None:
        hf_home_path = HF_HOME_DEFAULT

    os.environ["HF_HOME"] = hf_home_path
    datasets_cache_path = os.path.join(hf_home_path, "datasets")
    os.environ["HF_DATASETS_CACHE"] = datasets_cache_path
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Common setting
    os.makedirs(datasets_cache_path, exist_ok=True)

    print(f"Hugging Face datasets cache set to: {datasets_cache_path}")
    return datasets_cache_path


def set_global_random_seed(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
    print(f"Global random seed set to: {seed_val}")


def extract_model_name_from_glue_filename(filename, task_name):
    """
    Extracts the model name from filenames like:
    'albert-base-v2_mnli_52.28..._5_epochs.json' -> 'albert-base-v2'
    'bert-base-uncased_mrpc_..._5_epochs.json' -> 'bert-base-uncased'
    """
    task_marker = f"_{task_name}_"
    if task_marker in filename:
        return filename.split(task_marker)[0]
    else:
        # Fallback: try to get part before the first underscore if task marker not found,
        # but this might be too simplistic if model names have underscores.
        # A more robust fallback might be needed if filenames are inconsistent.
        print(
            f"  Warning: Task marker '{task_marker}' not found in filename '{filename}'. Using part before first '_' as model name.")
        return filename.split('_')[0]


def process_glue_tasks():
    """
    Processes each GLUE task to merge logits, responses, and true labels.
    """
    datasets_cache_dir = setup_huggingface_cache()
    set_global_random_seed(RANDOM_SEED)

    overall_summary = {}

    for task in GLUE_TASKS:
        print(f"\nProcessing GLUE Task: {task.upper()}")
        print("=" * 40)

        # 1. Load ground truth labels for the current task's "train" split
        ground_truth_labels = []
        expected_label_length = 0
        try:
            print(f"  Loading 'train' split for {task} to get ground truth labels...")
            dataset = load_dataset("glue", task, cache_dir=datasets_cache_dir)
            if 'train' not in dataset:
                print(f"  Error: 'train' split not found for task {task}. Skipping this task for labels.")
                overall_summary[task] = {"status": "Error: 'train' split not found", "models_processed": 0}
                continue

            # Convert to list for consistent JSON serialization and ease of use
            ground_truth_labels = list(dataset['train']['label'])
            expected_label_length = len(ground_truth_labels)
            print(
                f"  Successfully loaded {expected_label_length} ground truth labels for task '{task}' from its 'train' split.")
        except Exception as e:
            print(f"  Error loading ground truth labels for task {task}: {e}. Skipping this task.")
            overall_summary[task] = {"status": f"Error loading labels: {e}", "models_processed": 0}
            continue

        task_results_dir = os.path.join(BASE_RESULTS_DIR, task)
        if not os.path.isdir(task_results_dir):
            print(f"  Directory for task {task} not found: {task_results_dir}. Skipping this task.")
            overall_summary[task] = {"status": f"Directory not found: {task_results_dir}", "models_processed": 0}
            continue

        task_merged_data = {}
        files_processed_for_task = 0

        print(f"  Searching for '*_5_epochs.json' files in: {task_results_dir}")
        for filename in os.listdir(task_results_dir):
            if filename.endswith("_5_epochs.json"):
                model_name_short = extract_model_name_from_glue_filename(filename, task)
                file_path = os.path.join(task_results_dir, filename)

                print(f"    Processing file: {filename} (Model: {model_name_short})")

                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    logits_key = "logits"
                    responses_key = "responses"

                    logits = data.get(logits_key)
                    responses = data.get(responses_key)

                    if logits is not None and responses is not None:
                        num_examples_logits = len(logits) if isinstance(logits, list) else 0
                        num_classes_logits = len(logits[0]) if num_examples_logits > 0 and isinstance(logits[0],
                                                                                                      list) else 'N/A'
                        num_examples_responses = len(responses) if isinstance(responses, list) else 0

                        print(f"      Logits found: {num_examples_logits} examples, {num_classes_logits} classes.")
                        print(f"      Responses found: {num_examples_responses} examples.")
                        print(f"      Ground truth labels for this task ({task}): {expected_label_length} examples.")

                        if num_examples_logits == expected_label_length and num_examples_responses == expected_label_length:
                            task_merged_data[model_name_short] = {
                                "logits": logits,
                                "responses": responses,
                                "true_labels": ground_truth_labels  # Add common true labels for this task
                            }
                            files_processed_for_task += 1
                            print(
                                f"      Successfully extracted and stored data for {model_name_short} with true labels.")
                        else:
                            print(f"      Warning for {model_name_short} on task {task}: Length mismatch! "
                                  f"Logits ({num_examples_logits}), Responses ({num_examples_responses}) vs "
                                  f"Ground Truth Labels ({expected_label_length}). "
                                  f"This model's data for this task will NOT be fully merged with true labels if lengths differ.")
                            # Optionally, still save logits and responses if they match each other but not GT
                            if num_examples_logits == num_examples_responses:
                                task_merged_data[model_name_short] = {
                                    "logits": logits,
                                    "responses": responses,
                                    "true_labels_NOTE": "Length mismatch with original dataset labels for this task."
                                }
                                files_processed_for_task += 1  # Still counts as processed for logits/responses
                                print(
                                    f"      Stored logits and responses for {model_name_short}; true labels association problematic due to length mismatch.")
                            else:
                                print(
                                    f"      Skipping {model_name_short} for task {task} due to internal logit/response length mismatch.")


                    else:
                        missing_keys_info = []
                        if logits is None: missing_keys_info.append(f"'{logits_key}' (for logits)")
                        if responses is None: missing_keys_info.append(f"'{responses_key}' (for responses)")
                        print(
                            f"      Warning: Key(s) {', '.join(missing_keys_info)} not found or have null value in {filename}.")
                        print(f"      Available top-level keys: {list(data.keys())}")

                except json.JSONDecodeError:
                    print(f"      Error: Could not decode JSON from {filename}.")
                except Exception as e:
                    print(f"      Error processing file {filename}: {e}")
                print("      " + "-" * 20)

        if files_processed_for_task > 0:
            output_filename_task = os.path.join(MERGED_OUTPUT_DIR, f"{task}_logits_responses_with_true_labels.json")
            try:
                with open(output_filename_task, 'w') as outfile:
                    json.dump(task_merged_data, outfile, indent=4)
                print(f"  Successfully created merged file for task {task}: {output_filename_task}")
                overall_summary[task] = {"status": "Success", "models_processed": files_processed_for_task,
                                         "output_file": output_filename_task}
            except Exception as e:
                print(f"  Error writing merged file for task {task} ({output_filename_task}): {e}")
                overall_summary[task] = {"status": f"Error writing merged file: {e}",
                                         "models_processed": files_processed_for_task}
        else:
            print(f"  No data successfully processed for task {task}. No merged file created.")
            if os.path.exists(task_results_dir):  # Check if directory existed before saying no files found
                overall_summary[task] = {"status": "No valid JSON files or data found to process",
                                         "models_processed": 0}

    print("\n\n===== Overall Script Summary =====")
    for task, summary in overall_summary.items():
        print(f"Task {task}: Status: {summary['status']}, Models Processed: {summary.get('models_processed', 0)}")
        if "output_file" in summary:
            print(f"  -> Output: {summary['output_file']}")
    print("================================")


if __name__ == "__main__":
    process_glue_tasks()