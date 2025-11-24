import json
import os
import re
import glob

# --- Configuration ---
# List of GLUE tasks to process
GLUE_TASKS = ["mrpc", "qnli", "qqp", "mnli", "rte", "sst2"]

# Base directory where task-specific result folders are located
# (e.g., ./results/mnli, ./results/mrpc, etc.)
INPUT_DIR_BASE = "./GLUE_results"

# Directory where the output .jsonlines files will be saved
OUTPUT_DIR = "./GLUE_output_difficulty_jsonlines"
# --- End Configuration ---

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Starting processing for tasks: {', '.join(GLUE_TASKS)}")
print(f"Reading input from base directory: {INPUT_DIR_BASE}")
print(f"Writing output to directory: {OUTPUT_DIR}\n")

# Loop through each specified GLUE task
for task in GLUE_TASKS:
    task_input_dir = os.path.join(INPUT_DIR_BASE, task)
    output_filepath = os.path.join(OUTPUT_DIR, f"{task}.jsonlines")

    print(f"--- Processing task: {task} ---")

    # Check if the input directory for the task exists
    if not os.path.isdir(task_input_dir):
        print(f"Warning: Input directory not found for task '{task}': {task_input_dir}. Skipping.")
        continue

    # Find all .json files in the task's input directory
    input_files = glob.glob(os.path.join(task_input_dir, "*.json"))

    if not input_files:
        print(f"Warning: No .json files found in {task_input_dir}. Skipping task '{task}'.")
        continue

    print(f"Found {len(input_files)} JSON files to process.")

    # Open the output .jsonlines file in write mode (clears existing file)
    try:
        with open(output_filepath, 'w') as outfile:
            processed_count = 0
            skipped_count = 0
            # Process each input JSON file
            for input_filepath in input_files:
                filename = os.path.basename(input_filepath)

                # Attempt to parse the filename to extract model name part and epochs
                # Regex explanation:
                # (.*?)          - Capture group 1: Model name part (non-greedy match)
                # _              - Literal underscore
                # {re.escape(task)} - The specific task name (escaped in case of special chars)
                # _              - Literal underscore
                # .*?            - Any characters (non-greedy match for time, accuracy etc.)
                # _finetuning_   - Literal string
                # (\d+)          - Capture group 2: Number of epochs (one or more digits)
                # _epochs\.json$ - Literal string ending with .json
                match = re.match(rf"(.*?)_({re.escape(task)})_.*?_finetuning_(\d+)_epochs\.json$", filename)

                if not match:
                    print(f"  Warning: Could not parse filename format: {filename}. Skipping.")
                    skipped_count += 1
                    continue

                model_name_part = match.group(1)
                epochs = match.group(3)
                subject_id = f"{model_name_part}_{epochs}_epochs"

                try:
                    # Read the content of the input JSON file
                    with open(input_filepath, 'r') as infile:
                        data = json.load(infile)

                    # Ensure 'responses' key exists and is a list
                    if 'responses' not in data or not isinstance(data['responses'], list):
                        print(f"  Warning: 'responses' key missing or not a list in {filename}. Skipping.")
                        skipped_count += 1
                        continue

                    raw_responses = data['responses']

                    # Format the responses into the desired dictionary format {"q1": r1, "q2": r2, ...}
                    formatted_responses = {f"q{i+1}": response for i, response in enumerate(raw_responses)}

                    # Create the final dictionary for the JSON line
                    output_data = {
                        "subject_id": subject_id,
                        "responses": formatted_responses
                    }

                    # Convert the dictionary to a JSON string and write it as a line
                    outfile.write(json.dumps(output_data) + '\n')
                    processed_count += 1

                except json.JSONDecodeError:
                    print(f"  Warning: Invalid JSON content in {filename}. Skipping.")
                    skipped_count += 1
                except Exception as e:
                    print(f"  Error processing file {filename}: {e}. Skipping.")
                    skipped_count += 1

            print(f"Finished processing task '{task}'.")
            print(f"  Successfully processed: {processed_count} files.")
            print(f"  Skipped / Errored: {skipped_count} files.")
            print(f"  Output saved to: {output_filepath}\n")

    except IOError as e:
        print(f"Error opening or writing to output file {output_filepath}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while processing task {task}: {e}")


print("--- All tasks processed ---")