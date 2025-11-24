import json
import os
import re
import glob

# --- Configuration ---

# TASKS_CONFIG: A list of tuples, where each tuple is (folder_name, file_task_pattern_in_filename)
# folder_name: The name of the directory under INPUT_DIR_BASE that contains the JSON files for a task.
# file_task_pattern_in_filename: The specific task identifier string expected within the JSON filenames
#                                  (e.g., "ag_news" for files like "..._ag_news_...epochs_0.json").
TASKS_CONFIG = [
    ("ag_news_final_results", "ag_news"),
    # Example for another task:
    # ("some_other_task_folder", "some_other_task_pattern_in_file"),
]

# Base directory where task-specific result folders (like "ag_news_final_results") are located.
# If "ag_news_final_results" is in the same directory as this script, INPUT_DIR_BASE can be "."
INPUT_DIR_BASE = "."  # Modify if your "ag_news_final_results" folder is elsewhere

# Directory where the output .jsonlines files will be saved
OUTPUT_DIR_BASE = "./merged_jsonlines_output"


# --- End Configuration ---

def merge_json_files_for_task(folder_name_for_input, file_task_pattern, input_dir, output_filepath):
    """
    Merges JSON files for a specific task configuration.

    Args:
        folder_name_for_input (str): The name of the folder being processed (e.g., "ag_news_final_results").
        file_task_pattern (str): The task pattern string to match in filenames (e.g., "ag_news").
        input_dir (str): Full path to the directory containing the JSON files for this task.
        output_filepath (str): Path to the output .jsonlines file for this task.
    """
    print(f"--- Processing task from folder: '{folder_name_for_input}' (using file pattern: '{file_task_pattern}') ---")
    print(f"Attempting to read input from: {input_dir}")

    if not os.path.isdir(input_dir):
        print(f"Warning: Input directory not found: {input_dir}. Skipping task '{folder_name_for_input}'.")
        print(
            f"Please ensure the directory exists and the INPUT_DIR_BASE ('{INPUT_DIR_BASE}') and folder name ('{folder_name_for_input}') are correct.")
        return

    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    if not json_files:
        print(f"Warning: No .json files found in {input_dir}. Skipping task '{folder_name_for_input}'.")
        return

    print(f"Found {len(json_files)} JSON files to process for task '{folder_name_for_input}':")
    for file_path in json_files:
        print(f"  - {os.path.basename(file_path)}")
    print("--- Now starting to process these files... ---")

    subject_data = {}
    processed_files_count = 0
    skipped_files_count = 0

    # Updated Regex:
    # ^(.*?)_          - Capture group 1: Model name part (non-greedy, up to the first underscore before task pattern)
    # ({re.escape(file_task_pattern)})_ - Capture group 2: The specific task pattern from filename
    # .*?_epochs_      - Any characters (non-greedy) up to "_epochs_"
    # (\d+)            - Capture group 3: Number of epochs
    # \.json$          - Ends with .json
    # Example: albert_base_v2_ag_news_20ft_80eval_0.02_Acc_0.0726_epochs_0.json
    # Group 1 (model_name_part): albert_base_v2
    # Group 2 (task_pattern_in_file): ag_news
    # Group 3 (epochs_str): 0
    regex_pattern = rf"^(.*?)_({re.escape(file_task_pattern)})_.*?_epochs_(\d+)\.json$"

    for file_path in json_files:
        file_name = os.path.basename(file_path)
        match = re.match(regex_pattern, file_name)

        if match:
            model_name_part = match.group(1)
            # task_name_from_file = match.group(2) # This is file_task_pattern, e.g., "ag_news"
            epochs_str = match.group(3)

            subject_id = f"{model_name_part}_{epochs_str}_epochs"

            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    data = json.load(infile)

                if "responses" in data and isinstance(data["responses"], list):
                    responses_list = data["responses"]
                    response_dict = {f"q{i + 1}": response for i, response in enumerate(responses_list)}

                    if subject_id in subject_data:
                        print(
                            f"  Warning: Duplicate subject_id '{subject_id}' found from file '{file_name}'. Overwriting previous data for this subject_id.")
                    subject_data[subject_id] = response_dict
                    processed_files_count += 1
                else:
                    print(f"  Warning: 'responses' key missing or not a list in file: {file_name}. Skipping.")
                    skipped_files_count += 1

            except json.JSONDecodeError:
                print(f"  Warning: Could not decode JSON from file: {file_name}. Skipping.")
                skipped_files_count += 1
            except Exception as e:
                print(f"  Warning: An error occurred processing file {file_name}: {e}. Skipping.")
                skipped_files_count += 1
        else:
            print(f"  Info: File '{file_name}' did not match expected pattern for folder '{folder_name_for_input}' (pattern component: '{file_task_pattern}'). Skipping.")
            skipped_files_count += 1

    if subject_data:
        try:
            with open(output_filepath, "w", encoding="utf-8") as outfile:
                for subject_id, responses in subject_data.items():
                    json_line = {
                        "subject_id": subject_id,
                        "responses": responses
                    }
                    outfile.write(json.dumps(json_line) + "\n")

            print(f"Successfully merged {len(subject_data)} unique subjects into {output_filepath}.")
            print(f"  Files contributing to output: {processed_files_count}.")
            print(f"  Files skipped (no match or error): {skipped_files_count}.")

        except IOError as e:
            print(f"Error: Could not write to output file {output_filepath}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during file writing for {folder_name_for_input}: {e}")
    else:
        print(f"No valid data found to write for task from folder '{folder_name_for_input}' after processing {len(json_files)} files.")
        print(f"  Files successfully read and matched pattern: {processed_files_count}")
        print(f"  Files skipped (no match or error): {skipped_files_count}")
        print(f"Output file '{output_filepath}' will be empty or not created if it didn't exist.")

    print(f"--- Finished processing: {folder_name_for_input} ---\n")


def main():
    """
    Main function to orchestrate the merging process for all configured tasks.
    """
    print(f"Starting JSON merging process.")
    if TASKS_CONFIG:
        print(f"Tasks to process (folder_name, file_task_pattern): {TASKS_CONFIG}")
    else:
        print("No tasks configured in TASKS_CONFIG. Exiting.")
        return

    print(f"Base input directory: {os.path.abspath(INPUT_DIR_BASE)}") # Show absolute path for clarity
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR_BASE)}\n")

    try:
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
        print(f"Ensured output directory exists: {OUTPUT_DIR_BASE}")
    except OSError as e:
        print(f"Error: Could not create output directory {OUTPUT_DIR_BASE}: {e}. Exiting.")
        return

    for folder_name, file_task_pattern in TASKS_CONFIG:
        current_task_input_dir = os.path.join(INPUT_DIR_BASE, folder_name)
        # Output filename will be based on the file_task_pattern (e.g., "ag_news.jsonlines")
        current_output_file = os.path.join(OUTPUT_DIR_BASE, f"{file_task_pattern}.jsonlines")

        merge_json_files_for_task(folder_name, file_task_pattern, current_task_input_dir, current_output_file)

    print("--- All configured tasks processed ---")


if __name__ == "__main__":
    # Before running:
    # 1. Make sure `TASKS_CONFIG` is set up correctly.
    #    For your case: `TASKS_CONFIG = [("ag_news_final_results", "ag_news")]`
    # 2. Ensure `INPUT_DIR_BASE` points to the directory *containing* "ag_news_final_results".
    #    If "ag_news_final_results" is in the same directory as this script, `INPUT_DIR_BASE = "."` is correct.
    #    If "ag_news_final_results" is in, for example, "C:/data/", then `INPUT_DIR_BASE = "C:/data/"`.
    main()