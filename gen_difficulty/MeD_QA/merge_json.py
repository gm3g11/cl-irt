import json
import os
import re
import glob

# --- Configuration ---

# TASKS_CONFIG: A list of tuples, where each tuple is (folder_name, file_task_pattern_in_filename)
# folder_name: The name of the directory under INPUT_DIR_BASE that contains the JSON files for a task.
# file_task_pattern_in_filename: The specific keyword in your filenames that appears
#                                  AFTER the model name part and BEFORE other metrics or "_epochs_".
#                                  For your case, this should now be "MedQA".
TASKS_CONFIG = [
    # === ACTION REQUIRED: Configure this line for your setup ===
    # 1. Replace "results" with the actual name of your folder if different.
    # 2. Ensure the second item is "MedQA" (or your specific keyword).
    ("results", "MedQA"),
    # ============================================================
    # Example for another task (if you have more):
    # ("another_task_folder", "some_other_keyword"),
]

# Base directory where task-specific result folders (like "results") are located.
# If "results" is in the same directory as this script, INPUT_DIR_BASE can be "."
INPUT_DIR_BASE = "."  # Modify if your "results" folder is elsewhere

# Directory where the output .jsonlines files will be saved
OUTPUT_DIR_BASE = "./merged_jsonlines_output"
# The output filename will be based on the file_task_pattern (e.g., "MedQA.jsonlines")

# --- End Configuration ---

def merge_json_files_for_task(folder_name_for_input, file_task_pattern, input_dir, output_filepath):
    """
    Merges JSON files for a specific task configuration.

    Args:
        folder_name_for_input (str): The name of the folder being processed (e.g., "results").
        file_task_pattern (str): The task pattern string to match in filenames (e.g., "MedQA").
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

    # Print total number of JSON files found for this task
    print(f"Found {len(json_files)} JSON files to process for task '{folder_name_for_input}':")
    for file_path_iter in json_files:
        print(f"  - {os.path.basename(file_path_iter)}")
    print("--- Now starting to process these files... ---")

    subject_data = {}
    processed_files_count = 0
    skipped_files_count = 0

    # Regex Explanation:
    # ^(.*?)_ : Captures the model name part (non-greedy) up to the underscore before the file_task_pattern.
    # ({re.escape(file_task_pattern)}) : Captures your specific keyword (e.g., "MedQA").
    # (?:_.*?)? : Non-capturing group for any characters between your keyword and "_epochs_". This part is optional.
    # _epochs_ : Literal match for "_epochs_".
    # (\d+) : Captures the number of epochs.
    # \.json$ : Ensures the filename ends with ".json".
    regex_pattern = rf"^(.*?)_({re.escape(file_task_pattern)})(?:_.*?)?_epochs_(\d+)\.json$"

    for file_path in json_files:
        file_name = os.path.basename(file_path)
        match = re.match(regex_pattern, file_name)

        if match:
            model_name_part = match.group(1)
            # group(2) is the successfully matched file_task_pattern (e.g., "MedQA")
            epochs_str = match.group(3)

            # Construct the new name as per your requirement (e.g., albert_base_v2_5epochs)
            subject_id = f"{model_name_part}_{epochs_str}epochs"

            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    data = json.load(infile)

                if "responses" in data and isinstance(data["responses"], list):
                    responses_list = data["responses"]
                    response_dict = {f"q{i + 1}": response for i, response in enumerate(responses_list)}

                    # --- Print details for EACH processed file ---
                    print(f"\n  Successfully processed file: {file_name}")
                    print(f"    New Name (subject_id): {subject_id}")
                    print(f"    Number of responses: {len(responses_list)}")
                    # --- End of per-file details ---

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
            print(f"  Info: File '{file_name}' did not match expected pattern for folder '{folder_name_for_input}'.")
            print(f"        Pattern searched for: '{file_task_pattern}' within the structure defined by regex.")
            print(f"        Regex used: {regex_pattern}")
            print(f"        (Ensure the keyword '{file_task_pattern}' is correct and positioned as expected in the filename relative to '_epochs_'). Skipping.")
            skipped_files_count += 1

    if subject_data:
        try:
            os.makedirs(OUTPUT_DIR_BASE, exist_ok=True) # Ensure output directory exists
            with open(output_filepath, "w", encoding="utf-8") as outfile:
                for subject_id_key, responses_val in subject_data.items():
                    json_line = {
                        "subject_id": subject_id_key,
                        "responses": responses_val
                    }
                    outfile.write(json.dumps(json_line) + "\n")

            print(f"\nSuccessfully merged {len(subject_data)} unique subjects into {output_filepath}.")
            print(f"  Files contributing to output: {processed_files_count}.")
            print(f"  Files skipped (no match or error): {skipped_files_count}.")

        except IOError as e:
            print(f"Error: Could not write to output file {output_filepath}: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during file writing for {folder_name_for_input}: {e}")
    else:
        print(f"\nNo valid data found to write for task from folder '{folder_name_for_input}' after processing {len(json_files)} files.")
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

    print(f"Base input directory: {os.path.abspath(INPUT_DIR_BASE)}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR_BASE)}\n")

    try:
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
        print(f"Ensured output directory exists: {OUTPUT_DIR_BASE}")
    except OSError as e:
        print(f"Error: Could not create output directory {OUTPUT_DIR_BASE}: {e}. Exiting.")
        return

    for folder_name, file_task_pattern in TASKS_CONFIG:
        current_task_input_dir = os.path.join(INPUT_DIR_BASE, folder_name)
        # Output filename will be based on the file_task_pattern (e.g., "MedQA.jsonlines")
        current_output_file = os.path.join(OUTPUT_DIR_BASE, f"{file_task_pattern}.jsonlines")

        merge_json_files_for_task(folder_name, file_task_pattern, current_task_input_dir, current_output_file)

    print("--- All configured tasks processed ---")


if __name__ == "__main__":
    # Before running:
    # 1. Make sure `TASKS_CONFIG` is set up correctly as highlighted above.
    #    For your case, it should be similar to:
    #    TASKS_CONFIG = [("results", "MedQA")]
    #    - Adjust "results" if your folder has a different name.
    #    - Ensure the second part is "MedQA" or your actual keyword.
    #
    # 2. Ensure `INPUT_DIR_BASE` points to the directory *containing* your "results" folder.
    #    If "results" (or your equivalent folder) is in the same directory as this script,
    #    `INPUT_DIR_BASE = "."` is correct.
    #
    # 3. Ensure `OUTPUT_DIR_BASE` is set to your desired output location.
    main()