import json
import os
import re

def merge_json_files(input_dir, output_file, task):
    """
    Merge multiple JSON files into a single JSON Lines file with "subject_id" and "responses".

    Args:
        input_dir (str): Directory containing the JSON files.
        output_file (str): Path to the output .jsonlines file.
        task (str): The task name (e.g., "mnli").
    """
    subject_data = {}

    # Iterate through all files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".json"):
            # Extract model name and epochs from the file name
            match = re.search(r"(.+?)_" + task + r"_.*_finetuning_(\d+)_epochs", file_name)
            if match:
                model_name, epochs = match.groups()
                subject_id = f"{model_name}_{epochs}"

                # Load the JSON data from the file
                file_path = os.path.join(input_dir, file_name)
                with open(file_path, "r") as infile:
                    data = json.load(infile)
                    responses_list = data["responses"]  # Assumes "responses" key exists

                    # Map responses to "q1", "q2", etc.
                    response_dict = {f"q{i+1}": response for i, response in enumerate(responses_list)}

                    # Store the subject's data
                    subject_data[subject_id] = response_dict

    # Write the merged data to the output file
    with open(output_file, "w") as outfile:
        for subject_id, responses in subject_data.items():
            json_line = {
                "subject_id": subject_id,
                "responses": responses
            }
            outfile.write(json.dumps(json_line) + "\n")

    # Verify the output
    with open(output_file, "r") as f:
        lines = f.readlines()
        if len(lines) == 0:
            print("Warning: The output file is empty. Check the input directory and file names.")
        else:
            print(f"Successfully merged {len(lines)} subjects into {output_file}.")

# Example usage
input_dir = "GLUE_results/mnli"  # Adjust to your directory path
output_file = "mnli.jsonlines"
task = "mnli"
merge_json_files(input_dir, output_file, task)