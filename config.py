"""
PUDF Configuration File
=======================
Update the paths below to match your environment before running the code.
"""

import os

# =============================================================================
# USER CONFIGURATION - MODIFY THESE PATHS
# =============================================================================

# HuggingFace cache directory for storing downloaded models and datasets
# Example: HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
HF_HOME = "/path/to/your/huggingface_cache"

# Directory containing IRT difficulty files for GLUE tasks
# Example: GLUE_DIFFICULTY_DIR = "/afs/crc/group/ball_lab/gmeng_cl/cl_new/gen_difficulty/GLUE_output_difficulty_jsonlines"
GLUE_DIFFICULTY_DIR = "/path/to/your/gen_difficulty/GLUE_output_difficulty_jsonlines"

# Path to MedQA IRT difficulty file (best_parameters.json)
# Example: MEDQA_DIFFICULTY_FILE = "/afs/crc/group/ball_lab/gmeng_cl/cl_new/gen_difficulty/MeD_QA/merged_jsonlines_output/test-1pl/best_parameters.json"
MEDQA_DIFFICULTY_FILE = "/path/to/your/gen_difficulty/MeD_QA/merged_jsonlines_output/test-1pl/best_parameters.json"

# Base output directory for training results (optional)
# Example: OUTPUT_BASE_DIR = "/afs/crc/group/ball_lab/gmeng_cl/cl_new"
OUTPUT_BASE_DIR = "/path/to/your/output_directory"

# Email for HPC job notifications (used in shell scripts)
# Example: USER_EMAIL = "gmeng@nd.edu"
USER_EMAIL = "your_email@example.com"

# =============================================================================
# DO NOT MODIFY BELOW THIS LINE
# =============================================================================

# Set environment variables
os.environ["HF_HOME"] = HF_HOME
os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
