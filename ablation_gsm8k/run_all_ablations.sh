#!/bin/bash

# ============================================================================
# Run All Curriculum Learning Ablation Studies
# ============================================================================

set -e  # Exit on error

echo "================================================================================"
echo "GSM8K Curriculum Learning - Full Ablation Study Suite"
echo "================================================================================"
echo ""
echo "This script will run all experiments:"
echo "  - Baseline 1: Heuristic CL (4 experiments)"
echo "  - Baseline 2: PUDF (1 experiment)"
echo "  - Ablation 1: IRT Diff + Heuristic Sched (2 experiments)"
echo "  - Ablation 2: Heuristic Diff + IRT Sched (2 experiments)"
echo ""
echo "Total: 9 experiments"
echo "Estimated total time: 30-39 hours on H100"
echo "================================================================================"
echo ""

# Configuration
DIFFICULTY_FILE="${1:-../gen_diff_gsm8k/test-1pl/best_parameters.json}"
SEED=42

# Create results directory
RESULTS_DIR="./ablation_study_results_$(date +%Y%m%d_%H%M)"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"
echo "IRT Difficulty file: $DIFFICULTY_FILE"
echo "Random seed: $SEED"
echo ""

# Check if difficulty file exists
if [ ! -f "$DIFFICULTY_FILE" ]; then
    echo "ERROR: IRT difficulty file not found at: $DIFFICULTY_FILE"
    echo "Please provide the correct path as first argument:"
    echo "  $0 /path/to/difficulty.json"
    exit 1
fi

# Function to log experiment start/end
log_experiment() {
    local experiment_name=$1
    local status=$2
    local timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] $status: $experiment_name" >> "$RESULTS_DIR/experiment_log.txt"
    echo "[$timestamp] $status: $experiment_name"
}

# ============================================================================
# Baseline 1: Heuristic CL (4 experiments)
# ============================================================================

echo ""
echo "================================================================================"
echo "BASELINE 1: HEURISTIC CL"
echo "================================================================================"
echo ""

log_experiment "Baseline 1 - Heuristic CL" "STARTING"

python gsm8k_qwen2_5_7b_heuristic_cl.py \
    2>&1 | tee "$RESULTS_DIR/baseline1_heuristic_cl.log"

if [ $? -eq 0 ]; then
    log_experiment "Baseline 1 - Heuristic CL" "COMPLETED"
else
    log_experiment "Baseline 1 - Heuristic CL" "FAILED"
    echo "ERROR: Baseline 1 failed. Check log at: $RESULTS_DIR/baseline1_heuristic_cl.log"
fi

# ============================================================================
# Baseline 2: PUDF (1 experiment)
# ============================================================================

echo ""
echo "================================================================================"
echo "BASELINE 2: PUDF"
echo "================================================================================"
echo ""

log_experiment "Baseline 2 - PUDF" "STARTING"

python gsm8k_qwen2_5_7b_pudf.py \
    --difficulty-file "$DIFFICULTY_FILE" \
    --output-dir "$RESULTS_DIR/baseline2_pudf" \
    --seed $SEED \
    2>&1 | tee "$RESULTS_DIR/baseline2_pudf.log"

if [ $? -eq 0 ]; then
    log_experiment "Baseline 2 - PUDF" "COMPLETED"
else
    log_experiment "Baseline 2 - PUDF" "FAILED"
    echo "ERROR: Baseline 2 failed. Check log at: $RESULTS_DIR/baseline2_pudf.log"
fi

# ============================================================================
# Ablation 1: IRT Difficulty + Heuristic Scheduler (2 experiments)
# ============================================================================

echo ""
echo "================================================================================"
echo "ABLATION 1: IRT DIFFICULTY + HEURISTIC SCHEDULER"
echo "================================================================================"
echo ""

log_experiment "Ablation 1 - IRT + Heuristic" "STARTING"

python gsm8k_ablation_irt_diff_heuristic_sched.py \
    --difficulty-file "$DIFFICULTY_FILE" \
    --output-dir "$RESULTS_DIR/ablation1_irt_heuristic" \
    --seed $SEED \
    2>&1 | tee "$RESULTS_DIR/ablation1_irt_heuristic.log"

if [ $? -eq 0 ]; then
    log_experiment "Ablation 1 - IRT + Heuristic" "COMPLETED"
else
    log_experiment "Ablation 1 - IRT + Heuristic" "FAILED"
    echo "ERROR: Ablation 1 failed. Check log at: $RESULTS_DIR/ablation1_irt_heuristic.log"
fi

# ============================================================================
# Ablation 2a: Sentence Length + IRT Scheduler (1 experiment)
# ============================================================================

echo ""
echo "================================================================================"
echo "ABLATION 2a: SENTENCE LENGTH + IRT SCHEDULER"
echo "================================================================================"
echo ""

log_experiment "Ablation 2a - Sentence Length + IRT" "STARTING"

python gsm8k_ablation_heuristic_diff_irt_sched.py \
    --difficulty-measure sentence_length \
    --pudf-epochs 8 \
    --lr 1e-4 \
    --train-batch-size 16 \
    --eval-batch-size 32 \
    --theta-batch-size 32 \
    --initial-theta 0.0 \
    --lower-offset -1000.0 \
    --upper-offset 0.0 \
    --min-samples 100 \
    --output-dir "$RESULTS_DIR/ablation2a_sentence_irt" \
    --seed $SEED \
    2>&1 | tee "$RESULTS_DIR/ablation2a_sentence_irt.log"

if [ $? -eq 0 ]; then
    log_experiment "Ablation 2a - Sentence Length + IRT" "COMPLETED"
else
    log_experiment "Ablation 2a - Sentence Length + IRT" "FAILED"
    echo "ERROR: Ablation 2a failed. Check log at: $RESULTS_DIR/ablation2a_sentence_irt.log"
fi

# ============================================================================
# Ablation 2b: Word Rarity + IRT Scheduler (1 experiment)
# ============================================================================

echo ""
echo "================================================================================"
echo "ABLATION 2b: WORD RARITY + IRT SCHEDULER"
echo "================================================================================"
echo ""

log_experiment "Ablation 2b - Word Rarity + IRT" "STARTING"

python gsm8k_ablation_heuristic_diff_irt_sched.py \
    --difficulty-measure word_rarity \
    --pudf-epochs 8 \
    --lr 1e-4 \
    --train-batch-size 16 \
    --eval-batch-size 32 \
    --theta-batch-size 32 \
    --initial-theta 0.0 \
    --lower-offset -1000.0 \
    --upper-offset 0.0 \
    --min-samples 100 \
    --output-dir "$RESULTS_DIR/ablation2b_wordrarity_irt" \
    --seed $SEED \
    2>&1 | tee "$RESULTS_DIR/ablation2b_wordrarity_irt.log"

if [ $? -eq 0 ]; then
    log_experiment "Ablation 2b - Word Rarity + IRT" "COMPLETED"
else
    log_experiment "Ablation 2b - Word Rarity + IRT" "FAILED"
    echo "ERROR: Ablation 2b failed. Check log at: $RESULTS_DIR/ablation2b_wordrarity_irt.log"
fi

# ============================================================================
# Summary
# ============================================================================

echo ""
echo "================================================================================"
echo "ALL EXPERIMENTS COMPLETED"
echo "================================================================================"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Experiment log:"
cat "$RESULTS_DIR/experiment_log.txt"
echo ""

# Collect all results
echo "Collecting results..."
python3 << EOF
import json
import os
import glob

results_dir = "$RESULTS_DIR"
all_results = []

# Find all evaluation result files
result_files = []
result_files.extend(glob.glob(f"{results_dir}/**/evaluation_results.json", recursive=True))
result_files.extend(glob.glob(f"{results_dir}/**/test_evaluation_results.json", recursive=True))

for result_file in result_files:
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract key information
        difficulty = data.get('curriculum_config', {}).get('difficulty_measure', 
                              data.get('ablation_type', 'unknown'))
        scheduler = data.get('curriculum_config', {}).get('scheduler', 'unknown')
        accuracy = data.get('metrics', {}).get('accuracy', 0.0)
        
        all_results.append({
            'file': result_file,
            'difficulty': difficulty,
            'scheduler': scheduler,
            'accuracy': accuracy
        })
    except Exception as e:
        print(f"Error reading {result_file}: {e}")

# Sort by accuracy
all_results.sort(key=lambda x: x['accuracy'], reverse=True)

# Print results table
print("\n" + "=" * 80)
print("RESULTS SUMMARY (sorted by accuracy)")
print("=" * 80)
print(f"{'Rank':<6}{'Difficulty':<30}{'Scheduler':<20}{'Accuracy':<10}")
print("-" * 80)

for i, result in enumerate(all_results, 1):
    print(f"{i:<6}{result['difficulty']:<30}{result['scheduler']:<20}{result['accuracy']:.4f}")

print("\n")

# Save summary
summary_file = os.path.join(results_dir, "all_results_summary.json")
with open(summary_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"Detailed summary saved to: {summary_file}")
print("=" * 80)
EOF

echo ""
echo "All experiments completed! Check the results directory for details."
echo ""
echo "To analyze results further, see:"
echo "  - $RESULTS_DIR/experiment_log.txt (experiment timeline)"
echo "  - $RESULTS_DIR/all_results_summary.json (accuracy comparison)"
echo "  - Individual experiment directories for detailed outputs"
echo ""
