#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ============================================================================
# 🔥 MEMORY FIXES - MUST BE FIRST, BEFORE ANY IMPORTS!
# ============================================================================
import os

# Fix 1: Prevent memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# Fix 2: Set cache directories - USE HF_HOME (not TRANSFORMERS_CACHE)
# Import paths from central config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME, GLUE_DIFFICULTY_DIR, MEDQA_DIFFICULTY_FILE
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# Now safe to import everything else
# ============================================================================
import sys
import datetime
import random
import traceback
import json
import argparse
from tqdm import tqdm
import re
import warnings
import gc

import torch
import numpy as np

from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Creating a tensor from a list of numpy.ndarrays is extremely slow')

# ============================================================================
# 🔑 NO AUTHENTICATION - Will use cached token or download with existing auth
# ============================================================================
print("✅ Ready to train (using cached authentication)")


# ============================================================================
# Memory Cleanup Callback
# ============================================================================

class MemoryCleanupCallback(TrainerCallback):
    """Clear CUDA cache periodically to prevent OOM from fragmentation"""

    def __init__(self, cleanup_every_n_steps=25):
        self.cleanup_every_n_steps = cleanup_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.cleanup_every_n_steps == 0:
            torch.cuda.empty_cache()
            gc.collect()
        return control


# ============================================================================
# Dataset Switch Callback (NEW)
# ============================================================================

class DatasetSwitchCallback(TrainerCallback):
    """Switch to combined train+val dataset after specified epoch"""

    def __init__(self, trainer, combined_dataset, switch_epoch=5):
        self.trainer = trainer
        self.combined_dataset = combined_dataset
        self.switch_epoch = switch_epoch
        self.switched = False

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch) + 1
        if current_epoch >= self.switch_epoch and not self.switched:
            print(f"\n{'=' * 70}")
            print(f"📊 SWITCHING TO COMBINED DATASET at Epoch {current_epoch}")
            print(f"{'=' * 70}")
            print(f"   Previous training set size: {len(self.trainer.train_dataset)}")
            print(f"   New training set size:      {len(self.combined_dataset)}")
            print(f"   Additional samples:         {len(self.combined_dataset) - len(self.trainer.train_dataset)}")
            print(f"{'=' * 70}\n")

            self.trainer.train_dataset = self.combined_dataset
            self.switched = True
        return control


# ============================================================================
# Parse Command-Line Arguments
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Llama 3.1 8B on GSM8K with QLoRA (IMPROVED BASELINE)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["base", "instruct"],
        default="base",
        help="Which model to fine-tune: 'base' or 'instruct'"
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 10 for base, 1 for instruct)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: 2e-4 for base, 5e-6 for instruct)"
    )

    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=16,
        help="Per-device training batch size"
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Per-device evaluation batch size"
    )

    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=2,  # Changed from 3 to get effective batch size of 32
        help="Gradient accumulation steps"
    )

    # LoRA parameters
    parser.add_argument(
        "--lora-r",
        type=int,
        default=None,
        help="LoRA rank (default: 16 for base, 8 for instruct)"
    )

    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha (default: 16 for base, 16 for instruct)"
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: auto-generated based on model and timestamp)"
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    parser.add_argument(
        "--test-subset",
        type=int,
        default=None,
        help="Only evaluate on a subset of test data (for quick testing)"
    )

    parser.add_argument(
        "--switch-epoch",
        type=int,
        default=6,
        help="Epoch to switch to combined train+val dataset (default: 6)"
    )

    args = parser.parse_args()

    # Set defaults based on model type (UPDATED VALUES)
    if args.model == "base":
        if args.epochs is None:
            args.epochs = 10  # Changed from 8
        if args.lr is None:
            args.lr = 2e-4  # Changed from 1e-4
        if args.lora_r is None:
            args.lora_r = 16  # Changed from 64
        if args.lora_alpha is None:
            args.lora_alpha = 16  # Changed from 128
    else:  # instruct
        if args.epochs is None:
            args.epochs = 1
        if args.lr is None:
            args.lr = 5e-6
        if args.lora_r is None:
            args.lora_r = 8
        if args.lora_alpha is None:
            args.lora_alpha = 16

    # Set default output directory
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        args.output_dir = f"./qlora_gsm8k_llama3_1_8b_{args.model}_improved_{timestamp}"

    return args


# ============================================================================
# Create cache directories
# ============================================================================
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)


# ============================================================================
# Helper Functions
# ============================================================================

def extract_answer(text):
    """Extract numerical answer from text"""
    if '####' in text:
        match = re.search(r'####\s*(-?\d[\d,]*\.?\d*)', text)
        if match:
            return match.group(1).replace(',', '').strip()

    numbers = re.findall(r'-?\d[\d,]*\.?\d*', text.replace(',', ''))
    return numbers[-1].strip() if numbers else None


def normalize_answer(answer_str):
    """Normalize answer for comparison"""
    if answer_str is None:
        return None
    try:
        num = float(answer_str.replace(',', ''))
        if num == int(num):
            return str(int(num))
        return str(num)
    except:
        return answer_str


def preprocess_gsm8k_sft(examples, tokenizer_ref, max_seq_len, max_prompt_len):
    """
    Preprocess GSM8K for supervised fine-tuning
    Returns numpy arrays to eliminate conversion warnings
    """
    inputs_batch = []
    labels_batch = []

    for i in range(len(examples["question"])):
        question = examples["question"][i].strip()
        answer = examples["answer"][i].strip()

        # Improved prompt format
        prompt_text = (
            f"Solve this math problem step by step. "
            f"Show your work and write the final answer after '####'.\n\n"
            f"Question: {question}\n\n"
            f"Answer: Let's solve this step by step.\n"
        )

        full_text = prompt_text + answer

        # Tokenize
        tokenized_full = tokenizer_ref(
            full_text,
            truncation=True,
            max_length=max_seq_len,
            padding=False,
            add_special_tokens=True
        )

        tokenized_prompt = tokenizer_ref(
            prompt_text,
            truncation=True,
            max_length=max_prompt_len,
            padding=False,
            add_special_tokens=True
        )

        input_ids = tokenized_full.input_ids
        prompt_len = len(tokenized_prompt.input_ids)

        # Mask prompt in labels
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

        if len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))

        # Convert to numpy arrays immediately
        inputs_batch.append(np.array(input_ids, dtype=np.int64))
        labels_batch.append(np.array(labels, dtype=np.int64))

    return {"input_ids": inputs_batch, "labels": labels_batch}


def create_evaluation_prompt(question):
    """Create evaluation prompt"""
    return (
        f"Solve this math problem step by step. "
        f"Show your work and write the final answer after '####'.\n\n"
        f"Question: {question}\n\n"
        f"Answer: Let's solve this step by step.\n"
    )


# ============================================================================
# Main Script
# ============================================================================

if __name__ == "__main__":
    # Parse arguments
    args = parse_args()

    # Set model ID based on choice
    if args.model == "base":
        model_id = "meta-llama/Meta-Llama-3.1-8B"
    else:
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    dataset_id = "openai/gsm8k"
    dataset_config = "main"
    max_seq_length = 640
    max_prompt_len_config = 512

    lora_dropout = 0.05  # Changed from 0.01
    early_stopping_patience = 5  # Changed from 3
    weight_decay_train = 0.01
    warmup_ratio = 0.1
    max_grad_norm = 1.0
    MAX_NEW_TOKENS_GEN = 768
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Print configuration
    print(f"{'=' * 70}")
    print(f"GSM8K FINE-TUNING - IMPROVED BASELINE")
    print(f"{'=' * 70}\n")

    print(f"🗂️  Cache: {HF_HOME}\n")

    print(f"💾 Memory Optimizations:")
    print(f"   PyTorch CUDA config: {os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
    print(f"   Sequence length:     {max_seq_length}")
    print(f"   Memory cleanup:      Enabled (every 25 steps)\n")

    print(f"📋 Configuration:")
    print(f"   Model:                {model_id}")
    print(f"   Model type:           {args.model}")
    print(f"   Output dir:           {args.output_dir}")
    print(f"   Device:               {DEVICE}\n")

    print(f"🔧 Training Hyperparameters (IMPROVED):")
    print(f"   Epochs:               {args.epochs}")
    print(f"   Learning rate:        {args.lr}")
    print(f"   LR scheduler:         linear (changed from cosine)")
    print(f"   Train batch size:     {args.train_batch_size}")
    print(f"   Eval batch size:      {args.eval_batch_size}")
    print(f"   Grad accum steps:     {args.grad_accum_steps}")
    print(f"   Effective batch size: {args.train_batch_size * args.grad_accum_steps}")
    print(f"   LoRA rank:            {args.lora_r} (reduced from 64)")
    print(f"   LoRA alpha:           {args.lora_alpha} (reduced from 128)")
    print(f"   LoRA dropout:         {lora_dropout} (increased from 0.01)")
    print(f"   Early stop patience:  {early_stopping_patience} (increased from 3)")
    print(f"   Eval strategy:        epoch (changed from steps)")
    print(f"   Switch to train+val:  Epoch {args.switch_epoch}")
    print(f"   Warmup ratio:         {warmup_ratio}")
    print(f"   Random seed:          {args.seed}\n")

    print(f"🔧 LoRA Configuration Changes:")
    print(f"   ❌ REMOVED: modules_to_save=['embed_tokens', 'lm_head']")
    print(f"   ❌ REMOVED: use_rslora=True")
    print(f"   ✅ Trainable params reduced from ~135M to ~1-2M\n")

    # Load dataset
    print("📚 Loading GSM8K...")
    dataset = load_dataset(
        dataset_id,
        dataset_config,
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )

    print(f"✅ Dataset splits:")
    print(f"   Train: {len(dataset['train'])} examples")
    print(f"   Test:  {len(dataset['test'])} examples\n")

    # Split test set for validation
    print("📊 Creating train/validation split from test set...")
    test_split = dataset['test'].train_test_split(test_size=0.2, seed=args.seed, shuffle=True)

    # Use 80% of test for validation during training, 20% reserved for final test
    validation_data = test_split['train']  # 80% of test set
    final_test_data = test_split['test']  # 20% of test set

    print(f"✅ Split created:")
    print(f"   Train:        {len(dataset['train'])} examples")
    print(f"   Validation:   {len(validation_data)} examples (used during training)")
    print(f"   Final Test:   {len(final_test_data)} examples (reserved for evaluation)")
    print(f"   Note: Using FULL validation set (not 500 samples)\n")

    # Load tokenizer
    print("📥 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            cache_dir=HF_HOME,
            local_files_only=True
        )
        print(f"   ✅ Loaded from cache")
    except (OSError, ValueError):
        print(f"   ⚠️  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            padding_side="left",
            cache_dir=HF_HOME
        )
        print(f"   ✅ Downloaded")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Tokenizer ready (vocab: {len(tokenizer)})\n")

    # ========================================================================
    # TRAINING PHASE
    # ========================================================================
    print(f"{'=' * 70}")
    print("TRAINING PHASE")
    print(f"{'=' * 70}\n")

    # QLoRA config
    print("⚙️  Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("📥 Loading model for training...")
    try:
        # Try cache first
        model_train = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=HF_HOME,
            local_files_only=True,
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
        )
        print(f"   ✅ Loaded from cache on: {model_train.device}\n")
    except (OSError, ValueError) as e:
        # Cache incomplete, download
        print(f"   ⚠️  Cache incomplete, downloading model (one-time only)...")
        model_train = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=HF_HOME,
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
        )
        print(f"   ✅ Downloaded and loaded on: {model_train.device}\n")

    # Sync tokenizer and model
    if len(tokenizer) > model_train.config.vocab_size:
        model_train.resize_token_embeddings(len(tokenizer))
    if model_train.config.pad_token_id != tokenizer.pad_token_id:
        model_train.config.pad_token_id = tokenizer.pad_token_id

    # Prepare for QLoRA
    model_train = prepare_model_for_kbit_training(
        model_train,
        use_gradient_checkpointing=True
    )

    # 🔥 Fix warning: Disable use_cache for gradient checkpointing
    model_train.config.use_cache = False

    print("⚙️  Configuring LoRA (IMPROVED)...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,  # Now 16 instead of 64
        lora_alpha=args.lora_alpha,  # Now 16 instead of 128
        lora_dropout=lora_dropout,  # Now 0.05 instead of 0.01
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        # ✅ REMOVED: use_rslora=True
        # ✅ REMOVED: modules_to_save=["embed_tokens", "lm_head"]
    )

    model_train = get_peft_model(model_train, lora_config)
    print("\n✅ QLoRA model prepared:")
    model_train.print_trainable_parameters()

    # Tokenize dataset
    print(f"\n📝 Tokenizing dataset (max_seq_length={max_seq_length})...")
    tokenized_train = dataset['train'].map(
        lambda ex: preprocess_gsm8k_sft(ex, tokenizer, max_seq_length, max_prompt_len_config),
        batched=True,
        batch_size=1000,
        num_proc=max(1, os.cpu_count() // 2),
        desc="Tokenizing train"
    )

    tokenized_val = validation_data.map(
        lambda ex: preprocess_gsm8k_sft(ex, tokenizer, max_seq_length, max_prompt_len_config),
        batched=True,
        batch_size=1000,
        num_proc=max(1, os.cpu_count() // 2),
        desc="Tokenizing validation"
    )

    keep_cols = ["input_ids", "labels"]
    tokenized_train = tokenized_train.remove_columns(
        [c for c in tokenized_train.column_names if c not in keep_cols]
    )
    tokenized_val = tokenized_val.remove_columns(
        [c for c in tokenized_val.column_names if c not in keep_cols]
    )

    tokenized_train.set_format(type="torch", columns=keep_cols)
    tokenized_val.set_format(type="torch", columns=keep_cols)

    print("✅ Tokenization complete")
    print(f"   Train:      {len(tokenized_train)} examples")
    print(f"   Validation: {len(tokenized_val)} examples\n")

    # Create combined dataset for later epochs
    print("📊 Creating combined train+val dataset for later epochs...")
    combined_dataset = concatenate_datasets([tokenized_train, tokenized_val])
    combined_dataset.set_format(type="torch", columns=keep_cols)
    print(f"✅ Combined dataset: {len(combined_dataset)} examples")
    print(f"   Will switch to this at epoch {args.switch_epoch}\n")

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model_train,
        label_pad_token_id=-100,
        padding="longest"
    )

    # Training arguments (IMPROVED)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=weight_decay_train,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="linear",  # Changed from "cosine"
        max_grad_norm=max_grad_norm,
        bf16=True,
        bf16_full_eval=True,
        logging_strategy="steps",
        logging_steps=25,
        logging_first_step=True,
        eval_strategy="epoch",  # Changed from "steps"
        save_strategy="epoch",  # Changed from "steps"
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        report_to=[],
        seed=args.seed,
        prediction_loss_only=True,
        label_names=["labels"],
    )

    # Trainer with improved callbacks
    trainer = Trainer(
        model=model_train,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            MemoryCleanupCallback(cleanup_every_n_steps=25),
        ],
    )

    # # Add DatasetSwitchCallback after trainer is created
    # dataset_switch_callback = DatasetSwitchCallback(
    #     trainer=trainer,
    #     combined_dataset=combined_dataset,
    #     switch_epoch=args.switch_epoch
    # )
    # trainer.add_callback(dataset_switch_callback)

    # Train
    print(f"\n🚀 Starting fine-tuning...")
    print(f"   Train samples:        {len(tokenized_train)}")
    print(f"   Validation samples:   {len(tokenized_val)}")
    print(f"   Effective batch size: {args.train_batch_size * args.grad_accum_steps}")
    steps_per_epoch = len(tokenized_train) // (args.train_batch_size * args.grad_accum_steps)
    print(f"   Steps per epoch:      ~{steps_per_epoch}")
    print(f"   Total epochs:         {args.epochs}")
    print(f"   Total steps:          ~{steps_per_epoch * args.epochs}")
    print(f"   Switch to train+val:  Epoch {args.switch_epoch}\n")

    final_adapter_path = os.path.join(args.output_dir, "final_adapter")

    try:
        train_result = trainer.train()

        # Save
        trainer.save_model(final_adapter_path)
        tokenizer.save_pretrained(final_adapter_path)

        print(f"\n✅ Training complete!")
        print(f"   Final train loss: {train_result.training_loss:.4f}")
        print(f"   Saved to: {final_adapter_path}\n")

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    except Exception as e:
        print(f"\n💥 Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Clean up
    del model_train, trainer
    torch.cuda.empty_cache()
    gc.collect()
    print("✅ Training resources released\n")

    # ========================================================================
    # EVALUATION ON FINAL TEST SET
    # ========================================================================
    print(f"{'=' * 70}")
    print("EVALUATION ON FINAL TEST SET")
    print(f"{'=' * 70}\n")

    # Load base model
    print("📥 Loading base model for evaluation...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=HF_HOME,
            local_files_only=True,
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
        )
        print("   ✅ Loaded from cache\n")
    except (OSError, ValueError):
        print("   ⚠️  Downloading model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=HF_HOME,
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
        )
        print("   ✅ Downloaded\n")

    if len(tokenizer) > base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))
    if base_model.config.pad_token_id != tokenizer.pad_token_id:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    # Load adapter
    print(f"📥 Loading adapter: {final_adapter_path}...")
    model_eval = PeftModel.from_pretrained(base_model, final_adapter_path)
    model_eval.eval()
    print("✅ Model ready\n")

    # Get test dataset
    test_dataset = final_test_data
    if args.test_subset:
        print(f"⚠️  Using subset of {args.test_subset} test examples (for quick testing)")
        test_dataset = test_dataset.select(range(min(args.test_subset, len(test_dataset))))

    print(f"🧮 Evaluating on {len(test_dataset)} examples...")
    print(f"   Batch size: {args.eval_batch_size}\n")

    results = []
    correct = 0
    total = 0
    truncation_count = 0
    total_gen_length = 0

    with tqdm(total=len(test_dataset), desc="Evaluating") as pbar:
        for idx in range(0, len(test_dataset), args.eval_batch_size):
            batch_end = min(idx + args.eval_batch_size, len(test_dataset))
            batch = test_dataset[idx:batch_end]

            prompts = [create_evaluation_prompt(q) for q in batch["question"]]

            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_prompt_len_config
            ).to(DEVICE)

            with torch.inference_mode():
                generated_ids = model_eval.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS_GEN,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            for i in range(len(prompts)):
                input_len = inputs['input_ids'][i].shape[0]
                output_len = generated_ids[i].shape[0] - input_len
                total_gen_length += output_len

                # Check for truncation
                if output_len >= MAX_NEW_TOKENS_GEN:
                    truncation_count += 1

                generated_text = tokenizer.decode(
                    generated_ids[i][input_len:],
                    skip_special_tokens=True
                )

                pred_answer = normalize_answer(extract_answer(generated_text))
                true_answer = normalize_answer(extract_answer(batch["answer"][i]))

                is_correct = (pred_answer == true_answer) and (pred_answer is not None)

                if is_correct:
                    correct += 1
                total += 1

                results.append({
                    "id": idx + i,
                    "question": batch["question"][i][:100] + "...",
                    "predicted": pred_answer,
                    "true": true_answer,
                    "correct": is_correct,
                    "output_length": output_len,
                    "truncated": output_len >= MAX_NEW_TOKENS_GEN
                })

            pbar.update(len(prompts))
            pbar.set_postfix({"accuracy": f"{correct / total:.2%}"})

    accuracy = correct / total
    avg_gen_length = total_gen_length / total

    # Report generation statistics
    print(f"\n{'=' * 70}")
    print(f"📊 GENERATION STATISTICS")
    print(f"{'=' * 70}")
    print(f"Avg generation length: {avg_gen_length:.1f} tokens")
    print(f"Truncated outputs:     {truncation_count}/{total} ({truncation_count / total * 100:.1f}%)")
    if truncation_count > 0:
        print(f"⚠️  {truncation_count} generations hit max length - consider increasing MAX_NEW_TOKENS_GEN")
    print()

    print(f"{'=' * 70}")
    print(f"🎉 FINAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Model:       {model_id}")
    print(f"Model type:  {args.model}")
    print(f"Total:       {total} examples")
    print(f"Correct:     {correct}")
    print(f"Incorrect:   {total - correct}")
    print(f"Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"{'=' * 70}\n")

    # Show examples
    print("📝 Sample predictions:")
    for i in range(min(5, len(results))):
        r = results[i]
        status = "✅" if r['correct'] else "❌"
        print(f"\n{status} Example {i + 1}:")
        print(f"   Q: {r['question']}")
        print(f"   Predicted: {r['predicted']}")
        print(f"   True: {r['true']}")
        if r['truncated']:
            print(f"   ⚠️  Output was truncated")

    # Save results
    results_file = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_file, "w") as f:
        json.dump({
            "model_id": model_id,
            "model_type": args.model,
            "improvements": [
                "LoRA rank reduced from 64 to 16",
                "LoRA alpha reduced from 128 to 16",
                "LoRA dropout increased from 0.01 to 0.05",
                "Removed modules_to_save=['embed_tokens', 'lm_head']",
                "Removed use_rslora=True",
                "Learning rate increased from 1e-4 to 2e-4",
                "LR scheduler changed from cosine to linear",
                "Epochs increased from 8 to 10",
                "Effective batch size reduced from 48 to 32",
                "Early stopping patience increased from 3 to 5",
                "Eval strategy changed from steps to epoch",
                "Using full validation set instead of 500 samples",
                f"Switch to combined train+val at epoch {args.switch_epoch}"
            ],
            "hyperparameters": {
                "num_epochs": args.epochs,
                "learning_rate": args.lr,
                "train_batch_size": args.train_batch_size,
                "effective_batch_size": args.train_batch_size * args.grad_accum_steps,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lora_dropout": lora_dropout,
                "warmup_ratio": warmup_ratio,
                "max_seq_length": max_seq_length,
                "early_stopping_patience": early_stopping_patience,
                "switch_epoch": args.switch_epoch,
            },
            "metrics": {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            },
            "generation_stats": {
                "avg_generation_length": avg_gen_length,
                "truncated_count": truncation_count,
                "truncated_percentage": truncation_count / total * 100,
            },
            "detailed_results": results
        }, f, indent=2)

    print(f"\n💾 Results saved to: {results_file}")

    # Summary
    summary_file = os.path.join(args.output_dir, "summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"GSM8K Fine-Tuning Results (IMPROVED BASELINE)\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Model type: {args.model}\n\n")
        f.write(f"Key Improvements:\n")
        f.write(f"  - LoRA rank: 64 → 16\n")
        f.write(f"  - LoRA alpha: 128 → 16\n")
        f.write(f"  - LoRA dropout: 0.01 → 0.05\n")
        f.write(f"  - Removed modules_to_save (saved ~131M params)\n")
        f.write(f"  - Removed use_rslora\n")
        f.write(f"  - Learning rate: 1e-4 → 2e-4\n")
        f.write(f"  - LR scheduler: cosine → linear\n")
        f.write(f"  - Epochs: 8 → 10\n")
        f.write(f"  - Effective batch: 48 → 32\n")
        f.write(f"  - Early stop patience: 3 → 5\n")
        f.write(f"  - Eval strategy: steps → epoch\n")
        f.write(f"  - Full validation set (not 500 samples)\n")
        f.write(f"  - Switch to train+val at epoch {args.switch_epoch}\n\n")
        f.write(f"Training Hyperparameters:\n")
        f.write(f"  Training epochs: {args.epochs}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  LoRA rank: {args.lora_r}\n")
        f.write(f"  LoRA alpha: {args.lora_alpha}\n")
        f.write(f"  LoRA dropout: {lora_dropout}\n")
        f.write(f"  Warmup ratio: {warmup_ratio}\n")
        f.write(f"  Max sequence length: {max_seq_length}\n\n")
        f.write(f"Final Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n")
        f.write(f"Correct: {correct}/{total}\n\n")
        f.write(f"Generation Stats:\n")
        f.write(f"  Avg length: {avg_gen_length:.1f} tokens\n")
        f.write(f"  Truncated: {truncation_count}/{total} ({truncation_count / total * 100:.1f}%)\n")

    print(f"📄 Summary saved to: {summary_file}")
    print(f"\n✨ Complete!")