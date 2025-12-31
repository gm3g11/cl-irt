#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GSM8K Ablation Study: IRT Difficulty + Heuristic Scheduler
Uses pre-calculated IRT difficulty scores with linear/root curriculum schedulers
"""

# ============================================================================
# 🔥 MEMORY FIXES - MUST BE FIRST, BEFORE ANY IMPORTS!
# ============================================================================
import os

# Fix 1: Prevent memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

# Fix 2: Set cache directories
# Import paths from central config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME, GLUE_DIFFICULTY_DIR, MEDQA_DIFFICULTY_FILE
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# Imports
# ============================================================================
import sys
import datetime
import random
import traceback
import json
import re
import warnings
import gc
import time
import argparse
from tqdm import tqdm
from typing import Iterator, Dict, List, Any, Optional

import torch
import numpy as np

from datasets import load_dataset, Dataset
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
from torch.utils.data import DataLoader, Sampler
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Creating a tensor from a list of numpy.ndarrays is extremely slow')

print("✅ Imports ready")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model and Dataset
model_id = "Qwen/Qwen2.5-7B"
dataset_id = "openai/gsm8k"
dataset_config = "main"

# Sequence lengths
max_seq_length = 640
max_prompt_len_config = 512

# Training hyperparameters
per_device_train_bs = 32
per_device_eval_bs = 96
grad_accum_steps = 3
num_train_epochs = 7
early_stopping_patience = 3
learning_rate = 2e-4
weight_decay_train = 0.01
warmup_ratio = 0.1
max_grad_norm = 1.0

# LoRA parameters
lora_r = 64
lora_alpha = 128
lora_dropout = 0.01

# Evaluation settings
MAX_NEW_TOKENS_GEN = 512
eval_samples_during_training = 500

# Random seed
random_seed = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# HEURISTIC CURRICULUM LEARNING CONFIGURATION
# ============================================================================

# Curriculum strategies (same as heuristic CL)
SCHEDULERS = ['linear', 'root']
ORDERING = 'easiest'  # Start with easiest examples

# Heuristic parameters (same as heuristic CL)
COMPETENCY_PARAM = 5  # Number of epochs to reach full dataset
MIN_TRAIN_PERCENT = 0.05  # Start with at least 5% of data
C_INIT = 0.01  # Initial competency (1% of data)

# Output directory
BASE_OUTPUT_DIR = "./gsm8k_qwen2_5_7b_ablation_irt_diff_heuristic_sched"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# Cache directories
os.makedirs(HF_HOME, exist_ok=True)
os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)


# ============================================================================
# CUSTOM HEURISTIC SAMPLER (from heuristic CL code)
# ============================================================================

class HeuristicSampler(Sampler[int]):
    """Curriculum learning sampler that gradually increases dataset size"""
    
    def __init__(
        self,
        num_samples_total: int,
        batch_size: int,
        sorted_indices: List[int],
        heuristic_config: dict,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42
    ):
        if num_replicas <= 0 or rank < 0 or rank >= num_replicas:
            raise ValueError("Invalid num_replicas or rank.")
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size should be positive.")
            
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.batch_size = batch_size
        self._full_data_len = num_samples_total
        self._sorted_indices = sorted_indices
        self.heuristic_config = heuristic_config
        
        # Calculate minimum training length
        min_data_from_percent = int(self.heuristic_config['min_train_percent'] * self._full_data_len)
        min_data_from_batch = batch_size * num_replicas
        self._min_train_length = max(1, min_data_from_batch, min_data_from_percent)
        if self._min_train_length > self._full_data_len:
            self._min_train_length = self._full_data_len
        
        self.indices_for_epoch = []
        self.num_samples_epoch_replica = 0
        self.set_epoch(0)
    
    def _get_num_samples_for_epoch(self, epoch: int) -> int:
        """Calculate how many samples to use in this epoch"""
        scheduler_type = self.heuristic_config['scheduler']
        num_total = self._full_data_len
        competency_epoch_config = max(1, self.heuristic_config['competency_param'])
        c_init_val = self.heuristic_config['c_init']
        current_epoch_float = float(epoch)
        
        if scheduler_type == 'linear':
            if current_epoch_float < competency_epoch_config:
                epoch_competency = c_init_val + (1.0 - c_init_val) * (current_epoch_float / competency_epoch_config)
            else:
                epoch_competency = 1.0
        elif scheduler_type == 'root':
            if current_epoch_float < competency_epoch_config:
                epoch_competency = c_init_val + (1.0 - c_init_val) * np.sqrt(current_epoch_float / competency_epoch_config)
            else:
                epoch_competency = 1.0
        else:
            raise NotImplementedError(f"Scheduler '{scheduler_type}' unknown.")
        
        num_train = int(epoch_competency * num_total)
        num_train = max(self._min_train_length, num_train)
        num_train = min(num_train, num_total)
        return num_train
    
    def set_epoch(self, epoch: int) -> None:
        """Update sampler for new epoch"""
        self.epoch = epoch
        num_samples_for_epoch = self._get_num_samples_for_epoch(epoch)
        new_indices = self._sorted_indices[:num_samples_for_epoch]
        
        # Log epoch updates
        percent = (len(new_indices) / self._full_data_len * 100) if self._full_data_len > 0 else 0
        print(f"[Sampler] Epoch {epoch}: Using {len(new_indices)}/{self._full_data_len} samples ({percent:.1f}%)")
        
        self.indices_for_epoch = new_indices
        
        if self.num_replicas > 1:
            num_samples_this_epoch_total = len(self.indices_for_epoch)
            if num_samples_this_epoch_total % self.num_replicas != 0:
                self.indices_for_epoch = self.indices_for_epoch[:num_samples_this_epoch_total - (num_samples_this_epoch_total % self.num_replicas)]
            self.num_samples_epoch_replica = len(self.indices_for_epoch) // self.num_replicas
        else:
            self.num_samples_epoch_replica = len(self.indices_for_epoch)
    
    def __iter__(self) -> Iterator[int]:
        if not self.indices_for_epoch:
            return iter([])
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices_shuffled = [self.indices_for_epoch[i] for i in torch.randperm(len(self.indices_for_epoch), generator=g).tolist()]
        if self.num_replicas > 1:
            return iter(indices_shuffled[self.rank: len(indices_shuffled): self.num_replicas])
        else:
            return iter(indices_shuffled)
    
    def __len__(self) -> int:
        return self.num_samples_epoch_replica


# ============================================================================
# SAMPLER EPOCH CALLBACK
# ============================================================================

class SamplerEpochCallback(TrainerCallback):
    """Callback to update sampler's epoch at the start of each epoch"""
    
    def __init__(self, sampler):
        self.sampler = sampler
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Called at the beginning of each epoch"""
        if state.epoch is not None:
            epoch = int(state.epoch)
            self.sampler.set_epoch(epoch)
            print(f"\n{'='*60}")
            print(f"[Callback] Starting Epoch {epoch}")
            print(f"{'='*60}\n")
        return control


# ============================================================================
# CUSTOM TRAINER WITH HEURISTIC SAMPLER
# ============================================================================

class CustomHeuristicTrainer(Trainer):
    """Custom trainer that uses curriculum learning sampler"""
    
    def __init__(
        self,
        *args: Any,
        sorted_indices: Optional[List[int]] = None,
        heuristic_config: Optional[Dict[str, Any]] = None,
        num_samples_total: Optional[int] = None,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        if sorted_indices is None or heuristic_config is None or num_samples_total is None:
            raise ValueError("CustomHeuristicTrainer requires sorted_indices, heuristic_config, and num_samples_total.")
        self.sorted_indices = sorted_indices
        self.heuristic_config = heuristic_config
        self.num_samples_total = num_samples_total
        self.current_sampler = None
        print("CustomHeuristicTrainer initialized with curriculum learning.")
    
    def get_train_dataloader(self) -> DataLoader:
        """Override to use custom sampler"""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        heuristic_sampler = HeuristicSampler(
            num_samples_total=self.num_samples_total,
            batch_size=self._train_batch_size,
            sorted_indices=self.sorted_indices,
            heuristic_config=self.heuristic_config,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=self.args.seed
        )
        
        self.current_sampler = heuristic_sampler
        
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=heuristic_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_difficulties_from_file(difficulty_file_path, difficulty_key='diff'):
    """Load difficulty scores from JSON file (from PUDF code)"""
    print(f"Loading IRT difficulty scores from: {difficulty_file_path}")
    try:
        with open(difficulty_file_path, 'r') as f:
            data = json.load(f)
        
        if difficulty_key in data:
            scores = data[difficulty_key]
        elif isinstance(data, list):
            scores = data
        else:
            raise KeyError(f"Key '{difficulty_key}' not found and data is not a list.")
        
        return np.array(scores, dtype=float)
    except Exception as e:
        print(f"Error loading difficulty scores: {e}")
        traceback.print_exc()
        raise


def preprocess_gsm8k_sft(examples, tokenizer_ref, max_seq_len, max_prompt_len):
    """Preprocess GSM8K for supervised fine-tuning (Qwen-compatible)"""
    inputs_batch = []
    labels_batch = []

    for i in range(len(examples["question"])):
        question = examples["question"][i].strip()
        answer = examples["answer"][i].strip()

        prompt_text = (
            f"Solve this math problem step by step. "
            f"Show your work and write the final answer after '####'.\n\n"
            f"Question: {question}\n\n"
            f"Answer: Let's solve this step by step.\n"
        )

        full_text = prompt_text + answer

        tokenized_prompt = tokenizer_ref(
            prompt_text,
            truncation=False,
            padding=False,
            add_special_tokens=False
        )

        tokenized_full = tokenizer_ref(
            full_text,
            truncation=False,
            padding=False,
            add_special_tokens=False
        )

        prompt_ids = tokenized_prompt.input_ids
        full_ids = tokenized_full.input_ids

        input_ids = []
        if tokenizer_ref.bos_token_id is not None and getattr(tokenizer_ref, 'add_bos_token', True):
            input_ids.append(tokenizer_ref.bos_token_id)

        input_ids.extend(full_ids)

        if tokenizer_ref.eos_token_id is not None:
            input_ids.append(tokenizer_ref.eos_token_id)

        len_prompt_with_bos = len(prompt_ids)
        if tokenizer_ref.bos_token_id is not None and getattr(tokenizer_ref, 'add_bos_token', True):
            len_prompt_with_bos += 1

        labels = [-100] * len_prompt_with_bos + full_ids[len(prompt_ids):]

        if tokenizer_ref.eos_token_id is not None:
            labels.append(tokenizer_ref.eos_token_id)

        input_ids = input_ids[:max_seq_len]
        labels = labels[:max_seq_len]

        if len(labels) < len(input_ids):
            labels.extend([-100] * (len(input_ids) - len(labels)))

        inputs_batch.append(input_ids)
        labels_batch.append(labels)

    return {"input_ids": inputs_batch, "labels": labels_batch}


def create_evaluation_prompt(question):
    """Create evaluation prompt"""
    return (
        f"Solve this math problem step by step. "
        f"Show your work and write the final answer after '####'.\n\n"
        f"Question: {question}\n\n"
        f"Answer: Let's solve this step by step.\n"
    )


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


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def run_training_with_curriculum(
    scheduler: str,
    output_dir: str,
    global_tokenizer,
    dataset,
    difficulty_scores: np.ndarray
):
    """Run one experiment with IRT difficulty + heuristic scheduler"""
    print("\n" + "=" * 80)
    print(f"Starting Run: IRT Difficulty + {scheduler.upper()} Scheduler")
    print(f"Output Dir: {output_dir}")
    print("=" * 80 + "\n")
    os.makedirs(output_dir, exist_ok=True)
    
    model_train = None
    trainer = None
    base_model = None
    model_eval = None
    
    final_accuracy = 0.0
    run_status = "failed"
    
    try:
        # Step 1: Use pre-calculated IRT difficulties
        print(f"Using pre-calculated IRT difficulty scores...")
        print(f"   Min difficulty: {np.min(difficulty_scores):.4f}")
        print(f"   Max difficulty: {np.max(difficulty_scores):.4f}")
        print(f"   Mean difficulty: {np.mean(difficulty_scores):.4f}")
        
        # Step 2: Sort indices by IRT difficulty (easiest first)
        print(f"Sorting by IRT difficulty ({ORDERING})...")
        if ORDERING == 'easiest':
            sorted_indices = np.argsort(difficulty_scores).tolist()
        elif ORDERING == 'hardest':
            sorted_indices = np.argsort(difficulty_scores)[::-1].tolist()
        else:
            raise ValueError(f"Unknown ordering: {ORDERING}")
        
        print(f"Sorted {len(sorted_indices)} training examples")
        num_samples_total = len(dataset['train'])
        
        # Step 3: Tokenize datasets
        print("Tokenizing datasets...")
        tokenized_train = dataset['train'].map(
            lambda ex: preprocess_gsm8k_sft(ex, global_tokenizer, max_seq_length, max_prompt_len_config),
            batched=True,
            batch_size=1000,
            num_proc=max(1, os.cpu_count() // 2),
            desc="Tokenizing train"
        )
        
        eval_subset = dataset["test"].select(range(min(eval_samples_during_training, len(dataset["test"]))))
        tokenized_eval = eval_subset.map(
            lambda ex: preprocess_gsm8k_sft(ex, global_tokenizer, max_seq_length, max_prompt_len_config),
            batched=True,
            batch_size=1000,
            num_proc=max(1, os.cpu_count() // 2),
            desc="Tokenizing eval"
        )
        
        keep_cols = ["input_ids", "labels"]
        tokenized_train = tokenized_train.remove_columns(
            [c for c in tokenized_train.column_names if c not in keep_cols]
        )
        tokenized_eval = tokenized_eval.remove_columns(
            [c for c in tokenized_eval.column_names if c not in keep_cols]
        )
        tokenized_train.set_format(type="torch", columns=keep_cols)
        tokenized_eval.set_format(type="torch", columns=keep_cols)
        
        # Step 4: Load model with QLoRA
        print("Loading model with QLoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        
        model_train = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            cache_dir=HF_HOME,
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
        )
        
        if len(global_tokenizer) > model_train.config.vocab_size:
            model_train.resize_token_embeddings(len(global_tokenizer))
        if model_train.config.pad_token_id != global_tokenizer.pad_token_id:
            model_train.config.pad_token_id = global_tokenizer.pad_token_id
        
        model_train = prepare_model_for_kbit_training(model_train, use_gradient_checkpointing=True)
        model_train.config.use_cache = False
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_rslora=True,
            modules_to_save=["embed_tokens", "lm_head"],
        )
        
        model_train = get_peft_model(model_train, lora_config)
        print("\n✅ QLoRA model prepared:")
        model_train.print_trainable_parameters()
        
        # Step 5: Setup training arguments
        data_collator = DataCollatorForSeq2Seq(
            global_tokenizer,
            model=model_train,
            label_pad_token_id=-100,
            padding="longest"
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=per_device_train_bs,
            per_device_eval_batch_size=per_device_eval_bs,
            gradient_accumulation_steps=grad_accum_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay_train,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type="cosine",
            max_grad_norm=max_grad_norm,
            bf16=True,
            bf16_full_eval=True,
            logging_strategy="steps",
            logging_steps=25,
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            report_to=[],
            seed=random_seed,
            prediction_loss_only=True,
            label_names=["labels"],
        )
        
        # Curriculum config (HEURISTIC SCHEDULER)
        heuristic_config = {
            "scheduler": scheduler,
            "ordering": ORDERING,
            "competency_param": COMPETENCY_PARAM,
            "min_train_percent": MIN_TRAIN_PERCENT,
            "c_init": C_INIT,
        }
        
        # Step 6: Initialize custom trainer
        print("Initializing curriculum learning trainer...")
        trainer = CustomHeuristicTrainer(
            model=model_train,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            processing_class=global_tokenizer,
            data_collator=data_collator,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)
            ],
            sorted_indices=sorted_indices,
            heuristic_config=heuristic_config,
            num_samples_total=num_samples_total
        )
        
        # Add sampler epoch callback
        if hasattr(trainer, 'current_sampler') and trainer.current_sampler is not None:
            sampler_callback = SamplerEpochCallback(trainer.current_sampler)
            trainer.add_callback(sampler_callback)
            print("✅ Sampler epoch callback added\n")
        
        # Step 7: Train
        final_adapter_path = os.path.join(output_dir, "final_adapter")
        print(f"🚀 Starting training...")
        print(f"   Train samples:        {num_samples_total}")
        print(f"   Eval samples:         {len(tokenized_eval)}")
        print(f"   Difficulty:           IRT-based (pre-calculated)")
        print(f"   Scheduler:            {scheduler}")
        print(f"   Competency epochs:    {COMPETENCY_PARAM}")
        print(f"   Initial data %:       {C_INIT * 100:.1f}%")
        print(f"   Min data %:           {MIN_TRAIN_PERCENT * 100:.1f}%\n")
        
        train_result = trainer.train()
        
        # Save adapter
        print(f"\nSaving adapter to {final_adapter_path}...")
        try:
            trainer.save_model(final_adapter_path)
            global_tokenizer.save_pretrained(final_adapter_path)
            
            adapter_files = os.listdir(final_adapter_path)
            required_files = ['adapter_config.json', 'adapter_model.safetensors']
            
            if not all(f in adapter_files for f in required_files):
                raise ValueError(f"Incomplete adapter save. Found: {adapter_files}")
            
            safetensors_path = os.path.join(final_adapter_path, "adapter_model.safetensors")
            if os.path.exists(safetensors_path):
                size = os.path.getsize(safetensors_path)
                if size < 1000:
                    raise ValueError(f"adapter_model.safetensors is only {size} bytes!")
                print(f"✅ adapter_model.safetensors: {size:,} bytes")
            
            print(f"✅ Adapter saved successfully\n")
            
        except Exception as e_save:
            print(f"❌ Error saving adapter: {e_save}")
            raise
        
        print(f"✅ Training complete!")
        print(f"   Final train loss: {train_result.training_loss:.4f}\n")
        
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        
        del model_train, trainer
        model_train = None
        trainer = None
        torch.cuda.empty_cache()
        gc.collect()
        
        # Step 8: Evaluation
        print("\n" + "=" * 80)
        print("EVALUATION ON FULL TEST SET")
        print("=" * 80 + "\n")
        
        if not os.path.exists(final_adapter_path):
            raise FileNotFoundError(f"Adapter path not found: {final_adapter_path}")
        
        print("Loading model for evaluation...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=HF_HOME,
            attn_implementation="flash_attention_2" if torch.cuda.get_device_capability()[0] >= 8 else "sdpa",
        )
        
        if len(global_tokenizer) > base_model.config.vocab_size:
            base_model.resize_token_embeddings(len(global_tokenizer))
        if base_model.config.pad_token_id != global_tokenizer.pad_token_id:
            base_model.config.pad_token_id = global_tokenizer.pad_token_id
        
        model_eval = PeftModel.from_pretrained(base_model, final_adapter_path)
        model_eval.eval()
        print("✅ Model ready for evaluation\n")
        
        test_dataset = dataset["test"]
        print(f"🧮 Evaluating on {len(test_dataset)} examples...")
        
        results = []
        correct = 0
        total = 0
        
        with tqdm(total=len(test_dataset), desc="Evaluating") as pbar:
            for idx in range(0, len(test_dataset), per_device_eval_bs):
                batch_end = min(idx + per_device_eval_bs, len(test_dataset))
                batch = test_dataset[idx:batch_end]
                
                prompts = [create_evaluation_prompt(q) for q in batch["question"]]
                
                inputs = global_tokenizer(
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
                        pad_token_id=global_tokenizer.pad_token_id,
                        eos_token_id=global_tokenizer.eos_token_id
                    )
                
                for i in range(len(prompts)):
                    input_len = inputs['input_ids'][i].shape[0]
                    generated_text = global_tokenizer.decode(
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
                        "correct": is_correct
                    })
                
                pbar.update(len(prompts))
                pbar.set_postfix({"accuracy": f"{correct / total:.2%}"})
        
        final_accuracy = correct / total
        
        print(f"\n{'=' * 80}")
        print(f"🎉 FINAL RESULTS")
        print(f"{'=' * 80}")
        print(f"Difficulty:  IRT (pre-calculated)")
        print(f"Scheduler:   {scheduler}")
        print(f"Total:       {total} examples")
        print(f"Correct:     {correct}")
        print(f"Accuracy:    {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")
        print(f"{'=' * 80}\n")
        
        # Save results
        results_file = os.path.join(output_dir, "evaluation_results.json")
        with open(results_file, "w") as f:
            json.dump({
                "model_id": model_id,
                "ablation_type": "IRT_difficulty_heuristic_scheduler",
                "curriculum_config": {
                    "difficulty_measure": "IRT (pre-calculated)",
                    "scheduler": scheduler,
                    "ordering": ORDERING,
                    "competency_param": COMPETENCY_PARAM,
                    "min_train_percent": MIN_TRAIN_PERCENT,
                    "c_init": C_INIT,
                },
                "hyperparameters": {
                    "num_epochs": num_train_epochs,
                    "learning_rate": learning_rate,
                    "train_batch_size": per_device_train_bs,
                    "effective_batch_size": per_device_train_bs * grad_accum_steps,
                    "lora_r": lora_r,
                    "lora_alpha": lora_alpha,
                },
                "metrics": {
                    "accuracy": final_accuracy,
                    "correct": correct,
                    "total": total,
                },
                "sample_predictions": results[:10]
            }, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        
        run_status = "completed"
        
    except Exception as e:
        print(f"\n💥 Run failed: {e}")
        traceback.print_exc()
        run_status = f"error: {str(e)}"
    
    finally:
        print(f"Cleaning up resources...")
        
        for var_name, var_obj in [('model_train', model_train), ('trainer', trainer), 
                                    ('base_model', base_model), ('model_eval', model_eval)]:
            if var_obj is not None:
                try:
                    del var_obj
                except:
                    pass
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Cleanup complete. Status: {run_status}\n")
    
    return {
        "accuracy": final_accuracy,
        "status": run_status,
        "output_dir": output_dir
    }


# ============================================================================
# PARSE ARGUMENTS
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Ablation Study: IRT Difficulty + Heuristic Scheduler",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--difficulty-file",
        type=str,
        default="../gen_diff_gsm8k/test-1pl/best_parameters.json",
        help="Path to IRT difficulty scores JSON file"
    )
    
    parser.add_argument(
        "--difficulty-key",
        type=str,
        default="diff",
        help="Key in JSON file containing difficulty scores"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
        args.output_dir = f"./gsm8k_ablation_irt_diff_heuristic_{timestamp}"
    
    return args


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    args = parse_args()
    
    # Override global output directory
    BASE_OUTPUT_DIR = args.output_dir
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Override random seed
    random_seed = args.seed
    
    # Set seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    
    print(f"{'=' * 80}")
    print(f"ABLATION STUDY: IRT DIFFICULTY + HEURISTIC SCHEDULER")
    print(f"{'=' * 80}\n")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Exiting.")
        sys.exit(1)
    
    # Load tokenizer
    print(f"Loading tokenizer for {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        padding_side="left",
        cache_dir=HF_HOME
    )
    
    if tokenizer.pad_token is None:
        if tokenizer.eos_token:
            print("   Setting pad_token = eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        else:
            new_pad_token = "␂"
            if new_pad_token not in tokenizer.get_vocab():
                tokenizer.add_special_tokens({"pad_token": new_pad_token})
                print(f"   Added new pad_token: {new_pad_token}")
            else:
                tokenizer.pad_token = new_pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Tokenizer ready (vocab: {len(tokenizer)})\n")
    
    # Load dataset
    print(f"Loading GSM8K dataset...")
    dataset = load_dataset(
        dataset_id,
        dataset_config,
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )
    print(f"✅ Dataset loaded:")
    print(f"   Train: {len(dataset['train'])} examples")
    print(f"   Test:  {len(dataset['test'])} examples\n")
    
    # Load IRT difficulty scores
    difficulty_scores = load_difficulties_from_file(args.difficulty_file, args.difficulty_key)
    
    if len(difficulty_scores) != len(dataset['train']):
        raise ValueError(
            f"❌ Mismatch! Difficulty scores ({len(difficulty_scores)}) != "
            f"train samples ({len(dataset['train'])})."
        )
    
    print(f"✅ IRT difficulty scores loaded")
    print(f"   Min: {np.min(difficulty_scores):.4f}")
    print(f"   Max: {np.max(difficulty_scores):.4f}")
    print(f"   Mean: {np.mean(difficulty_scores):.4f}\n")
    
    # Run experiments for all schedulers
    overall_results = {}
    
    print("\n===== STARTING ABLATION EXPERIMENTS =====\n")
    
    for scheduler in SCHEDULERS:
        run_id = f"irt_diff_{scheduler}_{ORDERING}"
        run_output_dir = os.path.join(BASE_OUTPUT_DIR, run_id)
        
        print(f"\n{'#' * 80}")
        print(f"# Experiment: {run_id}")
        print(f"{'#' * 80}\n")
        
        run_result = run_training_with_curriculum(
            scheduler,
            run_output_dir,
            tokenizer,
            dataset,
            difficulty_scores
        )
        
        overall_results[run_id] = run_result
        time.sleep(5)
    
    # Print summary
    print("\n\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80 + "\n")
    
    print(f"{'Run ID':<40} {'Status':<15} {'Accuracy':<10}")
    print("-" * 80)
    for run_id, result in overall_results.items():
        status = result['status']
        acc = result['accuracy']
        print(f"{run_id:<40} {status:<15} {acc:.4f}")
    
    # Save summary
    summary_file = os.path.join(BASE_OUTPUT_DIR, "ablation_summary.json")
    with open(summary_file, "w") as f:
        json.dump(overall_results, f, indent=4)
    
    print(f"\n✅ Summary saved to: {summary_file}")
    print(f"\n✨ All ablation experiments complete!")
