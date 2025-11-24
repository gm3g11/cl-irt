# qwen_medqa_ablation_v9.py
import sys
import os
import datetime
import random
import traceback
import json
from tqdm import tqdm
import re
import time
import math
import gc
from typing import List, Dict, Any, Optional, Tuple, Iterator
import packaging
import torch
import numpy as np
import argparse
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    AutoConfig,
    logging as hf_logging,  # For verbosity control
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler
from torch.amp import GradScaler as TorchAmpGradScaler, autocast as torch_amp_autocast

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from huggingface_hub import whoami
import packaging.version  # For torch version check for gradient_checkpointing_kwargs

# --- IRT Scoring ---
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.special import expit


def theta_fn_irt(difficulties, student_prior, response_pattern):
    def fn(theta_val):
        theta_val = theta_val[0]
        probabilities = expit(theta_val - difficulties)
        log_likelihood = student_prior.logpdf(theta_val)
        probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
        for i, rp_val in enumerate(response_pattern):
            if rp_val == 1:
                log_likelihood += np.log(probabilities[i])
            elif rp_val == -1:
                log_likelihood += np.log(1 - probabilities[i])
        return -log_likelihood

    return fn


def calculate_theta_irt(difficulties, response_pattern, num_obs=-1, initial_theta_val=0.0):
    start_time_irt_calc = time.time()
    difficulties_np = np.array(difficulties, dtype=float)
    response_pattern_np = np.array(response_pattern, dtype=float)
    if len(difficulties_np) == 0 or len(difficulties_np) != len(response_pattern_np):
        return initial_theta_val, time.time() - start_time_irt_calc
    valid_indices = ~np.isnan(difficulties_np) & ~np.isnan(response_pattern_np)
    difficulties_filt, response_pattern_filt = difficulties_np[valid_indices], response_pattern_np[valid_indices]
    if len(difficulties_filt) == 0: return initial_theta_val, time.time() - start_time_irt_calc
    student_prior = norm(loc=0., scale=1.)
    if num_obs > 0 and len(difficulties_filt) > num_obs:
        num_obs_actual = min(num_obs, len(difficulties_filt))
        idx = np.random.choice(len(difficulties_filt), num_obs_actual, replace=False)
        difficulties_sample, response_pattern_sample = difficulties_filt[idx], response_pattern_filt[idx]
    else:
        difficulties_sample, response_pattern_sample = difficulties_filt, response_pattern_filt
    if len(difficulties_sample) == 0: return initial_theta_val, time.time() - start_time_irt_calc
    if not np.all(np.isin(response_pattern_sample, [-1, 1])): print(
        "  calculate_theta_irt: Warning - response pattern values other than 1/-1.")
    fn_min = theta_fn_irt(difficulties_sample, student_prior, response_pattern_sample)
    res = minimize(fn_min, [initial_theta_val], method='Nelder-Mead',
                   options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 500})
    est_theta = res['x'][0]
    if np.isnan(est_theta) or np.isinf(est_theta): est_theta = initial_theta_val
    return est_theta, time.time() - start_time_irt_calc


# --- End IRT Scoring ---

# --- Default Configurations ---
MODEL_ID = "Qwen/Qwen2.5-7B"
DATASET_ID = "GBaker/MedQA-USMLE-4-options"
DEFAULT_PUDF_DIFFICULTY_FILE_PATH = "/afs/crc/group/ball_lab/gmeng_cl/cl_new/gen_difficulty/MeD_QA/merged_jsonlines_output/test-1pl/best_parameters.json"
DIFFICULTY_JSON_KEY = "diff"
RANDOM_SEED = 63
ANSWER_MAP_KEYS = ["A", "B", "C", "D"];
NUM_CHOICES_MC = len(ANSWER_MAP_KEYS)
MAX_SEQ_LENGTH_SFT, MAX_PROMPT_LEN_SFT, MAX_TARGET_LEN_SFT = 512 + 10, 512, 10
MAX_NEW_TOKENS_FOR_GEN_ACCURACY = 1  # Specifically for choice prediction during accuracy eval

DEFAULT_INITIAL_CAPACITY_THETA_PUDF, DEFAULT_NUM_OBS_THETA_ESTIMATION_PUDF = 0.0, -1
DEFAULT_PUDF_LOWER_OFFSET, DEFAULT_PUDF_UPPER_OFFSET = -float('inf'), 0.0
DEFAULT_PUDF_MIN_SAMPLES_PER_EPOCH, DEFAULT_PUDF_ORDERING_PARAM = 100, 'easiest'
DEFAULT_HEURISTIC_ORDERING_PARAM, DEFAULT_COMPETENCY_PARAM_HEURISTIC = 'easiest', 5.0
DEFAULT_MIN_TRAIN_PERCENT_HEURISTIC, DEFAULT_C_INIT_HEURISTIC, DEFAULT_LOGGING_STEPS_HEURISTIC = 0.05, 0.01, 50
DEFAULT_QLORA_PER_DEVICE_TRAIN_BS, DEFAULT_QLORA_PER_DEVICE_EVAL_BS, DEFAULT_QLORA_GRAD_ACCUM_STEPS = 2, 4, 16
DEFAULT_QLORA_NUM_EPOCHS, DEFAULT_QLORA_LEARNING_RATE, DEFAULT_QLORA_WEIGHT_DECAY = 5, 1e-4, 0.01
DEFAULT_QLORA_R, DEFAULT_QLORA_ALPHA, DEFAULT_QLORA_DROPOUT = 16, 32, 0.05
DEFAULT_QLORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULT_EARLY_STOPPING_PATIENCE_SFT = 3


def setup_environment(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    HF_HOME = "/afs/crc/group/ball_lab/gmeng_cl/huggingface_cache"
    os.environ["HF_HOME"] = HF_HOME
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(HF_HOME, "models")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")
    os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    for cache_path in [os.environ["TRANSFORMERS_CACHE"], os.environ["HF_DATASETS_CACHE"], os.environ["HF_HUB_CACHE"]]:
        os.makedirs(cache_path, exist_ok=True)
    if "HF_TOKEN" in os.environ: del os.environ["HF_TOKEN"]; print("Removed HF_TOKEN.")
    try:
        user_info = whoami(token=os.getenv("HUGGING_FACE_HUB_TOKEN"))
        print(f"Logged in as: {user_info.get('name', 'Unknown User')}")
    except Exception as e:
        print(f"HF login failed: {e}.")


def set_seed(seed_value):
    torch.manual_seed(seed_value);
    np.random.seed(seed_value);
    random.seed(seed_value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(
        seed_value); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    print(f"Global random seed set to: {seed_value}")


def simple_tokenize(sent: str) -> List[str]:
    if not isinstance(sent, str): return []
    return [t.strip() for t in re.findall(r"[\w']+|[^\w\s]", re.sub(r'\s+', ' ', sent)) if t.strip()]


def get_example_rarities(texts: List[str]) -> List[float]:
    if not texts or not all(isinstance(t, str) for t in texts): return [0.0] * len(texts)
    corpus = [simple_tokenize(t) for t in texts];
    counts = {};
    N = sum(len(doc) for doc in corpus)
    if N == 0: return [0.0] * len(texts)
    for doc in corpus:
        for token in doc: counts[token] = counts.get(token, 0) + 1
    epsilon = 1e-9;
    result = []
    for doc in corpus:
        if not doc: result.append(0.0); continue
        log_probs = [np.log(counts.get(t, 0) / N + epsilon) for t in doc]
        result.append(-np.mean(log_probs) if log_probs else 0.0)
    return result


def combine_text_for_difficulty_medqa(example: Dict[str, Any]) -> Dict[str, str]:
    q = example.get("question", "").strip() if isinstance(example.get("question"), str) else ""
    o = example.get("options", {})
    opts = " ".join(str(o.get(k, "")) for k in ANSWER_MAP_KEYS if o.get(k)) if isinstance(o, dict) else ""
    full = f"{str(q).strip()} {opts.strip()}".strip();
    return {"full_text_for_difficulty": full if full else " "}


def calculate_heuristic_difficulty_scores(dataset_split: Dataset, measurer_type: str, text_col_for_difficulty_calc: str,
                                          map_num_procs: Optional[int]) -> List[float]:
    print(f"Calculating '{measurer_type}' difficulty scores using combined text...")
    dataset_with_combined_text = dataset_split.map(
        combine_text_for_difficulty_medqa,
        num_proc=map_num_procs,
        load_from_cache_file=False,
        desc="Combining text for difficulty")
    texts_for_scoring = dataset_with_combined_text[text_col_for_difficulty_calc]
    valid_entries = [(idx, text) for idx, text in enumerate(texts_for_scoring) if
                     isinstance(text, str) and text.strip()]
    valid_indices = [item[0] for item in valid_entries];
    valid_texts = [item[1] for item in valid_entries]
    if not valid_texts:
        print(
            f"Warning: No valid texts found in column '{text_col_for_difficulty_calc}' for difficulty scoring. Returning all zeros.")
        return [0.0] * len(dataset_split)
    if measurer_type == 'sentence_length':
        raw_scores_for_valid_texts = [len(text) for text in valid_texts]
    elif measurer_type == 'word_rarity':
        raw_scores_for_valid_texts = get_example_rarities(list(valid_texts))
        if raw_scores_for_valid_texts:
            scores_np = np.array(raw_scores_for_valid_texts)
            print(
                f"  DEBUG word_rarity: Raw scores - Min: {scores_np.min():.4f}, Max: {scores_np.max():.4f}, Mean: {scores_np.mean():.4f}, Std: {scores_np.std():.4f}, Count: {len(scores_np)}")
        else:
            print("  DEBUG word_rarity: No raw scores generated.")
    else:
        raise ValueError(f"Unsupported measurer: {measurer_type}")
    final_difficulty_scores = [0.0] * len(dataset_split)
    for original_idx, score in zip(valid_indices, raw_scores_for_valid_texts): final_difficulty_scores[
        original_idx] = score
    print(f"Calculated {measurer_type}. Processed {len(valid_texts)}/{len(dataset_split)} valid texts.");
    return final_difficulty_scores


def min_max_scale(scores, t_min, t_max):
    s_np = np.array(scores, float);
    if s_np.size == 0: return np.array([], float)
    min_r, max_r = s_np.min(), s_np.max()
    if not all(isinstance(v, (int, float)) for v in [t_min, t_max]): print(
        f"Warn: Invalid target range for scaling. Raw."); return s_np
    if min_r == max_r: return np.full_like(s_np, (t_min + t_max) / 2.0)
    return ((s_np - min_r) / (max_r - min_r)) * (t_max - t_min) + t_min


def load_pudf_irt_difficulties(f_path, key):
    try:
        with open(f_path, 'r') as f:
            data = json.load(f)
        s = data.get(key) if isinstance(data, dict) else data if isinstance(data, list) else None
        if s is None: raise KeyError(f"Key '{key}' error.")
        return np.array(s, float)
    except Exception as e:
        print(f"Err loading IRT diffs: {e}"); raise


def create_prompt_for_sft(ex: Dict[str, Any], style: str, chat_tokenizer=None) -> Tuple[str, str]:
    q, o, aidx = ex.get("question", "").strip(), ex.get("options", {}), str(ex.get("answer_idx", "")).strip().upper()
    target = aidx if aidx in ANSWER_MAP_KEYS else ""
    if style == "base":
        opts_str = "\n".join(f"{k}) {o.get(k, '')}" for k in ANSWER_MAP_KEYS) if isinstance(o, dict) else ""
        return f"Question: {q}\n\nOptions:\n{opts_str}\n\nAnswer:", target
    elif style == "chat_template_qwen":
        if not chat_tokenizer: raise ValueError("Tokenizer needed for chat template.")
        opts_str = "\n".join(f"{k}) {o.get(k, '')}" for k in ANSWER_MAP_KEYS) if isinstance(o, dict) else ""
        messages = [{"role": "user",
                     "content": f"Question: {q}\n\nOptions:\n{opts_str}\n\nWhat is the correct answer choice letter?"}]
        try:
            return chat_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), target
        except Exception as e:
            print(f"ERR applying chat template (NEEDS QWEN SPECIFIC IMPL): {e}"); return \
            create_prompt_for_sft(ex, "base")[0], target
    raise ValueError(f"Bad prompt_style: {style}")


def preprocess_sft_data(examples_batch: Dict[str, List], tokenizer, max_seq_len: int, max_prompt_len: int,
                        max_target_len: int, prompt_style: str):
    inputs_ids_batch, labels_batch, attention_masks_batch = [], [], []
    key0 = next(iter(examples_batch), None)
    if not key0: return {"input_ids": [], "labels": [], "attention_mask": []}
    num_examples_in_batch = len(examples_batch[key0])

    original_tokenizer_padding_side = tokenizer.padding_side
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    for i in range(num_examples_in_batch):
        raw_ex = {k: examples_batch[k][i] for k in examples_batch.keys()}
        prompt_txt, target_lttr = create_prompt_for_sft(raw_ex, prompt_style,
                                                        tokenizer if "chat_template" in prompt_style else None)
        if not target_lttr: continue

        sft_sequence_text = prompt_txt + " " + target_lttr  # Add space for token distinctness
        if tokenizer.eos_token and not sft_sequence_text.endswith(tokenizer.eos_token):
            sft_sequence_text += tokenizer.eos_token

        tokenized_full_sequence = tokenizer(
            sft_sequence_text, max_length=max_seq_len, padding="max_length",
            truncation=True, return_attention_mask=True, add_special_tokens=True)

        current_input_ids = tokenized_full_sequence["input_ids"]
        current_attention_mask = tokenized_full_sequence["attention_mask"]

        tokenized_prompt_only = tokenizer(prompt_txt, max_length=max_prompt_len, truncation=True,
                                          add_special_tokens=True)
        prompt_tokens_len = len(tokenized_prompt_only.input_ids)

        current_labels = list(current_input_ids)
        for k in range(len(current_labels)):
            if k < prompt_tokens_len or current_attention_mask[k] == 0:
                current_labels[k] = -100

        inputs_ids_batch.append(current_input_ids)
        labels_batch.append(current_labels)
        attention_masks_batch.append(current_attention_mask)

    tokenizer.padding_side = original_tokenizer_padding_side

    if not hasattr(preprocess_sft_data, 'printed_once_sft_medqa_v8') and inputs_ids_batch:  # Changed attr name
        print("--- Debug Unified SFT Preprocessing (MedQA v8 - First valid example of first batch call) ---")
        idx = 0
        if examples_batch[key0] and idx < len(examples_batch[key0]):
            raw_example_for_debug = {k: examples_batch[k][idx] for k in examples_batch.keys()}
            _pt, _tl = create_prompt_for_sft(raw_example_for_debug, prompt_style,
                                             tokenizer if "chat_template" in prompt_style else None)
            print(f"Prompt style: {prompt_style}\nInput Prompt Text was: '{_pt}'\nInput Target Letter was: '{_tl}'")
            print(f"Final Input IDs ({len(inputs_ids_batch[idx])}): {inputs_ids_batch[idx]}")
            print(f"Tokens (decoded from input_ids): {tokenizer.decode(inputs_ids_batch[idx])}")
            print(f"Final Labels ({len(labels_batch[idx])}): {labels_batch[idx]}")
            decoded_lbl_parts = ["[-100]" if lbl_id == -100 else tokenizer.decode([lbl_id]) for lbl_id in
                                 labels_batch[idx]]
            print(f"Decoded Labels: {' '.join(decoded_lbl_parts)}")
            print(f"Attention Mask ({len(attention_masks_batch[idx])}): {attention_masks_batch[idx]}")
            preprocess_sft_data.printed_once_sft_medqa_v8 = True;
            print("--- End Debug ---")

    return {"input_ids": inputs_ids_batch, "labels": labels_batch, "attention_mask": attention_masks_batch}


def select_data_for_pudf_epoch(
        full_ds_sft: Dataset, cap_theta: float, diff_col: str, order: str, low_off: float, up_off: float,
        min_samp: int, map_n_procs: Optional[int]):
    start_time = time.time()
    print(
        f"  PUDF Sel: cap={cap_theta:.4f}, win=[{cap_theta + low_off:.4f},{cap_theta + up_off:.4f}), min={min_samp}, ord={order}")
    if diff_col not in full_ds_sft.column_names: print(
        f"  Err: Diff col '{diff_col}' missing."); return full_ds_sft.select([])
    valid_ds = full_ds_sft.filter(lambda x: x[diff_col] is not None and not np.isnan(x[diff_col]), num_proc=map_n_procs,
                                  load_from_cache_file=False, desc="PUDF FiltValidDiff")
    if not valid_ds: print(f"  Warn: No valid diffs for PUDF sel."); return full_ds_sft.select([])
    sel_ds = valid_ds.filter(lambda x: (cap_theta + low_off) <= x[diff_col] < (cap_theta + up_off),
                             num_proc=map_n_procs, load_from_cache_file=False, desc="PUDF WinSel")
    print(f"  PUDF WinSel: {len(sel_ds)} samples.")
    if len(sel_ds) < min_samp and len(valid_ds) > 0:
        print(f"  PUDF Sel {len(sel_ds)} < min {min_samp}. Resorting from {len(valid_ds)} valid.")
        n_take = min(min_samp, len(valid_ds))
        try:
            sel_ds = valid_ds.sort(diff_col, reverse=(order == 'hardest'), load_from_cache_file=False).select(
                range(n_take))
        except Exception as e:
            print(f"  Err sorting for min_samp: {e}.")
    elif len(sel_ds) < min_samp and len(valid_ds) == 0:
        print(f"  PUDF Sel {len(sel_ds)} < min {min_samp}, but no valid items to sort from.")
    print(f"  PUDF Final Sel: {len(sel_ds)} samp. Time: {time.time() - start_time:.2f}s.");
    return sel_ds


class HeuristicSampler(Sampler[int]):
    def __init__(self, num_samples_total: int, batch_size: int, sorted_indices: list[int], heuristic_config: dict,
                 num_replicas: int = 1, rank: int = 0, seed: int = RANDOM_SEED):
        if num_replicas <= 0 or rank < 0 or rank >= num_replicas: raise ValueError("Invalid num_replicas or rank.")
        if not isinstance(batch_size, int) or batch_size <= 0: raise ValueError("batch_size should be positive.")
        self.num_replicas, self.rank, self.epoch, self.seed, self.batch_size = num_replicas, rank, 0, seed, batch_size
        self._full_data_len, self._sorted_indices, self.heuristic_config = num_samples_total, sorted_indices, heuristic_config
        min_perc = float(self.heuristic_config.get('min_train_percent', DEFAULT_MIN_TRAIN_PERCENT_HEURISTIC))
        min_len_perc = int(min_perc * self._full_data_len)
        min_len_batch_repl = batch_size * num_replicas if self._full_data_len >= batch_size * num_replicas else self._full_data_len
        self._min_train_length = max(1, min_len_perc, min_len_batch_repl)
        if self._min_train_length > self._full_data_len: self._min_train_length = self._full_data_len
        self.indices_for_epoch: List[int] = [];
        self.num_samples_epoch_replica = 0;
        self.replica_indices_epoch: List[int] = []
        self.set_epoch(0)

    def _get_num_samples_for_epoch(self, epoch: int) -> int:
        sched_type = self.heuristic_config['scheduler_type']
        comp_param = float(self.heuristic_config.get('competency_param', DEFAULT_COMPETENCY_PARAM_HEURISTIC))
        comp_epoch = max(1.0, comp_param)
        c_init_val = float(self.heuristic_config.get('c_init', DEFAULT_C_INIT_HEURISTIC))
        prog_ratio = float(epoch) / comp_epoch
        if sched_type == 'linear':
            competency = c_init_val + (1.0 - c_init_val) * prog_ratio if prog_ratio < 1.0 else 1.0
        elif sched_type == 'root':
            competency = c_init_val + (1.0 - c_init_val) * np.sqrt(max(0.0, prog_ratio)) if prog_ratio < 1.0 else 1.0
        else:
            raise NotImplementedError(f"Scheduler '{sched_type}' unknown.")
        return max(self._min_train_length, min(int(competency * self._full_data_len), self._full_data_len))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch;
        new_indices = self._sorted_indices[:self._get_num_samples_for_epoch(epoch)]
        if epoch == 0 or len(new_indices) != len(self.indices_for_epoch) or epoch % 5 == 0:
            print(
                f"[HeuristicSampler] E{epoch}: Sel {len(new_indices)}/{self._full_data_len}. MinLen {self._min_train_length}.")
        self.indices_for_epoch = new_indices;
        num_this_epoch = len(self.indices_for_epoch)
        if self.num_replicas > 1:
            if num_this_epoch == 0: self.replica_indices_epoch = []; self.num_samples_epoch_replica = 0; return
            base, extra = num_this_epoch // self.num_replicas, num_this_epoch % self.num_replicas
            start = self.rank * base + min(self.rank, extra)
            self.num_samples_epoch_replica = base + (1 if self.rank < extra else 0)
            end = start + self.num_samples_epoch_replica
            self.replica_indices_epoch = self.indices_for_epoch[start: end]
        else:
            self.replica_indices_epoch = self.indices_for_epoch; self.num_samples_epoch_replica = num_this_epoch

    def __iter__(self) -> Iterator[int]:
        if not self.replica_indices_epoch: return iter([])
        g = torch.Generator();
        g.manual_seed(self.seed + self.epoch)
        return iter([self.replica_indices_epoch[i] for i in
                     torch.randperm(len(self.replica_indices_epoch), generator=g).tolist()])

    def __len__(self) -> int:
        return self.num_samples_epoch_replica


class CustomHeuristicTrainer(Trainer):
    def __init__(self, *args: Any, sorted_indices_sampler=None, heuristic_config_sampler=None,
                 num_samples_total_sampler=None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        if not all(
                p is not None for p in [sorted_indices_sampler, heuristic_config_sampler, num_samples_total_sampler]):
            raise ValueError("CustomHeuristicTrainer missing required sampler args.")
        self.sorted_indices_sampler, self.heuristic_config_sampler, self.num_samples_total_sampler = sorted_indices_sampler, heuristic_config_sampler, num_samples_total_sampler
        print("CustomHeuristicTrainer initialized for QLoRA SFT.")

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None: raise ValueError("Trainer: training requires a train_dataset.")
        sampler = HeuristicSampler(self.num_samples_total_sampler, self._train_batch_size, self.sorted_indices_sampler,
                                   self.heuristic_config_sampler, self.args.world_size, self.args.process_index,
                                   self.args.seed)
        return DataLoader(self.train_dataset, batch_size=self._train_batch_size, sampler=sampler,
                          collate_fn=self.data_collator,
                          drop_last=self.args.dataloader_drop_last, num_workers=self.args.dataloader_num_workers,
                          pin_memory=self.args.dataloader_pin_memory)


def perform_custom_sft_evaluation(model_to_eval: torch.nn.Module, raw_eval_dataset: Dataset,
                                  sft_processed_val_dataset_for_loss: Optional[Dataset] = None,
                                  tokenizer_eval: AutoTokenizer = None, device_eval: torch.device = None,
                                  prompt_style_eval: str = 'base', batch_size_eval: int = 4,
                                  eval_prompt_max_len: int = 512, letter_token_ids_map: Dict[str, int] = None,
                                  max_new_toks_for_gen_accuracy: int = 1,
                                  desc_prefix: str = "SFT_Eval",
                                  current_irt_theta_init: float = 0.0, calculate_irt_theta_flag: bool = False,
                                  data_collator_for_loss=None, bf16_enabled_eval=False):
    model_to_eval.eval();
    avg_sft_loss, perplexity = float('nan'), float('inf');
    acc, est_theta, theta_time = 0.0, current_irt_theta_init, 0.0
    if sft_processed_val_dataset_for_loss and data_collator_for_loss and len(sft_processed_val_dataset_for_loss) > 0:
        print(f"  {desc_prefix}: SFT Loss on {len(sft_processed_val_dataset_for_loss)} samples...");
        sft_val_cols = ['input_ids', 'attention_mask', 'labels']
        sft_processed_val_dataset_for_loss.set_format(type='torch', columns=[c for c in sft_val_cols if
                                                                             c in sft_processed_val_dataset_for_loss.column_names])
        dl = DataLoader(sft_processed_val_dataset_for_loss, batch_size=batch_size_eval,
                        collate_fn=data_collator_for_loss, shuffle=False)
        tot_loss, n_batches = 0, 0
        for batch in tqdm(dl, desc=f"  {desc_prefix} SFT Loss", leave=False):
            batch = {k: v.to(device_eval) for k, v in batch.items() if hasattr(v, 'to')}
            if not batch.get('input_ids', torch.empty(0)).numel(): continue
            with torch.no_grad(), torch_amp_autocast(device_type=device_eval.type, enabled=bf16_enabled_eval):
                out = model_to_eval(**batch)
                if out.loss is not None: tot_loss += out.loss.item(); n_batches += 1
        if n_batches > 0: avg_sft_loss = tot_loss / n_batches; perplexity = np.exp(avg_sft_loss) if not np.isnan(
            avg_sft_loss) else float('inf')
        print(f"  {desc_prefix}: Avg SFT Loss={avg_sft_loss:.4f}, PPL={perplexity:.4f}")

    print(f"  {desc_prefix}: Choice Pred for Acc/IRT on {len(raw_eval_dataset)} raw samples...")
    pred_L, true_L, irt_D, irt_RP = [], [], [], []
    if not letter_token_ids_map:
        print(
            f"  {desc_prefix}: CRITICAL letter_token_ids_map is empty. Accuracy/IRT will be 0/default. Map: {letter_token_ids_map}")
        return avg_sft_loss, perplexity, 0.0, current_irt_theta_init, 0.0
    if len(raw_eval_dataset) == 0: print(
        f"  {desc_prefix}: Raw eval dataset empty."); return avg_sft_loss, perplexity, 0.0, current_irt_theta_init, 0.0

    orig_pad_side = tokenizer_eval.padding_side
    if tokenizer_eval.padding_side != "left":  # Ensure left padding for batch generation
        print(f"  {desc_prefix}: Temporarily setting tokenizer_eval.padding_side to 'left' for generation.")
        tokenizer_eval.padding_side = "left"

    with torch.inference_mode():
        for i in tqdm(range(0, len(raw_eval_dataset), batch_size_eval), desc=f"  {desc_prefix} Choice Pred",
                      leave=False):
            batch_raw = [raw_eval_dataset[j] for j in range(i, min(i + batch_size_eval, len(raw_eval_dataset)))]
            prompts, targets, cur_diffs = [], [], []
            for ex_r in batch_raw:
                p_txt, t_ltr = create_prompt_for_sft(ex_r, prompt_style_eval,
                                                     tokenizer_eval if "chat_template" in prompt_style_eval else None)
                prompts.append(p_txt);
                targets.append(t_ltr)
                if calculate_irt_theta_flag and "difficulty" in ex_r and ex_r["difficulty"] is not None:
                    cur_diffs.append(float(ex_r["difficulty"]))
                elif calculate_irt_theta_flag:
                    cur_diffs.append(np.nan)

            inputs = tokenizer_eval(prompts, return_tensors="pt", padding=True, truncation=True,
                                    max_length=eval_prompt_max_len).to(device_eval)
            pad_id = tokenizer_eval.pad_token_id if tokenizer_eval.pad_token_id is not None else getattr(
                model_to_eval.config, 'pad_token_id', 0)
            if pad_id is None and tokenizer_eval.eos_token_id is not None: pad_id = tokenizer_eval.eos_token_id
            if pad_id is None: pad_id = 0

            gen_ids = model_to_eval.generate(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_toks_for_gen_accuracy,
                pad_token_id=pad_id, eos_token_id=tokenizer_eval.eos_token_id,
                do_sample=False
            )

            for j_idx, g_ids_sample in enumerate(gen_ids):
                prompt_len = inputs.input_ids[j_idx].shape[0]
                predicted_letter_choice = None
                generated_tokens_ids = g_ids_sample[prompt_len:]

                if generated_tokens_ids.numel() > 0:
                    first_new_token_id = generated_tokens_ids[0].item()
                    predicted_letter_choice = next(
                        (L for L, tid in letter_token_ids_map.items() if tid == first_new_token_id), None)

                true_target_letter = targets[j_idx]

                # Enhanced Debug Print:
                print_this_example_flag = (i == 0 and j_idx < 2) or \
                                          (predicted_letter_choice != true_target_letter and \
                                           not getattr(perform_custom_sft_evaluation,
                                                       f"printed_mismatch_flag_{desc_prefix}_{i}", False))

                if print_this_example_flag:
                    raw_decoded_gen_debug = tokenizer_eval.decode(generated_tokens_ids, skip_special_tokens=False)
                    cleaned_decoded_gen_for_debug = tokenizer_eval.decode(generated_tokens_ids,
                                                                          skip_special_tokens=True).strip().upper()
                    print(f"\n--- DEBUG EVAL ({desc_prefix}) Example ---")
                    print(f"  Raw Prompt (last 50 toks): '...{tokenizer_eval.decode(inputs.input_ids[j_idx][-50:])}'")
                    print(f"  True Target Letter: '{true_target_letter}'")
                    print(
                        f"  Generated Token IDs (after prompt, max {max_new_toks_for_gen_accuracy}): {generated_tokens_ids.tolist()}")
                    print(f"  Raw Decoded Generation: '{raw_decoded_gen_debug}'")
                    print(f"  Cleaned Decoded (for potential startswith debug): '{cleaned_decoded_gen_for_debug}'")
                    print(f"  Predicted Choice (from first token ID vs map): '{predicted_letter_choice}'")
                    if predicted_letter_choice != true_target_letter:
                        setattr(perform_custom_sft_evaluation, f"printed_mismatch_flag_{desc_prefix}_{i}", True)
                    print("-------------------------------------------\n")

                pred_L.append(predicted_letter_choice);
                true_L.append(true_target_letter)
                if calculate_irt_theta_flag and j_idx < len(cur_diffs) and not np.isnan(cur_diffs[j_idx]):
                    difficulty_val = cur_diffs[j_idx]
                    irt_D.append(difficulty_val);
                    irt_RP.append(
                        1 if predicted_letter_choice == targets[j_idx] and predicted_letter_choice is not None else -1)

    tokenizer_eval.padding_side = orig_pad_side
    corr = sum(1 for p, t in zip(pred_L, true_L) if p == t and p is not None);
    acc = corr / len(true_L) if true_L else 0.0
    print(f"  {desc_prefix}: Custom Acc = {acc:.4f} ({corr}/{len(true_L)})")
    if calculate_irt_theta_flag:
        t_s = time.time()
        if irt_D and irt_RP:
            est_theta, _ = calculate_theta_irt(irt_D, irt_RP, -1, current_irt_theta_init)
        else:
            print(f"  {desc_prefix}: No data for IRT theta. Using init {current_irt_theta_init:.4f}.")
        theta_time = time.time() - t_s
    return avg_sft_loss, perplexity, acc, est_theta, theta_time


def run_training_with_heuristic_qlora_sft(args, train_full_hf_with_difficulty: Dataset,
                                          val_hf_raw: Dataset, test_hf_raw: Dataset,
                                          global_tokenizer: AutoTokenizer,
                                          run_output_dir: str,
                                          letter_token_ids_map_eval: Dict[str, int]):
    print(f"\nStarting Heuristic QLoRA SFT Run: Diff='{args.difficulty_measurer}', Sched='{args.training_scheduler}'")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_enabled = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and args.use_bf16
    map_num_procs = args.num_workers if args.num_workers > 0 else None
    heuristic_ordering_actual = args.heuristic_ordering if args.heuristic_ordering is not None else DEFAULT_HEURISTIC_ORDERING_PARAM
    print(f"  Sorting training data by 'difficulty' ({heuristic_ordering_actual} first)...")
    difficulty_values_np = np.array(train_full_hf_with_difficulty['difficulty'])
    if difficulty_values_np.size == 0: raise ValueError("Difficulty column is empty for Heuristic Sampler.")
    if heuristic_ordering_actual == 'easiest':
        sorted_original_indices = np.argsort(difficulty_values_np).tolist()
    elif heuristic_ordering_actual == 'hardest':
        sorted_original_indices = np.argsort(difficulty_values_np)[::-1].tolist()
    else:
        raise NotImplementedError(f"Ordering '{heuristic_ordering_actual}' not supported.")
    num_samples_total_for_sampler = len(train_full_hf_with_difficulty)

    print("  SFT Preprocessing datasets for Heuristic Trainer...")
    cols_to_remove_train = [c for c in train_full_hf_with_difficulty.column_names if
                            c not in ['difficulty', 'question', 'options', 'answer_idx']]
    sft_train_dataset = train_full_hf_with_difficulty.map(
        lambda ex_batch: preprocess_sft_data(ex_batch, global_tokenizer, MAX_SEQ_LENGTH_SFT, MAX_PROMPT_LEN_SFT,
                                             MAX_TARGET_LEN_SFT, args.prompt_style),
        batched=True, num_proc=map_num_procs, load_from_cache_file=False, remove_columns=cols_to_remove_train,
        desc="SFT Preprocessing Train (Heuristic)")

    cols_to_remove_val = [c for c in val_hf_raw.column_names if c not in ['question', 'options', 'answer_idx']]
    sft_val_dataset_for_trainer_loss = val_hf_raw.map(
        lambda ex_batch: preprocess_sft_data(ex_batch, global_tokenizer, MAX_SEQ_LENGTH_SFT, MAX_PROMPT_LEN_SFT,
                                             MAX_TARGET_LEN_SFT, args.prompt_style),
        batched=True, num_proc=map_num_procs, load_from_cache_file=False, remove_columns=cols_to_remove_val,
        desc="SFT Preprocessing Val (Heuristic)")

    final_trainer_cols = ['input_ids', 'labels', 'attention_mask']
    sft_train_dataset_for_trainer = sft_train_dataset.select_columns(
        [c for c in final_trainer_cols if c in sft_train_dataset.column_names])
    sft_val_dataset_for_trainer_loss = sft_val_dataset_for_trainer_loss.select_columns(
        [c for c in final_trainer_cols if c in sft_val_dataset_for_trainer_loss.column_names])

    sft_train_dataset_for_trainer.set_format(type="torch")
    sft_val_dataset_for_trainer_loss.set_format(type="torch")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16 if bf16_enabled else torch.float16,
                                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model_dtype_load = torch.bfloat16 if bf16_enabled else torch.float16
    model_load_kwargs_h = {"quantization_config": bnb_config, "device_map": "auto", "trust_remote_code": True,
                           "torch_dtype": model_dtype_load, "attn_implementation": "flash_attention_2",
                           "cache_dir": os.environ.get("TRANSFORMERS_CACHE")}
    base_model_h = AutoModelForCausalLM.from_pretrained(args.model_id, **model_load_kwargs_h)
    if len(global_tokenizer) > base_model_h.config.vocab_size: base_model_h.resize_token_embeddings(
        len(global_tokenizer))
    if base_model_h.config.pad_token_id != global_tokenizer.pad_token_id and global_tokenizer.pad_token_id is not None: base_model_h.config.pad_token_id = global_tokenizer.pad_token_id
    if base_model_h.config.pad_token_id is None: raise ValueError(
        "Pad token ID for heuristic base model None after sync.")

    qlora_r_h = args.qlora_r if args.qlora_r is not None else DEFAULT_QLORA_R
    qlora_alpha_h = args.qlora_alpha if args.qlora_alpha is not None else DEFAULT_QLORA_ALPHA
    qlora_dropout_h = args.qlora_dropout if args.qlora_dropout is not None else DEFAULT_QLORA_DROPOUT

    base_model_h = prepare_model_for_kbit_training(base_model_h,
                                                   use_gradient_checkpointing=args.use_gradient_checkpointing)
    lora_config_h = LoraConfig(task_type=TaskType.CAUSAL_LM, r=qlora_r_h, lora_alpha=qlora_alpha_h,
                               lora_dropout=qlora_dropout_h, target_modules=args.qlora_target_modules, bias="none")
    peft_model_h = get_peft_model(base_model_h, lora_config_h);
    peft_model_h.print_trainable_parameters()

    data_collator_h = DataCollatorForSeq2Seq(global_tokenizer, model=peft_model_h, label_pad_token_id=-100,
                                             padding="longest")
    world_size = torch.cuda.device_count() if args.num_gpus > 1 and torch.cuda.is_available() and torch.distributed.is_initialized() else 1
    train_bs_h = args.train_batch_size if args.train_batch_size is not None else DEFAULT_QLORA_PER_DEVICE_TRAIN_BS
    grad_acc_h = args.grad_accum_steps if args.grad_accum_steps is not None else DEFAULT_QLORA_GRAD_ACCUM_STEPS
    num_epochs_h = args.num_epochs if args.num_epochs is not None else DEFAULT_QLORA_NUM_EPOCHS
    eff_bs_h = train_bs_h * world_size * grad_acc_h
    steps_epoch_full_h = math.ceil(num_samples_total_for_sampler / eff_bs_h) if eff_bs_h > 0 else 0
    max_steps_h = math.ceil(num_epochs_h * steps_epoch_full_h) if steps_epoch_full_h > 0 else num_epochs_h
    print(f"  Heuristic Trainer: EffBS={eff_bs_h}, Steps/Epoch(Full)={steps_epoch_full_h}, MaxSteps={max_steps_h}")

    # Use packaging.version to parse torch version for gradient_checkpointing_kwargs
    gc_kwargs = {}
    if args.use_gradient_checkpointing and packaging.version.parse(torch.__version__) >= packaging.version.parse(
            "2.0.0"):
        gc_kwargs = {"use_reentrant": False}

    training_args_obj_heuristic = TrainingArguments(
        output_dir=os.path.join(run_output_dir, "ckpts_heuristic"), per_device_train_batch_size=train_bs_h,
        gradient_accumulation_steps=grad_acc_h,
        max_steps=max(1, max_steps_h),
        learning_rate=args.learning_rate if args.learning_rate is not None else DEFAULT_QLORA_LEARNING_RATE,
        weight_decay=args.weight_decay if args.weight_decay is not None else DEFAULT_QLORA_WEIGHT_DECAY,
        bf16=bf16_enabled, fp16=not bf16_enabled and torch.cuda.is_available(),
        logging_strategy="steps",
        logging_steps=args.logging_steps if args.logging_steps is not None else DEFAULT_LOGGING_STEPS_HEURISTIC,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        gradient_checkpointing=args.use_gradient_checkpointing, gradient_checkpointing_kwargs=gc_kwargs,
        dataloader_num_workers=args.num_workers, report_to="none", seed=args.seed,
        ddp_find_unused_parameters=False if world_size > 1 else None, )

    heuristic_sampler_cfg = {"scheduler_type": args.training_scheduler, "ordering": heuristic_ordering_actual,
                             "competency_param": args.competency_param_heuristic if args.competency_param_heuristic is not None else DEFAULT_COMPETENCY_PARAM_HEURISTIC,
                             "min_train_percent": args.min_train_percent_heuristic if args.min_train_percent_heuristic is not None else DEFAULT_MIN_TRAIN_PERCENT_HEURISTIC,
                             "c_init": args.c_init_heuristic if args.c_init_heuristic is not None else DEFAULT_C_INIT_HEURISTIC, }

    trainer_h = CustomHeuristicTrainer(model=peft_model_h, args=training_args_obj_heuristic,
                                       train_dataset=sft_train_dataset_for_trainer,
                                       eval_dataset=sft_val_dataset_for_trainer_loss,
                                       tokenizer=global_tokenizer, data_collator=data_collator_h, callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.patience_early_stopping if args.patience_early_stopping is not None else DEFAULT_EARLY_STOPPING_PATIENCE_SFT)],
                                       sorted_indices_sampler=sorted_original_indices,
                                       heuristic_config_sampler=heuristic_sampler_cfg,
                                       num_samples_total_sampler=num_samples_total_for_sampler)
    if args.use_gradient_checkpointing and hasattr(trainer_h.model, 'gradient_checkpointing_enable'):
        trainer_h.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=training_args_obj_heuristic.gradient_checkpointing_kwargs)
        if hasattr(trainer_h.model.config, "use_cache"): trainer_h.model.config.use_cache = False

    start_t, best_val_loss_h, test_acc_h, epochs_done_h = time.time(), float('inf'), 0.0, 0;
    t_time = 0.0
    if max_steps_h > 0 and num_samples_total_for_sampler > 0 and len(sft_train_dataset_for_trainer) > 0:
        train_res = trainer_h.train();
        t_time = time.time() - start_t;
        epochs_done_h = train_res.epoch if hasattr(train_res, 'epoch') else num_epochs_h
        print(f"  Heuristic Trainer finished. Time: {t_time:.2f}s");
        trainer_h.save_model(os.path.join(run_output_dir, "best_heuristic_adapter"))
        global_tokenizer.save_pretrained(os.path.join(run_output_dir, "best_heuristic_adapter"))
        if hasattr(train_res, 'metrics') and train_res.metrics: best_val_loss_h = train_res.metrics.get("eval_loss",
                                                                                                        float('inf'))

        final_eval_model_heuristic = trainer_h.model
        _, _, test_acc_h, _, _ = perform_custom_sft_evaluation(final_eval_model_heuristic, test_hf_raw, None,
                                                               global_tokenizer, DEVICE, args.prompt_style,
                                                               args.eval_batch_size or DEFAULT_QLORA_PER_DEVICE_EVAL_BS,
                                                               MAX_PROMPT_LEN_SFT, letter_token_ids_map_eval,
                                                               max_new_toks_for_gen_accuracy=MAX_NEW_TOKENS_FOR_GEN_ACCURACY,
                                                               # Use defined constant
                                                               desc_prefix="FinalHeuristicTest",
                                                               bf16_enabled_eval=bf16_enabled)
        print(f"  Final Test Accuracy (Heuristic): {test_acc_h:.4f}")
    else:
        print("  Skipping Heuristic training."); t_time = 0.0

    return {"best_val_metric (loss)": best_val_loss_h, "test_accuracy": test_acc_h, "training_time_s": t_time,
            "epochs_completed": epochs_done_h, "output_dir": run_output_dir}


def run_training_with_pudf_qlora_sft(args, train_full_hf_with_difficulty: Dataset,
                                     val_hf_raw: Dataset, test_hf_raw: Dataset,
                                     global_tokenizer: AutoTokenizer,
                                     run_output_dir: str,
                                     letter_token_ids_map_eval: Dict[str, int]):
    print(f"\nStarting PUDF QLoRA SFT Run: Diff='{args.difficulty_measurer}', Sched='pudf_theta'")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_enabled_pudf = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and args.use_bf16
    map_num_procs = args.num_workers if args.num_workers > 0 else None

    print("  SFT Preprocessing datasets for PUDF run...")
    sft_train_cols_to_remove = [c for c in train_full_hf_with_difficulty.column_names if
                                c not in ['difficulty', 'question', 'options', 'answer_idx']]
    sft_train_full_dataset = train_full_hf_with_difficulty.map(
        lambda ex: preprocess_sft_data(ex, global_tokenizer, MAX_SEQ_LENGTH_SFT, MAX_PROMPT_LEN_SFT, MAX_TARGET_LEN_SFT,
                                       args.prompt_style),
        batched=True, num_proc=map_num_procs, load_from_cache_file=False, remove_columns=sft_train_cols_to_remove,
        desc="SFT Preproc Train (PUDF)")

    sft_val_cols_to_remove = [c for c in val_hf_raw.column_names if
                              c not in ['difficulty', 'question', 'options', 'answer_idx']]
    sft_val_dataset_for_loss_pudf = val_hf_raw.map(
        lambda ex: preprocess_sft_data(ex, global_tokenizer, MAX_SEQ_LENGTH_SFT, MAX_PROMPT_LEN_SFT, MAX_TARGET_LEN_SFT,
                                       args.prompt_style),
        batched=True, num_proc=map_num_procs, load_from_cache_file=False, remove_columns=sft_val_cols_to_remove,
        desc="SFT Preproc Val (PUDF)")

    final_sft_cols = ['input_ids', 'labels', 'attention_mask']
    sft_val_dataset_for_loss_pudf = sft_val_dataset_for_loss_pudf.select_columns(
        [c for c in final_sft_cols if c in sft_val_dataset_for_loss_pudf.column_names])
    sft_val_dataset_for_loss_pudf.set_format(type="torch")

    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_compute_dtype=torch.bfloat16 if bf16_enabled_pudf else torch.float16,
                                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model_dtype_load = torch.bfloat16 if bf16_enabled_pudf else torch.float16
    model_load_kwargs_p = {"quantization_config": bnb_config, "device_map": "auto", "trust_remote_code": True,
                           "torch_dtype": model_dtype_load, "attn_implementation": "flash_attention_2",
                           "cache_dir": os.environ.get("TRANSFORMERS_CACHE")}
    base_model_pudf = AutoModelForCausalLM.from_pretrained(args.model_id, **model_load_kwargs_p)
    if len(global_tokenizer) > base_model_pudf.config.vocab_size: base_model_pudf.resize_token_embeddings(
        len(global_tokenizer))
    if base_model_pudf.config.pad_token_id != global_tokenizer.pad_token_id and global_tokenizer.pad_token_id is not None: base_model_pudf.config.pad_token_id = global_tokenizer.pad_token_id
    if base_model_pudf.config.pad_token_id is None: raise ValueError("Pad token ID for PUDF base model None.")

    qlora_r_p = args.qlora_r if args.qlora_r is not None else DEFAULT_QLORA_R
    qlora_alpha_p = args.qlora_alpha if args.qlora_alpha is not None else DEFAULT_QLORA_ALPHA
    qlora_dropout_p = args.qlora_dropout if args.qlora_dropout is not None else DEFAULT_QLORA_DROPOUT

    base_model_pudf = prepare_model_for_kbit_training(base_model_pudf,
                                                      use_gradient_checkpointing=args.use_gradient_checkpointing)
    lora_config_pudf = LoraConfig(task_type=TaskType.CAUSAL_LM, r=qlora_r_p, lora_alpha=qlora_alpha_p,
                                  lora_dropout=qlora_dropout_p, target_modules=args.qlora_target_modules, bias="none")
    peft_model_pudf = get_peft_model(base_model_pudf, lora_config_pudf);
    peft_model_pudf.print_trainable_parameters()

    optimizer_pudf = AdamW(peft_model_pudf.parameters(), lr=args.learning_rate or DEFAULT_QLORA_LEARNING_RATE,
                           weight_decay=args.weight_decay or DEFAULT_QLORA_WEIGHT_DECAY)
    data_collator_pudf = DataCollatorForSeq2Seq(global_tokenizer, model=peft_model_pudf, label_pad_token_id=-100,
                                                padding="longest")
    scaler_pudf = TorchAmpGradScaler(enabled=bf16_enabled_pudf)
    num_outer_epochs_p = args.num_epochs or DEFAULT_QLORA_NUM_EPOCHS
    train_bs_p = args.train_batch_size or DEFAULT_QLORA_PER_DEVICE_TRAIN_BS
    grad_acc_p = args.grad_accum_steps or DEFAULT_QLORA_GRAD_ACCUM_STEPS
    pudf_min_samps_actual = args.pudf_min_samples_per_epoch or DEFAULT_PUDF_MIN_SAMPLES_PER_EPOCH

    avg_exp_samps = max(pudf_min_samps_actual, int(len(sft_train_full_dataset) * 0.6))
    eff_batch_size_pudf = train_bs_p * grad_acc_p
    steps_avg_epoch_p = math.ceil(avg_exp_samps / eff_batch_size_pudf) if eff_batch_size_pudf > 0 else 0
    total_steps_approx_p = max(1, int(steps_avg_epoch_p * num_outer_epochs_p))
    lr_scheduler_pudf = get_linear_schedule_with_warmup(optimizer_pudf,
                                                        num_warmup_steps=max(1, int(0.1 * total_steps_approx_p)),
                                                        num_training_steps=total_steps_approx_p)

    cur_cap_theta_p = args.pudf_initial_capacity if args.pudf_initial_capacity is not None else DEFAULT_INITIAL_CAPACITY_THETA_PUDF
    best_val_loss_p, es_count_p, stats_p = float('inf'), 0, []
    patience_p = args.patience_early_stopping or DEFAULT_EARLY_STOPPING_PATIENCE_SFT
    overall_t_start_p, epochs_done_p = time.time(), 0

    raw_val_for_irt_eval = val_hf_raw  # This is the Dataset object for raw validation data
    if 'difficulty' not in raw_val_for_irt_eval.column_names: print(
        "Warn: PUDF eval val set (raw_val_for_irt_eval) no 'difficulty'. IRT on val may use init_theta.")

    pudf_outer_ordering_actual = args.pudf_ordering if args.pudf_ordering is not None else DEFAULT_PUDF_ORDERING_PARAM
    pudf_lower_b_actual = args.pudf_lower_bound if args.pudf_lower_bound is not None else DEFAULT_PUDF_LOWER_OFFSET
    pudf_upper_b_actual = args.pudf_upper_bound if args.pudf_upper_bound is not None else DEFAULT_PUDF_UPPER_OFFSET

    for epoch_p_idx in range(num_outer_epochs_p):
        epochs_done_p = epoch_p_idx + 1;
        ep_t_start = time.time()
        print(f"\nPUDF Outer E{epochs_done_p}/{num_outer_epochs_p}")
        val_loss_pre, _, acc_pre, est_theta_val, theta_t = perform_custom_sft_evaluation(
            model_to_eval=peft_model_pudf,
            raw_eval_dataset=raw_val_for_irt_eval,
            sft_processed_val_dataset_for_loss=sft_val_dataset_for_loss_pudf,
            tokenizer_eval=global_tokenizer,
            device_eval=DEVICE,
            prompt_style_eval=args.prompt_style,
            batch_size_eval=args.eval_batch_size or DEFAULT_QLORA_PER_DEVICE_EVAL_BS,
            eval_prompt_max_len=MAX_PROMPT_LEN_SFT,
            letter_token_ids_map=letter_token_ids_map_eval,
            max_new_toks_for_gen_accuracy=MAX_NEW_TOKENS_FOR_GEN_ACCURACY,
            desc_prefix=f"PUDF_E{epochs_done_p}_PreVal",
            current_irt_theta_init=cur_cap_theta_p,
            calculate_irt_theta_flag=True,
            data_collator_for_loss=data_collator_pudf,
            bf16_enabled_eval=bf16_enabled_pudf)

        theta_sel = est_theta_val if not np.isnan(est_theta_val) else cur_cap_theta_p
        if epoch_p_idx == 0 and args.difficulty_measurer != 'pudf_irt' and theta_sel < 1.0: theta_sel = 1.0
        print(
            f"  PUDF E{epochs_done_p}: CapSel={theta_sel:.4f} (est={est_theta_val:.4f}). PreValLoss={val_loss_pre:.4f}, Acc={acc_pre:.4f}")

        epoch_train_data = select_data_for_pudf_epoch(sft_train_full_dataset, theta_sel, 'difficulty',
                                                      pudf_outer_ordering_actual, pudf_lower_b_actual,
                                                      pudf_upper_b_actual, pudf_min_samps_actual, map_num_procs)
        n_sel_samps = len(epoch_train_data);
        avg_inner_loss = float('nan')
        if n_sel_samps > 0:
            req_cols_dl = ['input_ids', 'labels', 'attention_mask']
            actual_cols_epoch_data = [c for c in req_cols_dl if c in epoch_train_data.column_names]
            epoch_train_data.set_format(type='torch', columns=actual_cols_epoch_data)
            inner_dl = DataLoader(epoch_train_data, train_bs_p, collate_fn=data_collator_pudf, shuffle=True,
                                  num_workers=args.num_workers)
            peft_model_pudf.train();
            tot_inner_loss, grad_steps_inner, inner_step_iter = 0, 0, 0
            for step_i, batch_i in enumerate(tqdm(inner_dl, desc=f" InnerSFT E{epochs_done_p}", leave=False)):
                inner_step_iter = step_i + 1;
                batch_i = {k: v.to(DEVICE) for k, v in batch_i.items() if hasattr(v, 'to')}
                if not batch_i.get('input_ids', torch.empty(0)).numel(): continue
                with torch_amp_autocast(DEVICE.type, enabled=bf16_enabled_pudf):
                    out_i = peft_model_pudf(**batch_i);
                    loss_i = out_i.loss
                    if grad_acc_p > 1: loss_i /= grad_acc_p
                scaler_pudf.scale(loss_i).backward();
                tot_inner_loss += out_i.loss.item()
                if (step_i + 1) % grad_acc_p == 0 or (step_i + 1) == len(inner_dl):
                    scaler_pudf.unscale_(optimizer_pudf);
                    torch.nn.utils.clip_grad_norm_(peft_model_pudf.parameters(), 1.0)
                    scaler_pudf.step(optimizer_pudf);
                    scaler_pudf.update();
                    lr_scheduler_pudf.step();
                    optimizer_pudf.zero_grad();
                    grad_steps_inner += 1
            avg_inner_loss = tot_inner_loss / max(1, inner_step_iter)
            print(f"  PUDF E{epochs_done_p}: InnerLoss={avg_inner_loss:.4f} ({grad_steps_inner} steps)")
        else:
            print(f"  PUDF E{epochs_done_p}: No data selected. Skipping inner train.")

        if not np.isnan(est_theta_val) and est_theta_val <= cur_cap_theta_p and epoch_p_idx > 0:
            cur_cap_theta_p += 0.05
        elif not np.isnan(est_theta_val):
            cur_cap_theta_p = est_theta_val

        # Corrected: Pass raw_val_for_irt_eval to the post-training evaluation
        val_loss_post, _, acc_post, theta_post, _ = perform_custom_sft_evaluation(
            peft_model_pudf, raw_val_for_irt_eval, sft_val_dataset_for_loss_pudf, global_tokenizer, DEVICE,
            args.prompt_style,
            args.eval_batch_size or DEFAULT_QLORA_PER_DEVICE_EVAL_BS, MAX_PROMPT_LEN_SFT, letter_token_ids_map_eval,
            max_new_toks_for_gen_accuracy=MAX_NEW_TOKENS_FOR_GEN_ACCURACY,  # Use specific for accuracy
            desc_prefix=f"PUDF_E{epochs_done_p}_PostVal", current_irt_theta_init=cur_cap_theta_p,
            calculate_irt_theta_flag=True, data_collator_for_loss=data_collator_pudf,
            bf16_enabled_eval=bf16_enabled_pudf)
        print(f"  PUDF E{epochs_done_p}: PostValLoss={val_loss_post:.4f}, Acc={acc_post:.4f}, Theta={theta_post:.4f}")
        stats_p.append(
            {"epoch": epochs_done_p, "cap_sel": theta_sel, "n_train": n_sel_samps, "inner_loss": avg_inner_loss,
             "val_loss_pre": val_loss_pre, "val_acc_pre": acc_pre, "theta_est_pre": est_theta_val,
             "val_loss_post": val_loss_post, "val_acc_post": acc_post, "theta_est_post": theta_post,
             "duration": time.time() - ep_t_start, "theta_est_t": theta_t})

        current_metric_for_saving = val_loss_post
        if not np.isnan(current_metric_for_saving) and current_metric_for_saving < best_val_loss_p:
            print(f"  PUDF E{epochs_done_p}: New best val_loss: {current_metric_for_saving:.4f}. Saving.");
            best_val_loss_p = current_metric_for_saving;
            es_count_p = 0
            peft_model_pudf.save_pretrained(os.path.join(run_output_dir, "best_pudf_adapter"));
            global_tokenizer.save_pretrained(os.path.join(run_output_dir, "best_pudf_adapter"))
        elif not np.isnan(current_metric_for_saving):
            es_count_p += 1; print(f"  PUDF E{epochs_done_p}: No improvement. ES: {es_count_p}/{patience_p}")
        else:
            es_count_p += 1; print(f"  PUDF E{epochs_done_p}: Val loss NaN. ES: {es_count_p}/{patience_p}")
        if es_count_p >= patience_p: print(f"  PUDF Early stop E{epochs_done_p}."); break
        gc.collect();
        torch.cuda.empty_cache()

    tot_t_pudf = time.time() - overall_t_start_p;
    print(f"\nPUDF Training Finished. Time: {tot_t_pudf:.2f}s")
    test_acc_p = 0.0;
    best_adapter_p = os.path.join(run_output_dir, "best_pudf_adapter")
    if os.path.exists(best_adapter_p) and any(f.startswith("adapter_model") for f in os.listdir(best_adapter_p)):
        print(f"  Loading best PUDF adapter from {best_adapter_p} for test...")
        base_model_final_p = AutoModelForCausalLM.from_pretrained(args.model_id, quantization_config=bnb_config,
                                                                  torch_dtype=model_dtype_load, device_map="auto",
                                                                  trust_remote_code=True,
                                                                  attn_implementation="flash_attention_2",
                                                                  cache_dir=os.environ.get("TRANSFORMERS_CACHE"))
        if len(global_tokenizer) > base_model_final_p.config.vocab_size: base_model_final_p.resize_token_embeddings(
            len(global_tokenizer))
        if base_model_final_p.config.pad_token_id != global_tokenizer.pad_token_id and global_tokenizer.pad_token_id is not None: base_model_final_p.config.pad_token_id = global_tokenizer.pad_token_id
        final_peft_model_p = PeftModel.from_pretrained(base_model_final_p, best_adapter_p);
        final_peft_model_p.eval()
        _, _, test_acc_p, _, _ = perform_custom_sft_evaluation(final_peft_model_p, test_hf_raw, None, global_tokenizer,
                                                               DEVICE, args.prompt_style,
                                                               args.eval_batch_size or DEFAULT_QLORA_PER_DEVICE_EVAL_BS,
                                                               MAX_PROMPT_LEN_SFT, letter_token_ids_map_eval,
                                                               max_new_toks_for_gen_accuracy=MAX_NEW_TOKENS_FOR_GEN_ACCURACY,
                                                               # Use specific for accuracy
                                                               desc_prefix="FinalPUDFTest",
                                                               bf16_enabled_eval=bf16_enabled_pudf)
        print(f"  Final Test Accuracy (PUDF): {test_acc_p:.4f}")
    else:
        print(f"  Best PUDF adapter not found at {best_adapter_p}.")
    return {"best_val_metric (loss)": best_val_loss_p, "test_accuracy": test_acc_p, "training_time_s": tot_t_pudf,
            "epochs_completed": epochs_done_p, "detailed_epoch_stats": stats_p, "output_dir": run_output_dir}


def main():
    parser = argparse.ArgumentParser(description="Qwen MedQA QLoRA SFT Ablation Study Script")
    parser.add_argument('--difficulty_measurer', type=str, required=True,
                        choices=['pudf_irt', 'sentence_length', 'word_rarity'])
    parser.add_argument('--training_scheduler', type=str, required=True, choices=['linear', 'root', 'pudf_theta'])
    parser.add_argument('--output_dir_root', type=str, default="./qwen_medqa_ablation_sft_qlora")
    parser.add_argument('--pudf_difficulty_file', type=str, default=DEFAULT_PUDF_DIFFICULTY_FILE_PATH)
    parser.add_argument('--seed', type=int, default=RANDOM_SEED)
    parser.add_argument('--model_id', type=str, default=MODEL_ID)
    parser.add_argument('--dataset_id', type=str, default=DATASET_ID)
    parser.add_argument('--prompt_style', type=str, default='base', choices=['base', 'chat_template_qwen'],
                        help="IMPORTANT: 'base' for raw QA, 'chat_template_qwen' for Qwen chat models (VERIFY TEMPLATE).")
    parser.add_argument('--use_bf16', action='store_true', default=True)
    parser.add_argument('--num_epochs', type=int, default=None, help=f'Default: {DEFAULT_QLORA_NUM_EPOCHS}')
    parser.add_argument('--learning_rate', type=float, default=None, help=f'Default: {DEFAULT_QLORA_LEARNING_RATE}')
    parser.add_argument('--weight_decay', type=float, default=None, help=f'Default: {DEFAULT_QLORA_WEIGHT_DECAY}')
    parser.add_argument('--train_batch_size', type=int, default=None,
                        help=f'Default: {DEFAULT_QLORA_PER_DEVICE_TRAIN_BS}')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help=f'Default: {DEFAULT_QLORA_PER_DEVICE_EVAL_BS}')
    parser.add_argument('--grad_accum_steps', type=int, default=None, help=f'Default: {DEFAULT_QLORA_GRAD_ACCUM_STEPS}')
    parser.add_argument('--num_gpus', type=int, default=1, help='Num GPUs (basic DDP awareness).')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers & .map() num_proc.')
    parser.add_argument('--use_gradient_checkpointing', type=lambda x: (str(x).lower() == 'true'), default=True)
    parser.add_argument('--patience_early_stopping', type=int, default=None,
                        help=f'Default: {DEFAULT_EARLY_STOPPING_PATIENCE_SFT}')
    parser.add_argument('--logging_steps', type=int, default=None,
                        help=f'Trainer logging steps (default: {DEFAULT_LOGGING_STEPS_HEURISTIC}).')
    parser.add_argument('--qlora_r', type=int, default=None, help=f'LoRA r (default: {DEFAULT_QLORA_R}).')
    parser.add_argument('--qlora_alpha', type=int, default=None, help=f'LoRA alpha (default: {DEFAULT_QLORA_ALPHA}).')
    parser.add_argument('--qlora_dropout', type=float, default=None,
                        help=f'LoRA dropout (default: {DEFAULT_QLORA_DROPOUT}).')
    parser.add_argument('--qlora_target_modules', nargs='+', default=None,
                        help=f'LoRA target_modules (default: {DEFAULT_QLORA_TARGET_MODULES}).')
    parser.add_argument('--pudf_ordering', type=str, choices=['easiest', 'hardest'], default=None,
                        help=f'PUDF ordering (default: {DEFAULT_PUDF_ORDERING_PARAM}).')
    parser.add_argument('--pudf_num_obs_theta', type=int, default=None,
                        help=f'PUDF num_obs_theta (default: {DEFAULT_NUM_OBS_THETA_ESTIMATION_PUDF}).')
    parser.add_argument('--pudf_min_samples_per_epoch', type=int, default=None,
                        help=f'PUDF min_samples (default: {DEFAULT_PUDF_MIN_SAMPLES_PER_EPOCH}).')
    parser.add_argument('--pudf_lower_bound', type=float, default=None,
                        help=f'PUDF lower_bound (default: {DEFAULT_PUDF_LOWER_OFFSET}).')
    parser.add_argument('--pudf_upper_bound', type=float, default=None,
                        help=f'PUDF upper_bound (default: {DEFAULT_PUDF_UPPER_OFFSET}).')
    parser.add_argument('--pudf_initial_capacity', type=float, default=None,
                        help=f'PUDF initial_capacity (default: {DEFAULT_INITIAL_CAPACITY_THETA_PUDF}).')
    parser.add_argument('--heuristic_ordering', type=str, choices=['easiest', 'hardest'], default=None,
                        help=f'Heuristic ordering (default: {DEFAULT_HEURISTIC_ORDERING_PARAM}).')
    parser.add_argument('--competency_param_heuristic', type=float, default=None,
                        help=f'Heuristic competency_param (default: {DEFAULT_COMPETENCY_PARAM_HEURISTIC}).')
    parser.add_argument('--min_train_percent_heuristic', type=float, default=None,
                        help=f'Heuristic min_train_percent (default: {DEFAULT_MIN_TRAIN_PERCENT_HEURISTIC}).')
    parser.add_argument('--c_init_heuristic', type=float, default=None,
                        help=f'Heuristic c_init (default: {DEFAULT_C_INIT_HEURISTIC}).')
    args = parser.parse_args()
    if args.qlora_target_modules is None: args.qlora_target_modules = DEFAULT_QLORA_TARGET_MODULES

    run_dir = os.path.join(args.output_dir_root,
                           f"MedQA_{args.model_id.replace('/', '-')}_{args.difficulty_measurer}_{args.training_scheduler}_seed{args.seed}")
    setup_environment(run_dir);
    set_seed(args.seed)
    tok_cache = os.environ.get("TRANSFORMERS_CACHE")
    global_tokenizer = AutoTokenizer.from_pretrained(args.model_id, padding_side="left", trust_remote_code=True,
                                                     cache_dir=tok_cache)
    if global_tokenizer.pad_token is None or global_tokenizer.pad_token_id is None or \
            (
                    global_tokenizer.eos_token_id is not None and global_tokenizer.pad_token_id == global_tokenizer.eos_token_id):
        if global_tokenizer.eos_token and global_tokenizer.eos_token_id is not None:
            global_tokenizer.pad_token, global_tokenizer.pad_token_id = global_tokenizer.eos_token, global_tokenizer.eos_token_id
        else:
            pad_sym = "<|pad|>";
            global_tokenizer.add_special_tokens({"pad_token": pad_sym})
            if global_tokenizer.pad_token_id is None: global_tokenizer.pad_token_id = global_tokenizer.convert_tokens_to_ids(
                pad_sym)
        print(f" Global Tokenizer: Pad set to '{global_tokenizer.pad_token}' (ID: {global_tokenizer.pad_token_id})")
    if global_tokenizer.pad_token_id is None: raise ValueError("pad_token_id could not be set.")

    raw_ds = load_dataset(args.dataset_id, cache_dir=os.environ.get("HF_DATASETS_CACHE"))
    map_n_procs = args.num_workers if args.num_workers > 0 else None
    filt_ds = raw_ds.filter(lambda ex: ex["answer_idx"] and str(ex["answer_idx"]).strip().upper() in ANSWER_MAP_KEYS,
                            num_proc=map_n_procs, load_from_cache_file=False, desc="FiltAnsKey")
    if 'train' not in filt_ds or not filt_ds['train']: raise ValueError("MedQA 'train' empty post-filter.")

    t_min_diff, t_max_diff = -2.5, 2.5;
    irt_range_scores = None
    try:
        if os.path.exists(args.pudf_difficulty_file):
            irt_range_scores = load_pudf_irt_difficulties(args.pudf_difficulty_file, DIFFICULTY_JSON_KEY)
            if irt_range_scores.size > 0: t_min_diff, t_max_diff = irt_range_scores.min(), irt_range_scores.max()
            if t_min_diff == t_max_diff: t_min_diff -= 0.5; t_max_diff += 0.5
            if t_min_diff == t_max_diff: t_min_diff, t_max_diff = -1.0, 1.0
            print(f" IRT target scale: [{t_min_diff:.4f}, {t_max_diff:.4f}]")
        else:
            print(f" Warn: PUDF file {args.pudf_difficulty_file} not found. Default scale.")
    except Exception as e:
        print(f" Warn: Error loading IRT for scale: {e}. Default scale.")

    train_pool_raw = filt_ds['train']
    if args.difficulty_measurer == 'pudf_irt':
        if irt_range_scores is not None and len(irt_range_scores) == len(train_pool_raw):
            diff_vals = irt_range_scores.tolist()
        else:
            raise ValueError(
                f"PUDF IRT scores mismatch/err. Expected {len(train_pool_raw)}, got {len(irt_range_scores) if irt_range_scores is not None else 'None'}.")
    elif args.difficulty_measurer in ['sentence_length', 'word_rarity']:
        raw_h_scores = calculate_heuristic_difficulty_scores(train_pool_raw, args.difficulty_measurer,
                                                             'full_text_for_difficulty', map_n_procs)
        if not raw_h_scores and len(train_pool_raw) > 0: raise ValueError("Heuristic scoring empty for non-empty data.")
        scaled_h_scores = min_max_scale(np.array(raw_h_scores), t_min_diff, t_max_diff)
        diff_vals = scaled_h_scores.tolist();
        print(
            f" Scaled '{args.difficulty_measurer}' range: [{min(diff_vals) if diff_vals else 'N/A':.2f},{max(diff_vals) if diff_vals else 'N/A':.2f}]")
    else:
        raise ValueError(f"Unknown measurer: {args.difficulty_measurer}")

    train_full_hf = train_pool_raw.add_column("difficulty", diff_vals)
    print(f" Train pool with diff size: {len(train_full_hf)}. Cols: {train_full_hf.column_names}")

    val_hf_raw = filt_ds.get('validation')
    if not val_hf_raw or len(val_hf_raw) == 0:
        print(" Val set missing/empty. Splitting from train_full (10%)...");
        test_size_val_split = 0.1 if len(train_full_hf) >= 20 else (
            1 / len(train_full_hf) if len(train_full_hf) > 1 else 0)
        if test_size_val_split > 0 and len(train_full_hf) > 1:
            split = train_full_hf.train_test_split(test_size=test_size_val_split, seed=args.seed, shuffle=True)
            train_full_hf, val_hf_raw = split['train'], split['test']
            print(f" New train size: {len(train_full_hf)}, New val_hf_raw: {len(val_hf_raw)}")
        else:
            val_hf_raw = train_full_hf.select(range(min(10, len(train_full_hf)))) if len(
                train_full_hf) > 0 else train_full_hf.select([])
            print(
                f" Warn: Using {'first few of train' if len(val_hf_raw) > 0 else 'empty'} as val ({len(val_hf_raw)}).")

    test_hf_raw = filt_ds['test'];
    if not test_hf_raw or len(test_hf_raw) == 0: print("Warning: Test set is empty!")
    print(f" Final raw splits for eval: Train {len(train_full_hf)}, Val {len(val_hf_raw)}, Test {len(test_hf_raw)}")

    # Correctly build letter_tok_map to expect tokens like " A", " B" if that's what SFT targets
    letter_tok_map = {}
    for L in ANSWER_MAP_KEYS:
        # SFT format is "Answer: A<eos>", so the first token generated after "Answer:" should be for " A" (space + letter)
        # if the tokenizer treats space+letter as one token, or for " " if space and letter are separate.
        # The debug output shows " A" as a single token.
        # We need the token ID for the first token of " L" (space + letter)
        token_ids_with_space = global_tokenizer.encode(" " + L, add_special_tokens=False)
        if token_ids_with_space:  # Check if encoding is not empty
            letter_tok_map[L] = token_ids_with_space[0]  # The first token of " L"
        else:  # Fallback or error if " L" cannot be tokenized
            print(f"CRIT WARN! Could not tokenize ' {L}'. Accuracy will be affected.")

    if len(letter_tok_map) != NUM_CHOICES_MC:
        print(
            f"CRIT WARN! Letter token map incomplete: {letter_tok_map}. Accuracy will be affected. Expected {NUM_CHOICES_MC} keys.")
    print(f"DEBUG: letter_tok_map for eval (based on ' Letter'): {letter_tok_map}")

    summary = None
    if not train_full_hf or len(train_full_hf) == 0:
        print("ERROR: Training data pool is empty after all processing. Aborting.");
        summary = {"error": "Training data pool empty.", "test_accuracy": 0.0, "training_time_s": 0,
                   "epochs_completed": 0, "output_dir": run_dir}
    elif args.training_scheduler in ['linear', 'root']:
        summary = run_training_with_heuristic_qlora_sft(args, train_full_hf, val_hf_raw, test_hf_raw, global_tokenizer,
                                                        run_dir, letter_tok_map)
    elif args.training_scheduler == 'pudf_theta':
        summary = run_training_with_pudf_qlora_sft(args, train_full_hf, val_hf_raw, test_hf_raw, global_tokenizer,
                                                   run_dir, letter_tok_map)
    else:
        raise ValueError(f"Unsupported training_scheduler: {args.training_scheduler}")

    final_summary = summary

    if final_summary:
        summary_file_path = os.path.join(run_dir, "final_summary_medqa.json")
        with open(summary_file_path, 'w') as f:
            def convert_np(o):
                if isinstance(o, np.integer):
                    return int(o)
                elif isinstance(o, np.floating):
                    return float(o)
                elif isinstance(o, np.ndarray):
                    return o.tolist()
                return o

            serializable_summary = json.loads(json.dumps(final_summary, default=convert_np))
            json.dump(serializable_summary, f, indent=4)
        print(f"Summary saved to: {summary_file_path}")  # Corrected NameError here
    print(f"\nMedQA Ablation Run Finished: {args.difficulty_measurer} + {args.training_scheduler}")


if __name__ == "__main__":
    # Needed for gradient_checkpointing_kwargs in TrainingArguments
    import packaging.version

    main()