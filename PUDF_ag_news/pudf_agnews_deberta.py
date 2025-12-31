# pudf_agnews_deberta_manual_loss_final.py
# -*- coding: utf-8 -*-
import numpy as np
import random
import os
import time
import json
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset, DatasetDict, Dataset
import evaluate
import types
import gc
import traceback
from tqdm.auto import tqdm

# Assuming these custom modules are in the same directory or Python path
from build_features_minimal_pudf import get_epoch_training_data
from irt_scoring import calculate_theta

# --- AG News Specific Constants ---
AG_NEWS_DIFFICULTY_FILE_PATH = "../gen_difficulty/merged_jsonlines_output/test-1pl/best_parameters.json"
AG_NEWS_MAX_LENGTH = 128
DIFFICULTY_JSON_KEY = "diff"

transformers.logging.set_verbosity_error()


# --- PUDF Helper Functions ---

def create_pudf_tensor_dataset(hf_tokenized_dataset_split,
                               partition_name="unknown",
                               include_difficulty=True):
    print(f"Creating TensorDataset for AG News {partition_name} (include_difficulty={include_difficulty})...")

    print(f"  DEBUG ({partition_name}): Input HF Dataset columns: {hf_tokenized_dataset_split.column_names}")
    if 'labels' in hf_tokenized_dataset_split.column_names and hf_tokenized_dataset_split['labels'] is not None and len(hf_tokenized_dataset_split['labels']) > 0 :
        print(
            f"  DEBUG ({partition_name}): Sample of 'labels' column before set_format: {hf_tokenized_dataset_split['labels'][:3]}")
        if 'labels' in hf_tokenized_dataset_split.features:
            label_feature_in_func = hf_tokenized_dataset_split.features['labels']
            print(f"  DEBUG ({partition_name}): Feature of 'labels' column before set_format: {label_feature_in_func}")
        else:
            print(f"  DEBUG ({partition_name}): 'labels' feature not found in hf_tokenized_dataset_split.features.")
    else:
        print(f"  DEBUG ({partition_name}): 'labels' column seems to be missing or empty before set_format.")


    final_columns = ['input_ids', 'attention_mask']
    if 'token_type_ids' in hf_tokenized_dataset_split.column_names and 'token_type_ids' in hf_tokenized_dataset_split.features:
        final_columns.append('token_type_ids')
        print(f"  INFO ({partition_name}): 'token_type_ids' will be included in TensorDataset.")
    else:
        print(f"  INFO ({partition_name}): 'token_type_ids' will NOT be included in TensorDataset.")

    if 'labels' not in hf_tokenized_dataset_split.column_names:
        raise ValueError(f"'labels' column missing in {partition_name} for TensorDataset creation.")
    final_columns.append('labels')

    if include_difficulty:
        if 'difficulty' not in hf_tokenized_dataset_split.column_names:
            raise RuntimeError(
                f"Difficulty column required for {partition_name} (include_difficulty=True) but missing.")
        final_columns.append('difficulty')
    else:
        if 'difficulty' in hf_tokenized_dataset_split.column_names:
            print(
                f"Note: 'difficulty' column present in {partition_name} but EXCLUDED from TensorDataset as per include_difficulty=False.")
    try:
        hf_tokenized_dataset_split.set_format(type='torch', columns=final_columns)

        tensors_to_extract = []
        for col_name in final_columns:
            tensor_data = hf_tokenized_dataset_split[col_name]
            if col_name == 'labels':
                print(
                    f"  INFO ({partition_name}): 'labels' tensor from set_format has shape: {tensor_data.shape}, dtype: {tensor_data.dtype}")
                if tensor_data.ndim > 1:
                    print(
                        f"  INFO ({partition_name}): 'labels' tensor is {tensor_data.ndim}D. Attempting to make it 1D by squeeze(-1).")
                    if tensor_data.shape[-1] == 1:
                        tensor_data = tensor_data.squeeze(-1)
                    print(f"    Shape after potential squeeze: {tensor_data.shape}")
                if tensor_data.ndim != 1:
                    raise ValueError(
                        f"Labels tensor for {partition_name} is not 1D after processing. Final shape: {tensor_data.shape}")
                tensor_data = tensor_data.long()
            tensors_to_extract.append(tensor_data)

        print(f"  Shapes of tensors for TensorDataset ({partition_name}):")
        for i, col_n in enumerate(final_columns): print(f"    {col_n}: {tensors_to_extract[i].shape}")
        print(f"Final tensor order for {partition_name} TensorDataset: {final_columns}")
        return TensorDataset(*tensors_to_extract), final_columns
    except Exception as e:
        print(f"ERROR creating TensorDataset for {partition_name}: {e}")
        traceback.print_exc()
        raise


def evaluate_and_estimate_agnews(model, dataloader, device, num_labels, column_order,
                                 num_obs_theta=-1, mode='eval', desc_prefix=""):
    accuracy_metric_eval_local = evaluate.load("accuracy", cache_dir=os.environ.get("HF_DATASETS_CACHE",
                                                                                    "./hf_cache/datasets_eval"))
    total_loss = 0.0
    all_preds_logits_list = []
    all_labels_list = []
    all_difficulties_list = []
    model.eval()
    num_batches = 0

    try:
        input_ids_idx = column_order.index('input_ids')
        attention_mask_idx = column_order.index('attention_mask')
        token_type_ids_idx = column_order.index('token_type_ids') if 'token_type_ids' in column_order else -1
        labels_idx = column_order.index('labels')
        difficulty_idx = column_order.index('difficulty') if 'difficulty' in column_order else -1
    except ValueError as e:
        print(f"FATAL ERROR in ({desc_prefix}) evaluate_and_estimate_agnews: Column missing in expected order {column_order}. Error: {e}")
        traceback.print_exc()
        raise

    dataset_has_difficulty_column = (difficulty_idx != -1)
    if mode in ['estimate', 'eval_estimate'] and not dataset_has_difficulty_column:
        print(f"PUDF INFO ({desc_prefix}): Mode '{mode}' for theta estimation, but 'difficulty' column not found in provided data (column_order: {column_order}). Theta will default to 0.0.")

    pbar_desc = f"{desc_prefix} Eval"
    if mode == 'estimate':
        pbar_desc = f"{desc_prefix} Theta Estimation"
    elif mode == 'eval_estimate':
        pbar_desc = f"{desc_prefix} Eval & Theta Est"

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=pbar_desc, leave=False)):
        num_batches += 1
        try:
            input_ids = batch[input_ids_idx].to(device, non_blocking=True)
            attention_mask = batch[attention_mask_idx].to(device, non_blocking=True)
            token_type_ids_batch = None
            if token_type_ids_idx != -1:
                token_type_ids_batch = batch[token_type_ids_idx].to(device, non_blocking=True)
            labels_batch = batch[labels_idx].to(device, non_blocking=True)
            if labels_batch.ndim > 1 and labels_batch.shape[-1] == 1:
                labels_batch = labels_batch.squeeze(-1)
            elif labels_batch.ndim > 1:
                 print(f"Warning ({desc_prefix}): labels_batch for batch {batch_idx} has shape {labels_batch.shape} which is not 1D or squeezable to 1D.")
            labels_batch = labels_batch.long()
            difficulty_tensor_for_batch = None
            if dataset_has_difficulty_column:
                difficulty_tensor_for_batch = batch[difficulty_idx].cpu().numpy()
        except IndexError as e:
            print(f"ERROR ({desc_prefix}, batch {batch_idx}): IndexError unpacking batch (len={len(batch)}) using {column_order}. Error: {e}")
            traceback.print_exc()
            continue
        except Exception as e:
            print(f"ERROR ({desc_prefix}, batch {batch_idx}): Unexpected error unpacking batch: {e}")
            traceback.print_exc()
            continue

        with torch.no_grad():
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids_batch)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, num_labels), labels_batch.view(-1))
            except Exception as e:
                print(f"ERROR ({desc_prefix}, batch {batch_idx}): Model forward/loss calculation failed: {e}")
                print(f"  Shapes: input_ids: {input_ids.shape}, attention_mask: {attention_mask.shape}, labels: {labels_batch.shape}" +
                      (f", token_type_ids: {token_type_ids_batch.shape}" if token_type_ids_batch is not None else ""))
                traceback.print_exc()
                continue
        total_loss += loss.item()
        all_preds_logits_list.append(logits.detach().cpu().numpy())
        all_labels_list.append(labels_batch.detach().cpu().numpy())
        if dataset_has_difficulty_column and difficulty_tensor_for_batch is not None:
            all_difficulties_list.append(difficulty_tensor_for_batch)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    final_preds_logits = np.concatenate(all_preds_logits_list) if all_preds_logits_list else np.empty((0, num_labels))
    final_labels = np.concatenate(all_labels_list) if all_labels_list else np.empty(0, dtype=int)
    accuracy = 0.0
    if final_labels.size > 0 and final_preds_logits.size > 0 and final_preds_logits.shape[0] == final_labels.shape[0]:
        predictions_for_metric = np.argmax(final_preds_logits, axis=1)
        accuracy = accuracy_metric_eval_local.compute(predictions=predictions_for_metric, references=final_labels)['accuracy']
    elif final_labels.size == 0 and not all_preds_logits_list :
        print(f"Warning ({desc_prefix}): No data processed for metrics calculation.")
    else:
        print(f"Warning ({desc_prefix}): Could not compute accuracy. Predictions size: {final_preds_logits.shape}, Labels size: {final_labels.shape}")

    if mode == 'eval' or mode == 'eval_test_no_diff':
        return accuracy, avg_loss

    theta_hat, model_capacity_time = 0.0, 0.0
    if not dataset_has_difficulty_column:
        if mode in ['estimate', 'eval_estimate']:
            pass # Already warned
    elif not all_difficulties_list:
        if mode in ['estimate', 'eval_estimate']:
            print(f"Warning ({desc_prefix}): Mode '{mode}', 'difficulty' column (idx={difficulty_idx}) was expected, but no scores collected. Theta defaults to 0.0.")
    else:
        concatenated_difficulties = np.concatenate(all_difficulties_list)
        if len(concatenated_difficulties) != len(final_labels):
            print(f"ERROR ({desc_prefix}): Mismatch len diff ({len(concatenated_difficulties)}) / resp ({len(final_labels)}). Theta est skipped.")
        elif len(concatenated_difficulties) == 0 :
            print(f"Warning ({desc_prefix}): No data for theta est (concatenated_difficulties empty).")
        else:
            time_model_s = time.time()
            if final_preds_logits.shape[0] == final_labels.shape[0] and final_labels.size > 0 :
                predictions_for_metric_theta = np.argmax(final_preds_logits, axis=1)
                response_pattern_irt = (predictions_for_metric_theta == final_labels).astype(int) * 2 - 1
                num_total_obs = len(concatenated_difficulties)
                indices_for_theta = np.arange(num_total_obs)
                if num_obs_theta > 0 and num_obs_theta < num_total_obs:
                    indices_for_theta = np.random.choice(num_total_obs, num_obs_theta, replace=False)
                try:
                    theta_hat = calculate_theta(concatenated_difficulties[indices_for_theta], response_pattern_irt[indices_for_theta])[0]
                except Exception as e:
                    print(f"ERROR ({desc_prefix}) theta calc: {e}. Theta=0.")
                    traceback.print_exc()
                    theta_hat = 0.0
                model_capacity_time = time.time() - time_model_s
            else:
                print(f"Warning ({desc_prefix}): Skipping theta calculation due to lack of valid predictions/labels for response pattern.")

    if mode == 'estimate':
        return theta_hat, model_capacity_time
    if mode == 'eval_estimate':
        return accuracy, avg_loss, theta_hat, model_capacity_time
    print(f"Warning ({desc_prefix}): Unrecognized mode '{mode}' in eval func. Returning acc, loss.")
    return accuracy, avg_loss


def train_agnews_pudf_core(config, model, tokenizer, num_labels,
                           hf_actual_train_tokenized,
                           hf_val_tokenized,
                           hf_test_tokenized,
                           hf_theta_estimation_tokenized,
                           output_dir_base):
    task_output_dir = os.path.join(output_dir_base, config.task_name_for_pudf)
    best_model_dir = os.path.join(task_output_dir, "best_model")
    os.makedirs(task_output_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() and config.gpu >= 0 else 'cpu')
    model.to(device)
    use_amp = torch.cuda.is_available() and device.type == 'cuda'
    if hasattr(torch, 'compile') and use_amp and config.use_torch_compile:
        try:
            print("Attempting torch.compile...")
            model = torch.compile(model)
            print("Model compiled.")
        except Exception as e:
            print(f"Compile failed: {e}. Continuing.")

    print("Creating PUDF TensorDatasets from tokenized HF Datasets...")
    train_column_order, val_column_order, test_column_order, theta_estimation_column_order = None, None, None, None
    try:
        train_tensordataset, train_column_order = create_pudf_tensor_dataset(
            hf_actual_train_tokenized, "train_main_pool", include_difficulty=True)

        main_val_set_has_difficulty = 'difficulty' in hf_val_tokenized.column_names
        val_tensordataset, val_column_order = create_pudf_tensor_dataset(
            hf_val_tokenized, "validation_main", include_difficulty=main_val_set_has_difficulty)
        if not main_val_set_has_difficulty:
            print("INFO: Main validation set (for acc/loss) does not have 'difficulty' column. Theta from it will be 0.")

        theta_estimation_tensordataset, theta_estimation_column_order = create_pudf_tensor_dataset(
            hf_theta_estimation_tokenized, "theta_estimation_subset", include_difficulty=True)

        test_tensordataset, test_column_order = create_pudf_tensor_dataset(
            hf_test_tokenized, "test", include_difficulty=False)
    except Exception as e:
        print(f"FATAL: PUDF TensorDataset creation failed: {e}")
        traceback.print_exc()
        return 0.0, 0.0, 0.0

    dl_kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if device.type == 'cuda' else {}
    val_dataloader = DataLoader(val_tensordataset, batch_size=config.batch_size, shuffle=False, **dl_kwargs)
    theta_estimation_dataloader = DataLoader(theta_estimation_tensordataset, batch_size=config.batch_size, shuffle=False, **dl_kwargs)
    test_dataloader = DataLoader(test_tensordataset, batch_size=config.batch_size, shuffle=False, **dl_kwargs) if len(
        test_tensordataset) > 0 else None

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01, eps=1e-6, betas=(0.9, 0.999))
    num_est_steps_full_epoch = len(train_tensordataset) // config.batch_size if len(train_tensordataset) > 0 else 1
    if num_est_steps_full_epoch == 0 and len(train_tensordataset) > 0: num_est_steps_full_epoch = 1
    total_estimated_steps = num_est_steps_full_epoch * config.num_epochs
    num_warmup_steps = max(1, int(0.06 * total_estimated_steps)) if total_estimated_steps > 0 else 0
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=max(1, total_estimated_steps))
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_accuracy = 0.0
    early_stop_counter = 0
    training_stats_list = []
    total_pudf_overhead_time = 0.0
    prev_cap = -5.0
    cur_cap = 0.0
    estimated_theta_for_guidance = 0.0

    print(
        f"\nStarting PUDF AG News training loop: Max {config.num_epochs} epochs, Patience {config.patience_early_stopping}...")
    overall_training_start_time = time.time()
    actual_epochs_run = 0
    loss_fct_train = torch.nn.CrossEntropyLoss()

    for epoch in range(config.num_epochs):
        actual_epochs_run = epoch + 1
        print(f"\n======== Epoch {epoch + 1} / {config.num_epochs} ========")
        epoch_start_time = time.time()
        avg_train_loss_epoch = 0.0
        model_capacity_time_est_epoch = 0.0
        filter_time_epoch = 0.0

        if config.strategy == 'theta':
            print("Estimating capacity (theta) from dedicated training subset...")
            try:
                estimated_theta_for_guidance, capacity_estimation_time = evaluate_and_estimate_agnews(
                    model, theta_estimation_dataloader, device, num_labels,
                    theta_estimation_column_order,
                    num_obs_theta=config.num_obs_theta,
                    mode='estimate',
                    desc_prefix=f"Epoch {epoch + 1} CapEst (on Train Subset)"
                )
                model_capacity_time_est_epoch = capacity_estimation_time
                total_pudf_overhead_time += model_capacity_time_est_epoch
                print(f"Theta for curriculum guidance (from train subset): {estimated_theta_for_guidance:.4f} ({model_capacity_time_est_epoch:.2f}s)")

                if estimated_theta_for_guidance > prev_cap:
                    cur_cap = estimated_theta_for_guidance
                else:
                    cur_cap = prev_cap + 0.1 if prev_cap > -4.9 else 0.1
                    print(f"  Theta_guidance ({estimated_theta_for_guidance:.4f}) not > prev_cap ({prev_cap:.4f}). Adjusted cur_cap for filtering: {cur_cap:.4f}")
            except Exception as e:
                print(f"Warning: Capacity estimation on train subset failed: {e}. Using cur_cap={cur_cap:.4f}")
                traceback.print_exc()
                if prev_cap == -5.0 and cur_cap == 0.0:
                     cur_cap = 0.1
                elif cur_cap <= prev_cap :
                     cur_cap = prev_cap + 0.1 if prev_cap > -4.9 else 0.1
        else:
            cur_cap = np.inf
            estimated_theta_for_guidance = 'N/A'
            print(f"Strategy is not 'theta'. Using cur_cap = {cur_cap} for filtering.")

        print(f"Filtering training data (Capacity for filtering: {cur_cap:.4f})...")
        filter_time_s = time.time()
        try:
            filtered_training_data_dict = get_epoch_training_data(
                train_tensordataset, config, epoch,
                config.task_name_for_pudf, cur_cap,
                diffs_sorted_idx=None,
                lower_offset=config.lower_bound,
                upper_offset=config.upper_bound
            )
        except Exception as e:
            print(f"ERROR: get_epoch_training_data failed: {e}. Skipping epoch training.")
            traceback.print_exc()
            continue
        filter_time_epoch = time.time() - filter_time_s
        total_pudf_overhead_time += filter_time_epoch

        num_epoch_training_samples = len(filtered_training_data_dict.get('labels', []))
        if num_epoch_training_samples == 0:
            print("Warning: No data selected for training this epoch. Skipping training phase.")
            avg_train_loss_epoch = 0.0
        else:
            print(f"Selected {num_epoch_training_samples} examples for epoch training ({filter_time_epoch:.2f}s filtering).")
            try:
                tensors_for_epoch_dl = [filtered_training_data_dict['input_ids'],
                                        filtered_training_data_dict['attention_mask']]
                epoch_has_token_type_ids = 'token_type_ids' in filtered_training_data_dict and \
                                           train_column_order is not None and \
                                           'token_type_ids' in train_column_order
                if epoch_has_token_type_ids:
                    tensors_for_epoch_dl.append(filtered_training_data_dict['token_type_ids'])
                tensors_for_epoch_dl.extend([filtered_training_data_dict['labels'],
                                             filtered_training_data_dict['difficulty']])
                train_dataset_epoch_pytorch = TensorDataset(*tensors_for_epoch_dl)
                epoch_dataloader_batch_size = min(config.batch_size, num_epoch_training_samples) if num_epoch_training_samples > 0 else config.batch_size
                train_dataloader_epoch = DataLoader(train_dataset_epoch_pytorch, shuffle=True,
                                                    batch_size=epoch_dataloader_batch_size, **dl_kwargs)
            except KeyError as e:
                 print(f"ERROR creating epoch dataloader: Key {e} missing. Skipping training.")
                 traceback.print_exc()
                 continue
            except Exception as e:
                print(f"ERROR creating epoch dataloader: {e}. Skipping training.")
                traceback.print_exc()
                continue

            model.train()
            current_epoch_loss_sum = 0.0
            num_epoch_train_batches = 0
            optimizer.zero_grad(set_to_none=True)
            pbar_train = tqdm(train_dataloader_epoch, desc=f"Epoch {epoch + 1} Training", leave=False)
            for step, batch_train_epoch in enumerate(pbar_train):
                input_ids_tr = batch_train_epoch[0].to(device, non_blocking=True)
                attention_mask_tr = batch_train_epoch[1].to(device, non_blocking=True)
                current_idx_tr = 2
                token_type_ids_tr = None
                if epoch_has_token_type_ids:
                    token_type_ids_tr = batch_train_epoch[current_idx_tr].to(device, non_blocking=True)
                    current_idx_tr += 1
                labels_tr = batch_train_epoch[current_idx_tr].to(device, non_blocking=True)
                if labels_tr.ndim > 1 and labels_tr.shape[-1] == 1: labels_tr = labels_tr.squeeze(-1)
                labels_tr = labels_tr.long()
                try:
                    with autocast(device_type=device.type, enabled=use_amp):
                        outputs_tr = model(input_ids=input_ids_tr, attention_mask=attention_mask_tr, token_type_ids=token_type_ids_tr)
                        logits_tr = outputs_tr.logits
                        loss = loss_fct_train(logits_tr.view(-1, num_labels), labels_tr.view(-1))
                    if torch.isnan(loss) or loss is None:
                        print(f"Warning: NaN/None loss at step {step}. Skipping optimizer step.")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    current_epoch_loss_sum += loss.item()
                    pbar_train.set_postfix({'loss': f'{loss.item():.4f}'})
                    num_epoch_train_batches += 1
                except Exception as e:
                    print(f"ERROR training step {step}: {e}")
                    optimizer.zero_grad(set_to_none=True)
                    traceback.print_exc()
                    print("Attempting to skip step and continue epoch.")
                    continue
            avg_train_loss_epoch = current_epoch_loss_sum / num_epoch_train_batches if num_epoch_train_batches > 0 else 0.0
            print(f"Epoch {epoch + 1} Avg Training Loss: {avg_train_loss_epoch:.4f}")

        print("Evaluating on main validation set (for acc/loss)...")
        try:
            dev_accuracy_epoch, val_loss_epoch, theta_on_main_val, model_capacity_time_main_val_eval = evaluate_and_estimate_agnews(
                model, val_dataloader, device, num_labels,
                val_column_order,
                num_obs_theta=config.num_obs_theta, mode='eval_estimate',
                desc_prefix=f"Epoch {epoch + 1} MainVal"
            )

            print(
                f"Epoch {epoch + 1} Main Validation: Acc={dev_accuracy_epoch:.4f}, Loss={val_loss_epoch:.4f}, Theta(MainVal)={theta_on_main_val:.4f} ({model_capacity_time_main_val_eval:.2f}s for its theta part)")
            if theta_on_main_val == 0.0 and not main_val_set_has_difficulty:
                 print("    Note: Theta from main validation set is 0.0 (as expected, no 'difficulty' column or issue).")

            if config.strategy == 'theta':
                prev_cap = estimated_theta_for_guidance

            training_stats_list.append({
                'epoch': epoch + 1, 'Train Loss': avg_train_loss_epoch, 'Val Loss': val_loss_epoch,
                'Val Acc': dev_accuracy_epoch,
                'cur_cap_for_filter': cur_cap if config.strategy == 'theta' else 'N/A',
                'theta_guidance_from_train_subset': estimated_theta_for_guidance if config.strategy == 'theta' else 'N/A',
                'theta_on_main_val_set': theta_on_main_val,
                'pudf_cap_est_time_epoch': model_capacity_time_est_epoch,
                'pudf_filter_time_epoch': filter_time_epoch,
                'pudf_main_val_theta_time_epoch': model_capacity_time_main_val_eval,
                'n_train_samples': num_epoch_training_samples
            })

            if dev_accuracy_epoch > best_val_accuracy:
                print(f"Validation accuracy improved ({best_val_accuracy:.4f} --> {dev_accuracy_epoch:.4f}). Saving model...")
                best_val_accuracy = dev_accuracy_epoch
                early_stop_counter = 0
                model_to_save = getattr(model, '_orig_mod', model)
                model_to_save.save_pretrained(best_model_dir)
                tokenizer.save_pretrained(best_model_dir)
                serializable_config_snapshot = {k: (str(v) if v == -np.inf else v) if not isinstance(v, (
                list, dict, int, float, bool, type(None))) else v for k, v in config.__dict__.items()}
                with open(os.path.join(best_model_dir, "training_config_summary.json"), "w") as fconf:
                    json.dump(serializable_config_snapshot, fconf, indent=4)
                if os.path.exists(os.path.join(best_model_dir, "pytorch_model.bin")) or \
                   os.path.exists(os.path.join(best_model_dir, "model.safetensors")):
                    print("Model save confirmed.")
                else:
                    print(f"Warning: Model weights file not found in {best_model_dir}.")
            else:
                early_stop_counter += 1
                print(
                    f"Validation accuracy ({dev_accuracy_epoch:.4f}) did not improve vs best ({best_val_accuracy:.4f}). Early stop count: {early_stop_counter}/{config.patience_early_stopping}")
                if early_stop_counter >= config.patience_early_stopping:
                    print("Early stopping triggered.")
                    break
        except Exception as e:
            print(f"ERROR during validation or saving for epoch {epoch + 1}: {e}")
            traceback.print_exc()
            print("Stopping training for this task due to error in validation phase.")
            break

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"Epoch {epoch + 1} duration: {time.time() - epoch_start_time:.2f}s.")
    # END OF FOR EPOCH LOOP

    end_train_loop_time = time.time()
    train_loop_duration = end_train_loop_time - overall_training_start_time
    print("\n--- Training Loop Finished ---")
    print(
        f"Actual epochs run: {actual_epochs_run}, Total Time: {train_loop_duration:.2f}s, PUDF Overhead (cap_est + filter): {total_pudf_overhead_time:.2f}s, Best Val Acc: {best_val_accuracy:.4f}")

    stats_filename_base = f"{config.model_name.split('/')[-1]}_{config.task_name_for_pudf}"
    training_stats_filename = os.path.join(task_output_dir, f"training_stats_{stats_filename_base}.json")
    try:
        with open(training_stats_filename, "w") as f:
            json.dump(training_stats_list, f, indent=4, default=str)
        print(f"Training stats saved: {training_stats_filename}")
    except Exception as e:
        print(f"Warning: Error saving training stats: {e}")

    print("\n--- Final Test Evaluation ---")
    test_acc_final, test_loss_final = 0.0, 0.0
    if test_dataloader is None:
        print("Test dataloader is None. Skipping final test evaluation.")
        test_acc_final, test_loss_final = -3.0, -3.0
    elif os.path.isdir(best_model_dir) and \
         (os.path.exists(os.path.join(best_model_dir, "pytorch_model.bin")) or \
          os.path.exists(os.path.join(best_model_dir, "model.safetensors"))):
        print(f"Loading best model from: {best_model_dir} for final test evaluation...")
        try:
            model_loaded = AutoModelForSequenceClassification.from_pretrained(best_model_dir)
            model_loaded.to(device)
            if hasattr(torch, 'compile') and use_amp and config.use_torch_compile:
                try:
                    model_loaded = torch.compile(model_loaded)
                    print("Loaded best model compiled for testing.")
                except Exception: # Broad except for compile failure
                    print("Failed to compile loaded best model for testing.")
            test_acc_final, test_loss_final = evaluate_and_estimate_agnews(
                model_loaded, test_dataloader, device, num_labels,
                test_column_order,
                mode='eval_test_no_diff',
                desc_prefix="Final Test"
            )
            print(f'Final Test Accuracy: {test_acc_final:.4f}, Final Test Loss: {test_loss_final:.4f}')
            del model_loaded
        except Exception as e:
            print(f"ERROR during final test evaluation: {e}")
            traceback.print_exc()
            test_acc_final, test_loss_final = -1.0, -1.0
    else:
        print(f"Best model directory or weights file not found in {best_model_dir}. Cannot run final test evaluation.")
        test_acc_final, test_loss_final = -2.0, -2.0

    final_stats_filename = os.path.join(task_output_dir,
                                        f"final_stats_{stats_filename_base}_ValAcc_{best_val_accuracy:.4f}_TestAcc_{test_acc_final:.4f}.json")
    serializable_config = {
        k: (str(v) if v == -np.inf else v) if not isinstance(v, (list, dict, int, float, bool, type(None))) else v for
        k, v in config.__dict__.items()}
    final_summary = {"task": config.task_name_for_pudf, "model": config.model_name.split('/')[-1],
                     "strategy": config.strategy, "ordering": config.ordering, "lower_bound": config.lower_bound,
                     "upper_bound": config.upper_bound, "min_train_length": config.min_train_length,
                     "num_obs_theta": config.num_obs_theta, "num_epochs_set": config.num_epochs,
                     "num_epochs_run": actual_epochs_run, "best_validation_accuracy": best_val_accuracy,
                     "final_test_accuracy": test_acc_final, "final_test_loss": test_loss_final,
                     "total_training_loop_time_seconds": round(train_loop_duration, 2),
                     "total_pudf_overhead_seconds": round(total_pudf_overhead_time, 2),
                     "config_snapshot_at_best_val": serializable_config,
                     "training_stats_summary_path": training_stats_filename,
                     "best_model_path": best_model_dir
                     }
    try:
        with open(final_stats_filename, "w") as f:
            json.dump(final_summary, f, indent=4, default=str)
        print(f"Final summary results saved to: {final_stats_filename}")
    except Exception as e:
        print(f"Warning: Error saving final summary JSON: {e}")

    try:
        del model, tokenizer, optimizer, scheduler, scaler
        del train_tensordataset, val_tensordataset, test_tensordataset, theta_estimation_tensordataset
        del val_dataloader, test_dataloader, theta_estimation_dataloader
        if 'train_dataloader_epoch' in locals(): del train_dataloader_epoch
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print(f"===== Finished Task: {config.task_name_for_pudf} =====")
    return best_val_accuracy, test_acc_final, test_loss_final


# --- Main Run Function ---
def run_agnews_pudf_with_user_setup():
    # Import paths from central config file
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HF_HOME, GLUE_DIFFICULTY_DIR, MEDQA_DIFFICULTY_FILE
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.makedirs(os.environ["HF_DATASETS_CACHE"], exist_ok=True)
    script_random_seed = 63
    torch.manual_seed(script_random_seed)
    np.random.seed(script_random_seed)
    random.seed(script_random_seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(script_random_seed)
    print(f"Global random seed set to: {script_random_seed}")

    # Config object needed early for min_train_length for split warning
    config = types.SimpleNamespace()
    config.min_train_length = 500 # Default, will be set properly later before training

    print("Loading dataset: contemmcm/ag_news...")
    _dataset_from_hf = load_dataset("contemmcm/ag_news", cache_dir=os.environ["HF_DATASETS_CACHE"])
    complete_dataset_renamed = _dataset_from_hf['complete']
    if 'news_story' in complete_dataset_renamed.column_names and 'labeling' in complete_dataset_renamed.column_names:
        print("Renaming 'news_story'->'text', 'labeling'->'label'.")
        complete_dataset_renamed = complete_dataset_renamed.rename_column('news_story', 'text')
        complete_dataset_renamed = complete_dataset_renamed.rename_column('labeling', 'label')
    elif 'text' not in complete_dataset_renamed.column_names or 'label' not in complete_dataset_renamed.column_names:
        raise ValueError("Cannot find text/label columns.")

    print("Splitting 'complete' into train_full/val/test...")
    train_temp_split = complete_dataset_renamed.train_test_split(test_size=0.2, seed=script_random_seed, shuffle=True)
    user_train_dataset_full = train_temp_split['train']
    temp_dataset = train_temp_split['test']
    val_test_split = temp_dataset.train_test_split(test_size=0.5, seed=script_random_seed, shuffle=True)
    user_validation_dataset = val_test_split['train']
    user_test_dataset = val_test_split['test']

    print(f"Dataset sizes after initial split: Full Train Pool={len(user_train_dataset_full)}, Main Val={len(user_validation_dataset)}, Test={len(user_test_dataset)}")

    print(f"Loading {DIFFICULTY_JSON_KEY} scores from: {AG_NEWS_DIFFICULTY_FILE_PATH}")
    try:
        difficulty_file_abs_path = os.path.abspath(AG_NEWS_DIFFICULTY_FILE_PATH)
        print(f"Attempting to load difficulty scores from absolute path: {difficulty_file_abs_path}")
        with open(difficulty_file_abs_path, 'r') as file:
            difficulty_data_json = json.load(file)
        difficulty_scores_list = difficulty_data_json[DIFFICULTY_JSON_KEY]
        if len(difficulty_scores_list) != len(user_train_dataset_full):
            raise ValueError(
                f"Diff score count ({len(difficulty_scores_list)}) != full train_dataset size ({len(user_train_dataset_full)}). Ensure difficulty file matches the 'train' portion of the initial 0.2 split.")
        if 'difficulty' in user_train_dataset_full.column_names:
            user_train_dataset_full = user_train_dataset_full.remove_columns(['difficulty'])
        user_train_dataset_full_with_diff = user_train_dataset_full.add_column('difficulty', difficulty_scores_list)
        print("Diff scores added to the full training pool.")
    except Exception as e:
        print(f"FATAL: Error loading/adding diff scores: {e}")
        traceback.print_exc()
        return

    theta_estimation_set_size = 10000
    if len(user_train_dataset_full_with_diff) <= theta_estimation_set_size:
        raise ValueError(f"Full training dataset pool size ({len(user_train_dataset_full_with_diff)}) is too small to create a theta estimation set of size {theta_estimation_set_size}.")
    if (len(user_train_dataset_full_with_diff) - theta_estimation_set_size) < config.min_train_length: # Use temp config for this check
         print(f"Warning: After taking {theta_estimation_set_size} for theta estimation, remaining main training samples ({len(user_train_dataset_full_with_diff) - theta_estimation_set_size}) is {len(user_train_dataset_full_with_diff) - theta_estimation_set_size}, which might be less than min_train_length ({config.min_train_length}).")

    train_theta_split = user_train_dataset_full_with_diff.train_test_split(test_size=theta_estimation_set_size, seed=script_random_seed, shuffle=True)
    actual_user_train_dataset = train_theta_split['train']
    theta_estimation_user_dataset = train_theta_split['test']

    print(f"Split full training pool: {len(actual_user_train_dataset)} for main training, {len(theta_estimation_user_dataset)} for theta estimation.")
    print(f"Main validation set size (for acc/loss): {len(user_validation_dataset)}")

    hf_dataset_dict_for_tokenization = DatasetDict({
        'train': actual_user_train_dataset,
        'validation': user_validation_dataset,
        'test': user_test_dataset,
        'theta_estimation': theta_estimation_user_dataset
    })
    label_feature = hf_dataset_dict_for_tokenization['train'].features['label']
    num_labels_from_data = label_feature.num_classes
    print(f"Number of labels from data: {num_labels_from_data}")

    tokenizer_object = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False,
                                                     cache_dir=os.environ["TRANSFORMERS_CACHE"])

    def user_tokenize_function(examples):
        return tokenizer_object(examples["text"], padding="max_length", truncation=True, max_length=AG_NEWS_MAX_LENGTH)

    print("Tokenizing datasets...")
    num_cpus = os.cpu_count()
    num_proc_tokenize = max(1, num_cpus // 2 if num_cpus else 1)
    if num_cpus is None: num_proc_tokenize = 1
    print(f"Attempting tokenization with num_proc={num_proc_tokenize}...")

    tokenized_splits = {}
    for split_name, ds_split in hf_dataset_dict_for_tokenization.items():
        columns_to_keep_initially = ['text', 'label']
        if 'difficulty' in ds_split.column_names:
            columns_to_keep_initially.append('difficulty')
        actual_cols_to_keep = [col for col in columns_to_keep_initially if col in ds_split.column_names]
        if 'text' not in actual_cols_to_keep and 'text' in ds_split.column_names:
             actual_cols_to_keep.append('text')
        if 'text' not in ds_split.column_names:
            raise ValueError(f"'text' column missing in {split_name} before .map, cannot tokenize.")
        cols_to_remove_before_map = [col for col in ds_split.column_names if col not in actual_cols_to_keep]
        temp_ds_for_map = ds_split.remove_columns(cols_to_remove_before_map) if cols_to_remove_before_map else ds_split
        tokenized_splits[split_name] = temp_ds_for_map.map(user_tokenize_function, batched=True,
                                                           num_proc=num_proc_tokenize, remove_columns=['text'])
    hf_tokenized_dataset_dict = DatasetDict(tokenized_splits)

    for split_name_to_rename in hf_tokenized_dataset_dict.keys():
        if 'label' in hf_tokenized_dataset_dict[split_name_to_rename].column_names:
            hf_tokenized_dataset_dict[split_name_to_rename] = hf_tokenized_dataset_dict[split_name_to_rename].rename_column("label", "labels")
    print("Renamed 'label' column to 'labels' in all applicable splits.")
    print("Tokenization complete. Final structure:", hf_tokenized_dataset_dict)

    for split_name_debug, ds_debug in hf_tokenized_dataset_dict.items():
        print(f"\nDEBUGGING POST-TOKENIZATION for '{split_name_debug}': Columns: {ds_debug.column_names}")
        if 'labels' in ds_debug.column_names and len(ds_debug) > 0:
            print(f"  Sample of 'labels': {ds_debug['labels'][:3]}, Features: {ds_debug.features.get('labels', 'N/A')}")
        if 'difficulty' in ds_debug.column_names:
            print(f"  'difficulty' IS PRESENT. Sample: {ds_debug['difficulty'][:3] if len(ds_debug) > 0 else 'N/A'}")
        else:
            print(f"  'difficulty' IS NOT PRESENT.")

    model_object = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-base",
                                                                      num_labels=num_labels_from_data,
                                                                      cache_dir=os.environ["TRANSFORMERS_CACHE"])

    # Re-initialize config properly now
    config = types.SimpleNamespace()
    config.task_name_for_pudf = "ag_news_pudf"
    config.model_name = 'microsoft/deberta-v3-base'
    config.cache_dir = os.environ.get("TRANSFORMERS_CACHE")
    config.num_epochs = 20
    config.learning_rate = 1e-5
    config.patience_early_stopping = 3
    config.batch_size = 256
    config.strategy = 'theta'
    config.ordering = 'easiest'
    config.num_obs_theta = 1000
    config.min_train_length = 500
    config.lower_bound = -np.inf
    config.upper_bound = 0.0
    config.balanced = False
    config.use_length = False
    config.use_word_rarity = False
    config.gpu = 0
    config.num_workers = getattr(config, 'num_workers', 2)
    config.use_torch_compile = False
    config.task = config.task_name_for_pudf

    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir_base = f"./results_agnews_PUDF_{config.model_name.split('/')[-1]}_{config.strategy}_{run_timestamp}"
    print(f"Base output directory for PUDF run: {output_dir_base}")
    print(f"\n\n>>>>>>>> Starting AG News Training with PUDF ({config.strategy}) <<<<<<<<")
    print("--- Using PUDF Configuration ---")
    for key, value in config.__dict__.items(): print(f"  {key}: {'-Infinity' if value == -np.inf else str(value)}")
    print("-------------------------")
    task_start_time = time.time()
    results = {}
    try:
        best_val_acc, test_acc, test_loss = train_agnews_pudf_core(
            config,
            model_object,
            tokenizer_object,
            num_labels_from_data,
            hf_tokenized_dataset_dict['train'],
            hf_tokenized_dataset_dict['validation'],
            hf_tokenized_dataset_dict['test'],
            hf_tokenized_dataset_dict['theta_estimation'],
            output_dir_base
        )
        results[config.task_name_for_pudf] = {"best_val_acc": best_val_acc, "test_acc": test_acc, "test_loss": test_loss}
    except Exception as e:
        print(f"\n!FATAL ERROR Task {config.task_name_for_pudf}!")
        traceback.print_exc()
        results[config.task_name_for_pudf] = {"best_val_acc": "FATAL_ERROR", "test_acc": "FATAL_ERROR", "test_loss": "FATAL_ERROR"}
    task_end_time = time.time()
    print(f">>>>>>>> Finished Task: {config.task_name_for_pudf} in {task_end_time - task_start_time:.2f}s <<<<<<<<")

    print("\n\n===== AG News PUDF Run Summary (Manual Loss) =====")
    print(f"Strategy: {config.strategy}")
    print(f"Output Dir: {output_dir_base}")
    task_key = config.task_name_for_pudf
    if task_key in results:
        res = results[task_key]
        dev_res_str = f"{res['best_val_acc']:.4f}" if isinstance(res['best_val_acc'], float) else str(res['best_val_acc'])
        test_res_acc_str = f"{res['test_acc']:.4f}" if isinstance(res['test_acc'], float) else str(res['test_acc'])
        test_res_loss_str = f"{res.get('test_loss', 'N/A'):.4f}" if isinstance(res.get('test_loss'), float) else str(res.get('test_loss', 'N/A'))
        print(f"  - {task_key}: Best Val Acc={dev_res_str} / Test Acc={test_res_acc_str} / Test Loss={test_res_loss_str}")
    else:
        print(f"  - {task_key}: No results.")
    print("=====================")
    summary_file_path = os.path.join(output_dir_base, config.task_name_for_pudf, "overall_run_summary.json")
    try:
        os.makedirs(os.path.dirname(summary_file_path), exist_ok=True)
        with open(summary_file_path, "w") as f:
            json.dump(results, f, indent=4, default=str)
        print(f"Overall run summary saved to: {summary_file_path}")
    except Exception as e:
        print(f"Warning: Failed to save overall run summary: {e}")

if __name__ == '__main__':
    run_agnews_pudf_with_user_setup()