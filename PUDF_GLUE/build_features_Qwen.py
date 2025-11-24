# -*- coding: utf-8 -*-
"""
Essential feature building function for IRT curriculum learning with Qwen.
Optimized for Qwen (no token_type_ids) and improved theta strategy.
"""
import numpy as np
import torch
import traceback # Add for debugging

def get_epoch_training_data(ts, args, epoch, task, theta_hat=None, diffs_sorted_idx=None, lower_offset=-np.inf, upper_offset=0):
    """
    Selects a subset of training data for an epoch based on curriculum strategy.
    ... (rest of docstring) ...
    """
    print(f"\n--- [get_epoch_training_data Debug] Epoch: {epoch+1}, Task: {task}, Strategy: {args.strategy} ---")

    # Configuration attributes with defaults
    strategy = getattr(args, 'strategy', 'baseline')
    ordering = getattr(args, 'ordering', 'easiest')
    use_rarity = getattr(args, 'use_word_rarity', False)
    use_len = getattr(args, 'use_length', False)
    is_balanced = getattr(args, 'balanced', False) # Defined here
    try:
        print(f"[get_epoch Debug] Defined is_balanced: type={type(is_balanced)}, value={is_balanced}")
    except NameError:
         print("[get_epoch Debug] ALARM! 'is_balanced' is NOT DEFINED immediately after getattr!")
         return {} # Should not happen with correct file
    num_epochs = getattr(args, 'num_epochs', 20)
    min_train_len = getattr(args, 'min_train_length', 100)
    competency = getattr(args, 'competency', 50)

    # Handle baseline strategy FIRST
    if strategy == 'baseline':
        print(f"[get_epoch Debug] Strategy 'baseline', returning all {len(ts)} examples.")
        tensors = ts.tensors
        if len(tensors) != 4:
            raise ValueError(f"Baseline: Expected 4 tensors (input_ids, attention_mask, labels, difficulty), got {len(tensors)}")
        ids, mask, lbls, diffs = tensors
        final_dict = {'input_ids': ids, 'attention_mask': mask, 'labels': lbls, 'difficulty': diffs}
        print(f"[get_epoch Debug] Baseline return keys: {final_dict.keys()}")
        return final_dict

    # --- Setup for non-baseline strategies ---

    # Validate theta strategies
    if strategy in ['theta', 'theta-hard']:
        if theta_hat is None:
            raise ValueError(f"Strategy '{strategy}' requires theta_hat")
        print(f"[get_epoch Debug] Theta strategy, theta_hat={theta_hat:.4f}")

    # Unpack tensors
    tensors = ts.tensors
    print(f"[get_epoch Debug] Input TensorDataset has {len(tensors)} tensors.")
    if len(tensors) != 4:
        raise ValueError(f"Expected 4 tensors (input_ids, attention_mask, labels, difficulty), got {len(tensors)}")
    input_ids, attention_mask, labels, difficulties_tensor = tensors
    print(f"[get_epoch Debug] Initial labels shape: {labels.shape}")
    num_total_examples = len(input_ids)

    # ... (difficulty override logic) ...

    # Convert difficulties to numpy
    difficulties_np = difficulties_tensor.cpu().numpy()
    print(f"[get_epoch Debug] difficulties_np - Shape: {difficulties_np.shape}, Dtype: {difficulties_np.dtype}, HasNaN: {np.isnan(difficulties_np).any()}, HasInf: {np.isinf(difficulties_np).any()}")
    if difficulties_np.size == 0:
        print("[get_epoch Debug] WARNING: difficulties_np is empty!")
        return {}


    # Compute sorted indices if not provided
    if diffs_sorted_idx is None:
        print(f"[get_epoch Debug] diffs_sorted_idx is None. Computing based on ordering '{ordering}'.")
        computed_indices = None # Initialize temporary variable
        try:
            # >>>>> DEBUG PRINT: Confirm try block entry <<<<<
            print(f"[get_epoch Debug] Entering try block for argsort (ordering='{ordering}')...")
            # >>>>> END DEBUG PRINT <<<<<

            if ordering == 'easiest':
                computed_indices = np.argsort(difficulties_np)
            elif ordering == 'hardest':
                computed_indices = np.argsort(difficulties_np)[::-1]
            elif ordering == 'middleout':
                safe_difficulties = np.nan_to_num(difficulties_np)
                computed_indices = np.argsort(np.abs(safe_difficulties))
            else:
                print(f"Epoch {epoch+1}: Warning: Unknown ordering '{ordering}', using 'easiest'.")
                computed_indices = np.argsort(difficulties_np)

            print(f"[get_epoch Debug] Value computed inside 'if is None': type={type(computed_indices)}, len={len(computed_indices) if computed_indices is not None else 'None'}")
            if computed_indices is None: print("[get_epoch Debug] ALARM! computed_indices is None after argsort block!")
            elif computed_indices.size == 0: print("[get_epoch Debug] WARNING! computed_indices is empty after argsort block!")

            # Assign to the main variable *ONLY IF COMPUTATION SUCCEEDED*
            diffs_sorted_idx = computed_indices

        except Exception as e:
             print(f"[get_epoch Debug] ERROR during np.argsort or related operation!")
             # >>>>> LOOK FOR THIS TRACEBACK IN YOUR FULL LOGS <<<<<
             traceback.print_exc()
             # >>>>> Explicitly set to None on error <<<<<
             diffs_sorted_idx = None # Ensure it's None if computation failed


    # Check AGAIN before indexing / train_2 creation
    print(f"[get_epoch Debug] Value of diffs_sorted_idx JUST BEFORE indexing: type={type(diffs_sorted_idx)}")
    if diffs_sorted_idx is None:
        print("[get_epoch Debug] ALARM! diffs_sorted_idx is None before indexing! Error likely occurred during computation.")
        print("[get_epoch Debug] Returning empty dictionary due to missing sorted indices.")
        return {} # Prevent crash by returning empty dict
    elif len(diffs_sorted_idx) == 0:
        print("[get_epoch Debug] WARNING! diffs_sorted_idx is an empty array/list before indexing!")
        print("[get_epoch Debug] Returning empty dictionary because sorted indices are empty.")
        return {} # Return empty dict if indices are empty


    # Intermediate sorted dataset
    try:
        # >>>>> DEBUG PRINT 3: Check variables right before train_2 creation <<<<<
        try:
            print(f"[get_epoch Debug] BEFORE train_2 creation:")
            variables_exist = True
            # Check each variable needed for train_2
            for var_name in ['input_ids', 'attention_mask', 'labels', 'difficulties_tensor', 'diffs_sorted_idx']:
                 exists = var_name in locals() or var_name in globals()
                 print(f"  - {var_name} exists: {exists}")
                 if exists:
                     var_value = locals().get(var_name, globals().get(var_name))
                     print(f"  - type({var_name}): {type(var_value)}")
                     if hasattr(var_value, 'shape'): print(f"  - {var_name}.shape: {var_value.shape}")
                     elif hasattr(var_value, '__len__'): print(f"  - len({var_name}): {len(var_value)}")
                 else:
                     variables_exist = False
            if not variables_exist:
                 print("[get_epoch Debug] ALARM! One or more variables needed for train_2 do not exist!")
        except Exception as e_dbg:
            print(f"[get_epoch Debug] ERROR checking variables before train_2 creation: {e_dbg}")
        # >>>>> END DEBUG PRINT 3 <<<<<

        # Use the validated diffs_sorted_idx
        print("[get_epoch Debug] Attempting to create train_2...")
        train_2 = {
            'input_ids': input_ids[diffs_sorted_idx],
            'attention_mask': attention_mask[diffs_sorted_idx],
            'labels': labels[diffs_sorted_idx],
            'difficulty': difficulties_tensor[diffs_sorted_idx]
        }
        print(f"[get_epoch Debug] Created train_2 dict. Keys: {train_2.keys()}. Labels shape: {train_2.get('labels', torch.tensor([])).shape}")
        if len(train_2.get('input_ids', [])) == 0:
             print("[get_epoch Debug] WARNING: train_2 is empty after initial sorting/indexing.")
             # This might be okay if subsequent filtering is expected, but could lead to issues.

    except NameError as ne: # Catch specific error if variables still missing
         print(f"[get_epoch Debug] CAUGHT NameError creating train_2: {ne}")
         traceback.print_exc()
         print("[get_epoch Debug] Local variables at time of NameError:", locals().keys())
         return {}
    except IndexError as e:
        print(f"[get_epoch Debug] ERROR creating train_2! Sorted indices might be out of bounds or empty.")
        print(f"Size of diffs_sorted_idx: {len(diffs_sorted_idx)}")
        if len(diffs_sorted_idx) > 0: print(f"Max sorted index: {np.max(diffs_sorted_idx)}, Original dataset size: {len(input_ids)}")
        traceback.print_exc()
        return {}
    except Exception as e:
        print(f"[get_epoch Debug] UNEXPECTED ERROR creating train_2!")
        traceback.print_exc()
        return {}


    # Apply balanced sampling if enabled
    try:
        print(f"[get_epoch Debug] Checking is_balanced BEFORE 'if' statement: type={type(is_balanced)}, value={is_balanced}")
    except NameError: print("[get_epoch Debug] ALARM! 'is_balanced' does NOT exist right before the 'if' statement!") ; return {}
    except Exception as e: print(f"[get_epoch Debug] Error printing is_balanced before 'if': {e}") ; return {}

    if is_balanced:
        print(f"Epoch {epoch+1}: Applying balanced sampling.")
        # ... (balancing logic as before) ...
    else:
        print(f"[get_epoch Debug] is_balanced is False. Skipping balancing.")


    # --- Select subset based on strategy ---
    # ... (if/elif/else strategy logic as corrected in previous step) ...
    # ... (make sure all strategies handle potential empty train_2/indices) ...
    # Example for theta return
    if strategy == 'ordered':
        print(f"Epoch {epoch+1}: Strategy 'ordered', returning {len(train_2.get('input_ids',[]))} sorted examples.")
        print(f"[get_epoch Debug] Ordered return keys: {train_2.keys()}")
        return train_2

    elif strategy == 'simple':
        # ... (simple logic as before, ensuring num_train handles empty train_2) ...
        num_total_in_train2 = len(train_2.get('input_ids',[]))
        if num_total_in_train2 == 0: return {} # Return empty if train_2 empty
        data_per_epoch = num_total_examples / (num_epochs / 2.0) if num_epochs > 0 else num_total_examples
        num_train = min(int(data_per_epoch * (epoch + 1)), num_total_examples) if epoch % 2 == 0 else min(int(data_per_epoch * epoch), num_total_examples)
        num_train = min(num_train, num_total_in_train2) # Can't select more than available
        effective_min_train_len = min(min_train_len, num_total_in_train2)
        num_train = max(num_train, effective_min_train_len) # Apply min length
        print(f"Epoch {epoch+1}: Strategy 'simple', selecting {num_train} examples.")
        final_dict = {key: val[:num_train] for key, val in train_2.items()}
        print(f"[get_epoch Debug] Simple return keys: {final_dict.keys()}")
        return final_dict


    elif strategy == 'theta':
        num_total_in_train2 = len(train_2.get('input_ids',[]))
        if num_total_in_train2 == 0:
             print("[get_epoch Debug] Theta strategy: train_2 is empty. Returning empty dict.")
             return {}
        try:
            difficulties_sorted = train_2['difficulty']
            lower_bound_val = theta_hat + lower_offset
            upper_bound_val = theta_hat + upper_offset
            print(f"[get_epoch Debug] Theta bounds: [{lower_bound_val:.4f}, {upper_bound_val:.4f}]")
            train_idx_mask = (difficulties_sorted.cpu() >= lower_bound_val) & (difficulties_sorted.cpu() <= upper_bound_val)
            train_idx = torch.where(train_idx_mask)[0].cpu()
            print(f"[get_epoch Debug] Theta initial selection count: {len(train_idx)}")

            effective_min_train_len = min(min_train_len, num_total_in_train2)
            if len(train_idx) < effective_min_train_len:
                print(f"[get_epoch Debug] Theta count {len(train_idx)} < min_train_len {effective_min_train_len}. Selecting closest...")
                current_difficulties_np = train_2['difficulty'].cpu().numpy()
                distances = np.abs(current_difficulties_np - theta_hat)
                closest_original_indices_in_train_2 = np.argsort(distances)[:effective_min_train_len]
                final_indices = torch.tensor(closest_original_indices_in_train_2, dtype=torch.long)
                print(f"Epoch {epoch+1}: Adjusted theta selection to {len(final_indices)} examples closest to theta_hat={theta_hat:.3f}.")
            else:
                final_indices = train_idx
                print(f"Epoch {epoch+1}: Strategy 'theta' (theta_hat={theta_hat:.3f}, lower_offset={lower_offset}, upper_offset={upper_offset}), selected {len(final_indices)} examples.")

            print(f"[get_epoch Debug] Theta strategy final_indices count: {len(final_indices)}")
            if len(final_indices) == 0: return {}

            try:
                final_dict = {key: val[final_indices] for key, val in train_2.items()}
                print(f"[get_epoch Debug] Theta strategy FINAL dict keys: {final_dict.keys()}")
                if 'labels' not in final_dict and len(final_indices)>0: print(f"!!!!!!!! CRITICAL: 'labels' key IS MISSING! !!!!!!!") # Use PLURAL
                elif 'labels' in final_dict: print(f"[get_epoch Debug] Final dict 'labels' shape: {final_dict['labels'].shape}") # Use PLURAL
                return final_dict
            except IndexError as e: print(f"[get_epoch Debug] ERROR during final indexing in theta strategy!") ; traceback.print_exc(); return {}

        except Exception as e: print(f"[get_epoch Debug] UNEXPECTED ERROR in 'theta' strategy block!") ; traceback.print_exc(); return {}


    elif strategy == 'theta-hard':
        # ... (theta-hard logic as corrected before) ...
         if len(train_2.get('input_ids',[])) == 0: return {}
         # ... (calculate final_indices) ...
         if len(final_indices) == 0: return {}
         try:
             final_dict = {key: val[final_indices] for key, val in train_2.items()}
             print(f"[get_epoch Debug] Theta-hard return keys: {final_dict.keys()}")
             return final_dict
         except Exception as e: print(f"[get_epoch Debug] ERROR during final processing in theta-hard: {e}"); return {}

    elif strategy in ['naacl-linear', 'naacl-root']:
         # ... (naacl logic as corrected before) ...
         if len(train_2.get('input_ids',[])) == 0: return {}
         # ... (calculate num_train) ...
         try:
             final_dict = {key: val[:num_train] for key, val in train_2.items()}
             print(f"[get_epoch Debug] NAACL return keys: {final_dict.keys()}")
             return final_dict
         except Exception as e: print(f"[get_epoch Debug] ERROR during final processing in naacl: {e}"); return {}


    # Fallback
    else:
         print(f"[get_epoch Debug] Strategy '{strategy}' hit NotImplementedError.")
         raise NotImplementedError(f"Strategy '{strategy}' not implemented or code flow issue.")