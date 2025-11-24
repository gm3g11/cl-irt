# -*- coding: utf-8 -*-
"""
Essential feature building function for IRT curriculum learning with Llama.
Contains only the necessary get_epoch_training_data function.
"""
import copy
import numpy as np
import torch
import gc # Keep gc import just in case, though not strictly used here

def get_epoch_training_data(ts, args, epoch, task, theta_hat=None, diffs_sorted_idx=None, lower_offset=-np.inf, upper_offset=0):
    """
    Selects a subset of training data for an epoch based on curriculum strategy.

    Args:
        ts (torch.utils.data.TensorDataset): Full training dataset including difficulties.
        args (types.SimpleNamespace): Configuration object with strategy parameters.
        epoch (int): Current epoch number.
        task (str): Name of the GLUE task (used by some strategies, not explicitly here).
        theta_hat (float, optional): Estimated model ability. Required for 'theta' strategies. Defaults to None.
        diffs_sorted_idx (np.array, optional): Pre-sorted indices based on difficulty. Calculated if None. Defaults to None.
        lower_offset (float, optional): Lower bound relative to theta_hat for 'theta' strategy. Defaults to -np.inf.
        upper_offset (float, optional): Upper bound relative to theta_hat for 'theta' strategy. Defaults to 0.

    Returns:
        dict: A dictionary containing tensors ('input_ids', 'attention_mask', 'labels', 'difficulty',
              and potentially 'token_type_ids') for the selected subset of data.
              Returns the full sorted dataset dict for 'ordered' strategy.
              For 'baseline', returns a dict representation of the input ts.
    """
    # Use getattr for safe access to potentially missing config attributes
    strategy = getattr(args, 'strategy', 'baseline')
    ordering = getattr(args, 'ordering', 'easiest')
    use_rarity = getattr(args, 'use_word_rarity', False) # Check attribute safely
    use_len = getattr(args, 'use_length', False) # Check attribute safely
    is_balanced = getattr(args, 'balanced', False) # Check attribute safely
    num_epochs = getattr(args, 'num_epochs', 20) # Default if not present
    min_train_len = getattr(args, 'min_train_length', 100)
    competency = getattr(args, 'competency', 50) # Default for NAACL strategies

    # Handle baseline strategy first
    if strategy == 'baseline':
        # Return data in the expected dictionary format
        tensors = ts.tensors
        if len(tensors) == 5:
             ids, mask, type_ids, lbls, diffs = tensors
             return {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': type_ids, 'labels': lbls, 'difficulty': diffs}
        elif len(tensors) == 4:
             ids, mask, lbls, diffs = tensors
             return {'input_ids': ids, 'attention_mask': mask, 'labels': lbls, 'difficulty': diffs}
        else:
             raise ValueError(f"Baseline: Unexpected number of tensors in input: {len(tensors)}")

    # Assertions for theta strategies
    if strategy == 'theta':
        assert theta_hat is not None, "theta_hat must be provided for theta strategy"
        # assert ordering == 'easiest', "theta strategy currently assumes ordering='easiest'" # Relaxed assertion
    if strategy == 'theta-hard':
        assert theta_hat is not None, "theta_hat must be provided for theta-hard strategy"
        # assert ordering == 'hardest', "theta-hard strategy currently assumes ordering='hardest'" # Relaxed assertion

    c_init = 0.01  # Constant from NAACL '19 paper reference

    # --- Conditionally Unpack Tensors ---
    tensors = ts.tensors
    token_type_ids = None # Initialize

    if len(tensors) == 5:
        input_ids, attention_mask, token_type_ids, labels, difficulties_tensor = tensors
        # print("build_features: Unpacked 5 tensors (incl. token_type_ids)") # Optional debug
    elif len(tensors) == 4:
        input_ids, attention_mask, labels, difficulties_tensor = tensors
        # print("build_features: Unpacked 4 tensors (no token_type_ids)") # Optional debug
    else:
        raise ValueError(f"get_epoch_training_data: Unexpected tensor count: {len(tensors)}")
    # --- End Unpacking ---

    # Use alternative difficulties if specified (safe access via getattr)
    if use_len:
        print("build_features: Using length as difficulty override.")
        difficulties_tensor = torch.tensor([len(p) for p in input_ids], dtype=torch.float32, device=input_ids.device) # Match type and device
    if use_rarity:
        # This would require passing the original text data or precomputed rarities
        print("build_features: use_word_rarity=True requires precomputed rarity tensor or text data. NOT IMPLEMENTED HERE. Using default difficulties.")
        pass # Keep original difficulties_tensor

    # Convert difficulties to numpy for sorting
    difficulties_np = difficulties_tensor.cpu().numpy()

    # Calculate sorted indices if not provided, based on ordering arg
    if diffs_sorted_idx is None:
        print(f"build_features: Sorting data based on '{ordering}' ordering.")
        if ordering == 'easiest':
            diffs_sorted_idx = np.argsort(difficulties_np)
        elif ordering == 'hardest':
            diffs_sorted_idx = np.argsort(difficulties_np)[::-1]
        elif ordering == 'middleout':
            diffs_sorted_idx = np.argsort(np.abs(difficulties_np))
        else:
            print(f"Warning: Unrecognized ordering '{ordering}'. Defaulting to 'easiest'.")
            diffs_sorted_idx = np.argsort(difficulties_np)

    # Create intermediate dictionary with sorted tensors ('train_2')
    # Handle potential absence of token_type_ids
    train_2 = {
        'input_ids': input_ids[diffs_sorted_idx],
        'attention_mask': attention_mask[diffs_sorted_idx],
        'labels': labels[diffs_sorted_idx],
        'difficulty': difficulties_tensor[diffs_sorted_idx]
    }
    if token_type_ids is not None:
        train_2['token_type_ids'] = token_type_ids[diffs_sorted_idx]

    # Apply balanced sampling if requested
    if is_balanced:
        print("build_features: Applying balanced sampling...")
        per_label_lists = {}
        sorted_labels_np = train_2['labels'].cpu().numpy()
        num_labels_overall = len(np.unique(sorted_labels_np)) # Get number of unique labels
        print(f"Balancing across {num_labels_overall} labels.")

        for i in range(len(sorted_labels_np)):
            label_item = sorted_labels_np[i].item()
            if label_item not in per_label_lists: per_label_lists[label_item] = []
            per_label_lists[label_item].append(i) # Store index within sorted array

        max_label_len = 0
        if per_label_lists: max_label_len = max(len(v) for v in per_label_lists.values())

        balanced_idx_in_sorted = []
        for l_idx in range(max_label_len):
            for label_key in sorted(per_label_lists.keys()): # Sort keys for consistency
                if l_idx < len(per_label_lists[label_key]):
                    balanced_idx_in_sorted.append(per_label_lists[label_key][l_idx])

        # Re-create train_2 dict using balanced indices
        train_2_balanced = { key: val[balanced_idx_in_sorted] for key, val in train_2.items() }
        train_2 = train_2_balanced # Overwrite with balanced version

    # --- Select final subset based on strategy ---
    train = {} # Final dictionary to return
    num_total_examples = len(input_ids)

    if strategy == 'ordered':
        print(f"build_features: Strategy 'ordered', returning {num_total_examples} sorted examples.")
        return train_2 # Return the fully sorted (and potentially balanced) set

    elif strategy == 'simple':
        data_per_epoch = num_total_examples / (num_epochs / 2.0) if num_epochs > 0 else num_total_examples
        # Original logic: ramp up even/odd epochs differently? Seems complex.
        # Simpler linear ramp: num_train = min(int(data_per_epoch * (epoch + 1)), num_total_examples)
        # Using original logic:
        if epoch % 2 == 0: num_train = min(int(data_per_epoch * (epoch + 1)), num_total_examples)
        else: num_train = min(int(data_per_epoch * epoch), num_total_examples)
        num_train = max(num_train, min_train_len); num_train = min(num_train, num_total_examples)
        print(f"build_features: Strategy 'simple', selecting {num_train} examples for epoch {epoch+1}.")
        for key, val in train_2.items(): train[key] = val[:num_train]
        return train

    elif strategy == 'theta':
        difficulties_sorted = train_2['difficulty'] # Use difficulties from sorted intermediate dict
        lower_bound_val = theta_hat + lower_offset
        upper_bound_val = theta_hat + upper_offset
        # Find indices where difficulty is within bounds using torch tensor operations
        train_idx_mask = (difficulties_sorted >= lower_bound_val) & (difficulties_sorted <= upper_bound_val)
        train_idx = torch.where(train_idx_mask)[0].cpu() # Get indices on CPU

        final_indices = train_idx
        if len(train_idx) < min_train_len:
             print(f"Theta strategy selected only {len(train_idx)}, falling back to min {min_train_len} easiest.")
             # Take first min_train_len indices from train_2 (which is sorted by 'ordering')
             final_indices = torch.arange(min(min_train_len, num_total_examples))

        print(f"build_features: Strategy 'theta' ({lower_offset=}, {upper_offset=}, {theta_hat=:.3f}), selected {len(final_indices)} examples.")
        for key, val in train_2.items(): train[key] = val[final_indices]
        return train

    elif strategy == 'theta-hard':
        difficulties_sorted = train_2['difficulty']
        train_idx_mask = difficulties_sorted >= theta_hat
        train_idx = torch.where(train_idx_mask)[0].cpu()

        final_indices = train_idx
        if len(train_idx) < min_train_len:
             print(f"Theta-hard strategy selected only {len(train_idx)}, falling back to min {min_train_len} hardest.")
             final_indices = torch.arange(min(min_train_len, num_total_examples))

        print(f"build_features: Strategy 'theta-hard' ({theta_hat=:.3f}), selected {len(final_indices)} examples.")
        for key, val in train_2.items(): train[key] = val[final_indices]
        return train

    elif strategy in ['naacl-linear', 'naacl-root']:
        if strategy == 'naacl-linear': epoch_competency = np.min([1.0, epoch * ((1.0 - c_init) / competency) + c_init])
        else: epoch_competency = np.min([1.0, np.sqrt(epoch * ((1.0 - c_init ** 2) / competency) + c_init ** 2)])
        num_train = int(epoch_competency * num_total_examples)
        num_train = max(num_train, min_train_len); num_train = min(num_train, num_total_examples)
        print(f"build_features: Strategy '{strategy}', competency {epoch_competency:.3f}, selecting {num_train} examples.")
        for key, val in train_2.items(): train[key] = val[:num_train]
        return train

    else:
        raise NotImplementedError(f"Strategy '{strategy}' not implemented.")
