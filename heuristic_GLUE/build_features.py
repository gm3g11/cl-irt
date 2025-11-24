# return features for the datasets we're working with
import copy
import gc
import numpy as np
import pandas as pd
import re
import time
import json
import torch
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
import warnings # To suppress specific warnings if needed

# --- Helper Functions ---

def tokenize(sent):
    """Basic tokenization splitting on non-word characters."""
    if not isinstance(sent, str):
        sent = str(sent) # Ensure input is a string
    # Keep it simple, split on whitespace and remove punctuation if needed,
    # but the original regex split is kept here.
    tokens = [x.strip() for x in re.split(r'\W+', sent) if x.strip()]
    return ' '.join(tokens) if tokens else "" # Return empty string if no tokens


# --- SNLI / SST-B Specific Loaders (Classic Style with GloVe) ---
# Note: These require specific file structures and external GloVe embeddings.

def _snli_sstb_preprocess(X, train=False, bert_like=False):
    """Helper to preprocess SNLI/SST-B pandas DataFrames."""
    labels_list = []
    data_list = []
    pids_list = []
    diffs_list = []
    error_count = 0

    required_cols = ['gold_label', 'sentence1']
    if not bert_like: # Non-BERT needs sentence2
        required_cols.append('sentence2')
    if train:
        required_cols.append('difficulty')

    # Check if required columns exist
    if not all(col in X.columns for col in required_cols):
        raise ValueError(f"Input DataFrame missing required columns. Needed: {required_cols}, Found: {X.columns.tolist()}")

    for index, row in X.iterrows():
        try:
            lbl = row.gold_label
            # Skip rows with '-' label (common in SNLI)
            if lbl == '-':
                continue

            s1 = str(row.sentence1) # Ensure string
            s2 = str(row.get('sentence2', '')) if not bert_like else '' # Handle missing sentence2 gracefully

            if not bert_like:
                # Tokenize here if not using a Transformer tokenizer later
                s1_tokens = tokenize(s1).split(' ')
                s2_tokens = tokenize(s2).split(' ')
                processed_sent = [s1_tokens, s2_tokens]
            else:
                # Keep raw strings for external tokenizer
                processed_sent = [s1, str(row.get('sentence2', '')) if 'sentence2' in row else ''] # Keep pair structure if sentence2 exists


            pid = row.get('pairID', index) # Use index if pairID is missing
            pair_diff = row.difficulty if train else 0.0

            data_list.append(processed_sent)
            labels_list.append(lbl)
            diffs_list.append(pair_diff)
            pids_list.append(pid)

        except Exception as e:
            print(f'Warning: Exception processing row {index}: {e}')
            error_count += 1

    if error_count > 0:
        print(f'Processed with {error_count} errors.')

    # Use 'labels' consistently as the key
    result = {'phrase': data_list, 'labels': labels_list, 'pairID': pids_list, 'difficulty': diffs_list}
    return result

def _load_glove_and_build_vocab(dataframes, data_dir):
    """Builds vocab from dataframes and loads corresponding GloVe vectors."""
    print("Building vocabulary from datasets...")
    vocab = set()
    for df in dataframes:
        for index, row in df.iterrows():
            try:
                s1 = str(row['sentence1'])
                vocab.update(tokenize(s1).split(' '))
                if 'sentence2' in row:
                     s2 = str(row['sentence2'])
                     vocab.update(tokenize(s2).split(' '))
            except Exception:
                continue # Skip rows causing errors

    print(f"Vocabulary size: {len(vocab)}")
    if not vocab:
        raise ValueError("Vocabulary is empty after processing data.")

    print("Loading GloVe vectors...")
    glove_file = os.path.join(data_dir, 'raw', 'glove.840B.300d.txt')
    if not os.path.exists(glove_file):
        raise FileNotFoundError(f"GloVe file not found: {glove_file}")

    vectors = []
    w2i = {}
    i2w = {}
    found_vectors = 0
    with open(glove_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            vals = line.rstrip().split(' ')
            word = vals[0]
            if word in vocab:
                try:
                    vec = np.array(list(map(float, vals[1:])), dtype='float32')
                    # Check vector dimension (optional but good)
                    if len(vec) != 300:
                         print(f"Warning: Skipping word '{word}' with incorrect dimension {len(vec)} at line {i+1}")
                         continue
                    w2i[word] = len(vectors) # Assign index based on current vector count
                    i2w[len(vectors)] = word
                    vectors.append(vec)
                    found_vectors += 1
                except ValueError:
                    print(f"Warning: Skipping word '{word}' due to non-float value at line {i+1}")

    print(f"Found {found_vectors} vectors for {len(vocab)} vocab words.")

    # Handle OOV words - Add them to mapping without a vector or with zeros
    oov_count = 0
    next_idx = len(vectors)
    for word in vocab:
        if word not in w2i:
            w2i[word] = next_idx
            i2w[next_idx] = word
            # Option 1: Add zero vectors for OOV
            # vectors.append(np.zeros(300, dtype='float32'))
            # Option 2: Don't add vector (handle embedding lookup later)
            next_idx += 1
            oov_count += 1
    print(f"Added {oov_count} OOV words to mapping.")

    # Add PAD token
    pad_idx = next_idx
    w2i['<PAD>'] = pad_idx
    i2w[pad_idx] = '<PAD>'
    # vectors.append(np.zeros(300, dtype='float32')) # Add PAD vector if needed

    print(f"Final mapping size: {len(w2i)}")
    # Convert vectors list to a single numpy array
    vectors_np = np.array(vectors)
    return w2i, i2w, vectors_np


def load_snli(data_dir):
    """Loads SNLI data, preprocesses, loads GloVe."""
    print('Loading SNLI data...')
    train_file = os.path.join(data_dir, 'processed', 'snli_1.0_train_diff.txt')
    dev_file = os.path.join(data_dir, 'raw', 'snli_1.0_dev.txt')
    test_file = os.path.join(data_dir, 'raw', 'snli_1.0_test.txt')

    try:
        train_df = pd.read_csv(train_file, sep='\t', usecols=['gold_label', 'sentence1', 'sentence2', 'pairID', 'difficulty'])
        dev_df = pd.read_csv(dev_file, sep='\t', usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
        test_df = pd.read_csv(test_file, sep='\t', usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
    except FileNotFoundError as e:
        print(f"Error loading SNLI files: {e}")
        raise
    except ValueError as e: # Handle potential usecols error
         print(f"Error reading columns: {e}")
         raise


    print('Preprocessing SNLI data...')
    # bert_like=False means tokenize internally
    out_train = _snli_sstb_preprocess(train_df, train=True, bert_like=False)
    out_dev = _snli_sstb_preprocess(dev_df, train=False, bert_like=False)
    out_test = _snli_sstb_preprocess(test_df, train=False, bert_like=False)
    gc.collect()

    print("Encoding labels...")
    le = LabelEncoder()
    # Use the consistent 'labels' key from _snli_sstb_preprocess
    le.fit(out_train['labels'])
    out_train['labels'] = le.transform(out_train['labels'])
    out_dev['labels'] = le.transform(out_dev['labels'])
    out_test['labels'] = le.transform(out_test['labels'])

    print(f"Training set size: {len(out_train['labels'])}")

    w2i, i2w, vectors = _load_glove_and_build_vocab([train_df, dev_df, test_df], data_dir)

    return out_train, out_dev, out_test, w2i, i2w, vectors

def load_sstb(data_dir):
    """Loads SST-B data, preprocesses, loads GloVe."""
    print('Loading SST-B data...')
    # Adjust filenames as needed
    train_file = os.path.join(data_dir, 'processed', 'sstb_train_diff.tsv')
    dev_file = os.path.join(data_dir, 'raw', 'sstb_dev.tsv')
    test_file = os.path.join(data_dir, 'raw', 'sstb_test.labeled.tsv') # Assuming test file has labels

    try:
        # Assuming SST-B has columns like 'sentence', 'label', 'id', 'difficulty'
        train_df = pd.read_csv(train_file, sep='\t', names=['sentence', 'label', 'id', 'difficulty'], header=0)
        # Dev/Test might have different structure or no header
        dev_df = pd.read_csv(dev_file, sep='\t', names=['sentence', 'label'], header=0) # Example structure
        test_df = pd.read_csv(test_file, sep='\t', names=['label', 'sentence'], header=None) # Example structure for test
        test_df = test_df[['sentence', 'label']] # Reorder if needed

        # Add placeholder sentence2 for compatibility with preprocess function if needed
        # Or adapt preprocess to handle single sentences directly
        train_df['sentence1'] = train_df['sentence']
        train_df['gold_label'] = train_df['label']
        train_df['pairID'] = train_df['id']

        dev_df['sentence1'] = dev_df['sentence']
        dev_df['gold_label'] = dev_df['label']
        dev_df['pairID'] = dev_df.index # Use index as ID

        test_df['sentence1'] = test_df['sentence']
        test_df['gold_label'] = test_df['label']
        test_df['pairID'] = test_df.index # Use index as ID

    except FileNotFoundError as e:
        print(f"Error loading SST-B files: {e}")
        raise
    except Exception as e:
        print(f"Error reading SST-B files (check format/columns): {e}")
        raise


    print('Preprocessing SST-B data...')
    # bert_like=False means tokenize internally
    out_train = _snli_sstb_preprocess(train_df, train=True, bert_like=False)
    out_dev = _snli_sstb_preprocess(dev_df, train=False, bert_like=False)
    out_test = _snli_sstb_preprocess(test_df, train=False, bert_like=False)
    gc.collect()

    print("Encoding labels...")
    le = LabelEncoder()
    # Use the consistent 'labels' key
    le.fit(out_train['labels'])
    out_train['labels'] = le.transform(out_train['labels'])
    out_dev['labels'] = le.transform(out_dev['labels'])
    out_test['labels'] = le.transform(out_test['labels'])

    print(f"Training set size: {len(out_train['labels'])}")

    w2i, i2w, vectors = _load_glove_and_build_vocab([train_df, dev_df, test_df], data_dir)

    return out_train, out_dev, out_test, w2i, i2w, vectors


# --- BERT-Compatible Loaders ---

def load_snli_bert(data_dir):
    """Loads SNLI data, returns raw strings for external tokenization."""
    print('Loading SNLI data for BERT...')
    train_file = os.path.join(data_dir, 'processed', 'snli_1.0_train_diff.txt')
    dev_file = os.path.join(data_dir, 'raw', 'snli_1.0_dev.txt')
    test_file = os.path.join(data_dir, 'raw', 'snli_1.0_test.txt')
    try:
        train_df = pd.read_csv(train_file, sep='\t', usecols=['gold_label', 'sentence1', 'sentence2', 'pairID', 'difficulty'])
        dev_df = pd.read_csv(dev_file, sep='\t', usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
        test_df = pd.read_csv(test_file, sep='\t', usecols=['gold_label', 'sentence1', 'sentence2', 'pairID'])
    except FileNotFoundError as e: print(f"Error loading SNLI files: {e}"); raise
    except ValueError as e: print(f"Error reading columns: {e}"); raise

    print('Preprocessing SNLI data (BERT)...')
    out_train = _snli_sstb_preprocess(train_df, train=True, bert_like=True)
    out_dev = _snli_sstb_preprocess(dev_df, train=False, bert_like=True)
    out_test = _snli_sstb_preprocess(test_df, train=False, bert_like=True)
    gc.collect()
    # No label encoding here, assume handled later or labels are already suitable
    return out_train, out_dev, out_test


def load_sstb_bert(data_dir):
    """Loads SST-B data, returns raw strings for external tokenization."""
    print('Loading SST-B data for BERT...')
    train_file = os.path.join(data_dir, 'processed', 'sstb_train_diff.tsv')
    dev_file = os.path.join(data_dir, 'raw', 'sstb_dev.tsv')
    test_file = os.path.join(data_dir, 'raw', 'sstb_test.labeled.tsv')
    try:
        train_df = pd.read_csv(train_file, sep='\t', names=['sentence', 'label', 'id', 'difficulty'], header=0)
        dev_df = pd.read_csv(dev_file, sep='\t', names=['sentence', 'label'], header=0)
        test_df = pd.read_csv(test_file, sep='\t', names=['label', 'sentence'], header=None)
        test_df = test_df[['sentence', 'label']]

        train_df['sentence1'] = train_df['sentence']; train_df['gold_label'] = train_df['label']; train_df['pairID'] = train_df['id']
        dev_df['sentence1'] = dev_df['sentence']; dev_df['gold_label'] = dev_df['label']; dev_df['pairID'] = dev_df.index
        test_df['sentence1'] = test_df['sentence']; test_df['gold_label'] = test_df['label']; test_df['pairID'] = test_df.index
    except FileNotFoundError as e: print(f"Error loading SST-B files: {e}"); raise
    except Exception as e: print(f"Error reading SST-B files: {e}"); raise

    print('Preprocessing SST-B data (BERT)...')
    # Pass bert_like=True, handles single sentence via sentence1
    out_train = _snli_sstb_preprocess(train_df, train=True, bert_like=True)
    out_dev = _snli_sstb_preprocess(dev_df, train=False, bert_like=True)
    out_test = _snli_sstb_preprocess(test_df, train=False, bert_like=True)
    gc.collect()
    return out_train, out_dev, out_test


# --- GLUE Specific Loader (More Modern Style) ---

def parse_line(line_parts, task):
    """Parses a tokenized line from a GLUE TSV file."""
    try:
        if task == 'CoLA':
            # Index, Label, ?, Sentence
            return line_parts[0], line_parts[3], None, line_parts[1]
        elif task == 'SST-2':
            # Sentence, Label
            return -1, line_parts[0], None, line_parts[1] # Use -1 if ID is missing
        elif task == 'MRPC':
            # Label, ?, ?, Sent1, Sent2
            return -1, line_parts[3], line_parts[4], line_parts[0]
        elif task in ['QNLI', 'RTE', 'WNLI']:
            # Index, Sent1, Sent2, Label
            return line_parts[0], line_parts[1], line_parts[2], line_parts[3]
        elif task == 'QQP':
             # Index, ?, ?, Q1, Q2, Label
            return line_parts[0], line_parts[3], line_parts[4], line_parts[5]
        elif task == 'MNLI':
             # Index, ..., Sent1, Sent2, ..., Label
            return line_parts[0], line_parts[8], line_parts[9], line_parts[-1] # Assume label is last for MNLI dev/test
        else:
            raise NotImplementedError(f"Parsing not implemented for task: {task}")
    except IndexError as e:
        print(f"Warning: IndexError parsing line for task {task}. Line parts: {line_parts}. Error: {e}")
        # Return placeholders on error
        return -2, "PARSE_ERROR", None, -1

def get_example_rarities(list_of_strings):
    """Calculates word rarity for a list of strings. Higher score = rarer."""
    if not isinstance(list_of_strings, list) or not all(isinstance(s, str) for s in list_of_strings):
         raise ValueError("Input to get_example_rarities must be a list of strings.")

    print(f"Calculating word rarity for {len(list_of_strings)} examples...")
    result = []
    tokenized_corpus = [tokenize(text).split(' ') for text in list_of_strings]

    counts = dict()
    N = 0
    for tokens in tokenized_corpus:
        valid_tokens = [t for t in tokens if t]
        N += len(valid_tokens)
        for tok in valid_tokens:
            counts.setdefault(tok, 0)
            counts[tok] += 1

    if N == 0:
        print("Warning: Corpus resulted in 0 tokens for rarity calculation.")
        return [0.0] * len(list_of_strings)

    # Calculate rarity using mean negative log probability
    epsilon = 1e-9 # Avoid log(0)
    for tokens in tokenized_corpus:
        valid_tokens = [t for t in tokens if t]
        if not valid_tokens:
            p_hat = 0.0 # Assign 0 rarity to empty sentences/sequences
        else:
            log_probs = [np.log(counts.get(tok, 0) / N + epsilon) for tok in valid_tokens]
            p_hat = -np.mean(log_probs) # Higher value means rarer on average

        result.append(p_hat)
    print("Word rarity calculation finished.")
    return result

def load_glue_task(datadir, diffdir, taskname, calculate_rarity=True, seed=42):
    """
    Loads GLUE data, merges with IRT difficulties, optionally calculates word rarity.
    Performs a 90/10 stratified train/dev split using the provided seed.
    Returns dictionaries: train_result, dev_result, test_result (using GLUE dev as test).
    """
    GLUETASKS = ['CoLA', 'SST-2', 'MRPC', 'MNLI', 'QNLI', 'RTE', 'WNLI', 'QQP']
    if taskname not in GLUETASKS: raise ValueError(f"Task {taskname} not supported.")

    print(f"Loading GLUE task: {taskname}")
    # Define file paths
    train_file_path = os.path.join(datadir, taskname, 'train.tsv')
    # Determine dev/test file paths based on task
    if taskname == 'MNLI':
        dev_file_path = os.path.join(datadir, taskname, 'dev_matched.tsv')
        test_file_path = os.path.join(datadir, taskname, 'dev_matched.tsv') # Often use dev_matched as test
    else:
        dev_file_path = os.path.join(datadir, taskname, 'dev.tsv')
        test_file_path = os.path.join(datadir, taskname, 'dev.tsv') # Often use dev as test

    # Check if files exist
    if not os.path.exists(train_file_path): raise FileNotFoundError(f"Train file not found: {train_file_path}")
    if not os.path.exists(dev_file_path): raise FileNotFoundError(f"Dev file not found: {dev_file_path}")
    # Test file might be different (e.g., test_matched.tsv), adapt if needed

    # Load Raw Data into lists/dictionaries
    train_data = {'id': [], 's1': [], 's2': [], 'label': []}
    dev_data = {'id': [], 's1': [], 's2': [], 'label': []}
    test_data = {'id': [], 's1': [], 's2': [], 'label': []} # Load labels for dev set used as test

    print("Reading and parsing files...")
    try:
        with open(train_file_path, 'r', encoding='utf-8') as f:
            if taskname != 'CoLA': next(f) # Skip header for most tasks
            for i, line in enumerate(f):
                parts = line.strip().split('\t')
                lid, s1, s2, label = parse_line(parts, taskname)
                if lid == -1: lid = f"{taskname}_train_{i}" # Generate ID if missing
                if lid == -2: continue # Skip lines with parsing errors
                train_data['id'].append(lid)
                train_data['s1'].append(s1)
                train_data['s2'].append(s2) # s2 might be None
                train_data['label'].append(label)

        with open(dev_file_path, 'r', encoding='utf-8') as f:
            if taskname != 'CoLA': next(f)
            for i, line in enumerate(f):
                parts = line.strip().split('\t')
                lid, s1, s2, label = parse_line(parts, taskname)
                if lid == -1: lid = f"{taskname}_dev_{i}"
                if lid == -2: continue
                dev_data['id'].append(lid)
                dev_data['s1'].append(s1)
                dev_data['s2'].append(s2)
                dev_data['label'].append(label)

        # Load test data (using dev set path as specified)
        with open(test_file_path, 'r', encoding='utf-8') as f:
            if taskname != 'CoLA': next(f)
            for i, line in enumerate(f):
                parts = line.strip().split('\t')
                 # Assume same parsing logic applies to dev set used as test
                lid, s1, s2, label = parse_line(parts, taskname)
                if lid == -1: lid = f"{taskname}_test_{i}"
                if lid == -2: continue
                test_data['id'].append(lid)
                test_data['s1'].append(s1)
                test_data['s2'].append(s2)
                test_data['label'].append(label) # Keep label for test set evaluation

    except Exception as e:
        print(f"Error reading or parsing GLUE files for {taskname}: {e}")
        raise

    print(f"Raw counts: Train={len(train_data['id'])}, Dev={len(dev_data['id'])}, Test={len(test_data['id'])}")

    # Load IRT difficulties
    train_diff_file = os.path.join(diffdir, f'{taskname.lower()}-1pl', 'best_parameters.json') # Corrected path separator
    print(f"Loading IRT difficulties from: {train_diff_file}")
    if not os.path.exists(train_diff_file):
         raise FileNotFoundError(f"IRT Difficulty file not found: {train_diff_file}")
    try:
        with open(train_diff_file, 'r') as file:
            irt_data = json.load(file)
        if 'diff' not in irt_data or len(irt_data['diff']) != len(train_data['id']):
             raise ValueError(f"IRT difficulty data mismatch for {taskname}. Expected {len(train_data['id'])}, got {len(irt_data.get('diff', []))}")
        train_data['difficulty'] = irt_data['diff']
    except Exception as e:
        print(f"Error loading IRT difficulties for {taskname}: {e}")
        raise

    # Calculate Word Rarity if requested
    if calculate_rarity:
        print("Calculating word rarity...")
        # Prepare flat list of strings for rarity calculation
        train_texts_for_rarity = []
        for s1, s2 in zip(train_data['s1'], train_data['s2']):
             text = str(s1)
             if s2 is not None: text += " " + str(s2)
             train_texts_for_rarity.append(text)
        train_data['example_rarity'] = get_example_rarities(train_texts_for_rarity)
        if len(train_data['example_rarity']) != len(train_data['id']):
             raise ValueError(f"Word rarity calculation mismatch for {taskname}.")

    # Perform 90/10 stratified split on the training data
    print(f"Performing 90/10 stratified split using seed: {seed}")
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
    try:
        # Use labels for stratification
        train_indices, dev_indices = next(splitter.split(train_data['s1'], train_data['label']))
    except ValueError as e:
         print(f"Warning: Stratified split failed for {taskname} (maybe too few samples per class?). Using non-stratified split. Error: {e}")
          # Fallback to non-stratified split
         from sklearn.model_selection import train_test_split
         indices = list(range(len(train_data['id'])))
         train_indices, dev_indices = train_test_split(indices, test_size=0.1, random_state=seed)


    # Create final dictionaries
    train_result = {
        'phrase': [[train_data['s1'][i], train_data['s2'][i]] for i in train_indices],
        'labels': [train_data['label'][i] for i in train_indices], # Use 'labels' key
        'pairID': [train_data['id'][i] for i in train_indices],
        'difficulty': [train_data['difficulty'][i] for i in train_indices], # IRT difficulty
    }
    if calculate_rarity:
        train_result['example_rarity'] = [train_data['example_rarity'][i] for i in train_indices]

    dev_result = {
        'phrase': [[train_data['s1'][i], train_data['s2'][i]] for i in dev_indices],
        'labels': [train_data['label'][i] for i in dev_indices], # Use 'labels' key
        'pairID': [train_data['id'][i] for i in dev_indices],
        'difficulty': [train_data['difficulty'][i] for i in dev_indices], # IRT difficulty for dev split
    }
    # If rarity was calculated, add it to dev_result too
    if calculate_rarity:
        dev_result['example_rarity'] = [train_data['example_rarity'][i] for i in dev_indices]


    # Test result uses the original GLUE dev set
    test_result = {
        'phrase': [[s1, s2] for s1, s2 in zip(test_data['s1'], test_data['s2'])],
        'labels': test_data['label'], # Use 'labels' key
        'pairID': test_data['id'],
        # No difficulty scores assumed for the original test/dev set
    }

    print(f"Split sizes: Train={len(train_result['labels'])}, Dev={len(dev_result['labels'])}, Test={len(test_result['labels'])}")
    return train_result, dev_result, test_result


# --- Curriculum Learning Data Selection ---

def get_epoch_training_data(ts, config, epoch):
    """
    Selects training data subset based on scheduler defined in config.
    Expects ts to be a TensorDataset with difficulty scores as the last tensor.
    Requires config object with attributes: training_scheduler, ordering,
                                          num_epochs, competency, min_train_length, balanced.
    """
    scheduler_type = getattr(config, 'training_scheduler', 'baseline') # Default to baseline
    ordering_type = getattr(config, 'ordering', 'easiest')
    is_balanced = getattr(config, 'balanced', False)
    min_len = getattr(config, 'min_train_length', 128)
    competency_param = getattr(config, 'competency', 5)
    # num_epochs for pacing calculation - using competency is preferred
    # num_epochs = getattr(config, 'num_epochs', 20) # Use if needed by specific pacing

    if not isinstance(ts, torch.utils.data.TensorDataset):
        raise TypeError(f"Input 'ts' must be a TensorDataset, got {type(ts)}")

    num_total_examples = len(ts)
    tensors = ts.tensors
    num_tensors = len(tensors)

    # Safely extract tensors, assuming difficulty is last
    if num_tensors < 3: # Need at least input_ids, attention_mask, labels
        raise ValueError(f"TensorDataset 'ts' has too few tensors ({num_tensors}). Expected at least 3.")

    input_ids = tensors[0]
    attention_mask = tensors[1]
    # Check for token_type_ids (if num_tensors > 3 and last is difficulty)
    has_token_type_ids = num_tensors > 4
    token_type_ids = tensors[2] if has_token_type_ids else None
    label_idx = 3 if has_token_type_ids else 2
    labels = tensors[label_idx]
    difficulty_idx = num_tensors - 1
    difficulties = tensors[difficulty_idx]
    print(f"Extracted tensors: Has TTI={has_token_type_ids}, Label Idx={label_idx}, Diff Idx={difficulty_idx}")


    if scheduler_type == 'baseline':
        print("Scheduler: baseline - Using full dataset.")
        data_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'difficulty': difficulties}
        if has_token_type_ids: data_dict['token_type_ids'] = token_type_ids
        return data_dict

    # --- Logic for Schedulers Requiring Sorting ---
    c_init = 0.01  # Initial competency proportion

    # Structures to hold sorted and selected data
    train_sorted = {'input_ids': None, 'attention_mask': None, 'token_type_ids': None, 'labels': None, 'difficulty': None}
    train_epoch = {'input_ids': None, 'attention_mask': None, 'token_type_ids': None, 'labels': None, 'difficulty': None}

    print(f"Sorting data based on difficulty (Ordering: {ordering_type})...")
    difficulty_np = difficulties.cpu().numpy()
    if ordering_type == 'easiest': diffs_sorted_idx = np.argsort(difficulty_np)
    elif ordering_type == 'hardest': diffs_sorted_idx = np.argsort(difficulty_np)[::-1]
    elif ordering_type == 'middleout': diffs_sorted_idx = np.argsort(np.abs(difficulty_np))
    else: raise NotImplementedError(f"Ordering '{ordering_type}' not implemented.")

    # Apply balancing if requested
    if is_balanced:
        print("Applying balanced sorting...")
        # (Balancing logic remains the same as before)
        per_label_lists = {}
        unique_labels = torch.unique(labels).cpu().numpy()
        for ul in unique_labels: per_label_lists[ul] = []
        for idx in diffs_sorted_idx:
            label_item = labels[idx].item()
            if label_item in per_label_lists: per_label_lists[label_item].append(idx)
        max_length = max(len(v) for v in per_label_lists.values()) if per_label_lists else 0
        train_2_idx = []
        for l in range(max_length):
            for k in sorted(per_label_lists.keys()):
                v = per_label_lists[k]
                if l < len(v): train_2_idx.append(v[l])
        if not train_2_idx: print("Warning: No indices selected after balancing. Using original sort.")
        else: diffs_sorted_idx = np.array(train_2_idx)

    # Populate train_sorted with sorted tensors
    train_sorted['input_ids'] = input_ids[diffs_sorted_idx]
    train_sorted['attention_mask'] = attention_mask[diffs_sorted_idx]
    if has_token_type_ids: train_sorted['token_type_ids'] = token_type_ids[diffs_sorted_idx]
    train_sorted['labels'] = labels[diffs_sorted_idx]
    train_sorted['difficulty'] = difficulties[diffs_sorted_idx]

    # --- Calculate number of examples for the epoch ---
    num_train = 0
    competency_epoch = max(1, competency_param) # Epoch number to reach 100% data

    if scheduler_type == 'linear': # Renamed from 'naacl-linear'
        if epoch >= competency_epoch: epoch_competency = 1.0
        else: epoch_competency = c_init + (1.0 - c_init) * (epoch / competency_epoch)
        num_train = int(epoch_competency * num_total_examples)
        print(f"Scheduler: linear - Epoch {epoch+1} competency={epoch_competency:.3f}")

    elif scheduler_type == 'root': # Renamed from 'naacl-root'
        if epoch >= competency_epoch: epoch_competency = 1.0
        else: epoch_competency = c_init + (1.0 - c_init) * np.sqrt(epoch / competency_epoch)
        num_train = int(epoch_competency * num_total_examples)
        print(f"Scheduler: root - Epoch {epoch+1} competency={epoch_competency:.3f}")

    # Add other schedulers like 'simple', 'theta' here if needed, ensuring they use config
    # elif scheduler_type == 'simple': ...
    # elif scheduler_type == 'theta': ... # Would need theta_hat from config

    else:
        raise NotImplementedError(f"Scheduler '{scheduler_type}' logic not defined here.")

    # Enforce minimum length and bounds
    num_train = max(min_len, num_train)
    num_available = len(train_sorted['input_ids']) # Number available after sorting/balancing
    num_train = min(num_train, num_available)
    print(f"Selecting {num_train} examples (Min: {min_len}, Available: {num_available}).")

    # Slice the sorted data
    train_epoch['input_ids'] = train_sorted['input_ids'][:num_train]
    train_epoch['attention_mask'] = train_sorted['attention_mask'][:num_train]
    if has_token_type_ids: train_epoch['token_type_ids'] = train_sorted['token_type_ids'][:num_train]
    train_epoch['labels'] = train_sorted['labels'][:num_train]
    train_epoch['difficulty'] = train_sorted['difficulty'][:num_train] # Keep difficulty slice

    return train_epoch