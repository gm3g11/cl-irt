import os
import glob
import numpy as np

# the only valid GLUE splits we care about
DATASETS = {"rte", "mnli", "mrpc", "qqp", "sst2", "qnli"}

# structure to hold: results[model][dataset] = list of (time_min, accuracy)
results = {}

for folder in glob.glob("glue_baseline*"):
    # for each file like final_stats_*.json
    for path in glob.glob(os.path.join(folder, "final_stats*.json")):
        fname = os.path.basename(path)
        # strip prefix / suffix
        core = fname[len("final_stats_"):-len(".json")]
        
        # find which dataset this is
        ds = None
        for d in DATASETS:
            token = f"_{d}_"
            if token in core:
                ds = d
                model_part, rest = core.split(token, maxsplit=1)
                break
        if ds is None:
            # not one of our GLUE sets
            continue
        
        # extract model name
        # if there's an underscore in model_part, assume it's the prefix + actual model:
        #   e.g. "meta-llama_Meta-Llama-3.1-8B" → "Meta-Llama-3.1-8B"
        if "_" in model_part:
            model_name = model_part.split("_")[-1]
        else:
            model_name = model_part
        
        # guard against edge-case where model==dataset
        if model_name == ds:
            model_name = f"{model_name}_{ds}"
        
        # now parse the rest: "{time}s_{epochs}epochs_{accuracy}"
        time_str, epochs_str, acc_str = rest.split("_")
        # time in seconds → minutes
        time_sec = float(time_str.rstrip("s"))
        time_min = time_sec / 60.0
        # accuracy
        acc = float(acc_str)
        
        # accumulate
        results.setdefault(model_name, {}) \
               .setdefault(ds, []) \
               .append((time_min, acc))

# compute and print summary stats
print(f"{'Model':<30} {'Dataset':<6}  {'#runs':>5}  {'Time μ±σ (min)':>20}  {'Acc μ±σ':>15}")
print("-"*80)
for model, ds_map in sorted(results.items()):
    for ds, vals in sorted(ds_map.items()):
        times = np.array([t for t, a in vals])
        accs  = np.array([a for t, a in vals])
        μ_t, σ_t = times.mean(), times.std(ddof=0)
        μ_a, σ_a = accs.mean(),  accs.std(ddof=0)
        print(f"{model:<30} {ds:<6}  {len(vals):>5d}    "
              f"{μ_t:7.2f} ± {σ_t:5.2f}     "
              f"{μ_a:6.4f} ± {σ_a:6.4f}")

