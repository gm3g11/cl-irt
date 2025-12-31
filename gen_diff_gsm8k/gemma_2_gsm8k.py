from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Explicitly set cache directory
cache_dir = HF_HOME

# Load model and tokenizer
model_name = "google/gemma-2-9b-it"
print(f"Loading {model_name}...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=cache_dir
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
    cache_dir=cache_dir
)

# GSM8K question
question = "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"

# Format with CoT prompt
messages = [
    {"role": "user", "content": f"{question}\nLet's think step by step."}
]

# Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    do_sample=False
)

# Decode
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("Question:", question)
print("\nModel Response:")
print(response)
