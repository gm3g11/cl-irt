from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading {model_name}...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for H100
    device_map="auto"  # Automatically uses your GPU
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# GSM8K question
question = "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?"

# Format with CoT prompt
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
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
    temperature=0.0,
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
