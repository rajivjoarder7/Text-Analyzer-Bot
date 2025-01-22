import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the PEGASUS tokenizer and model for abstractive summarization
model_name = "google/pegasus-xsum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to the available device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")

model = model.to(device)

def summarize_text(text):
    # Calculate desired length based on input length
    input_length = len(text.split())
    target_length = int(0.35 * input_length)  # Set to ~35% of input length
    max_length = min(0.4 * input_length, 250)  # Upper bound (40%) with absolute max
    min_length = max(0.3 * input_length, 50)   # Lower bound (30%) with absolute min

    # Ensure integer values for max_length and min_length
    max_length = int(max_length)
    min_length = int(min_length)
    
    # Tokenize the input for PEGASUS model
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the correct device

    # Generate summary using PEGASUS with dynamic length limits
    summary_ids = model.generate(
        **inputs,
        max_length=max_length,      # Dynamic maximum length
        min_length=min_length,      # Dynamic minimum length
        num_beams=5,                # Beam search for better quality
        length_penalty=2.0,         # Discourage overly short summaries
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
