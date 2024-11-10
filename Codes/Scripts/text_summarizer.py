import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the PEGASUS tokenizer and model for abstractive summarization
model_name = "google/pegasus-xsum"  # "xsum" version for more detailed summaries
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to the available device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def summarize_text(text):
    # Tokenize the input for PEGASUS model
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to the correct device

    # Generate summary using PEGASUS
    summary_ids = model.generate(
        **inputs,
        max_length=200,               # Set this higher for longer summaries
        min_length=60,                # Ensure minimum summary length
        num_beams=5,                  # Beam search for better quality
        length_penalty=2.0,           # Discourage very short summaries
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
