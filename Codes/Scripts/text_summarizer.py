from transformers import pipeline
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1  # -1 for CPU, 0 for GPU

# Load the summarization pipeline
model_name = "facebook/bart-large-cnn"
summarizer = pipeline("summarization", model=model_name, device=device)

def summarize_text(text, summary_ratio=0.3):
    try:
        summary = summarizer(text, max_length=int(len(text.split()) * summary_ratio), min_length=50, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {str(e)}"
