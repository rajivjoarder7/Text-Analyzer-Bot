# qa_bot.py

from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from datetime import datetime

# Load pre-trained T5 model fine-tuned on Question Answering
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set device based on availability
model_name = "t5-large"  # Large T5 model suitable for QA tasks
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

# Confidence threshold for generating QA
CONFIDENCE_THRESHOLD = 0.5

def answer_question(question, context):
    """
    Answer the question using T5 for abstractive generation.
    """
    try:
        today = datetime.now().strftime("%d-%B-%Y")
        day_of_week = datetime.now().strftime("%A")
        context += f"\nToday's date is {today}, today is {day_of_week}."
        
        # Tokenize the input
        inputs = tokenizer.encode(f"question: {question} context: {context}", return_tensors="pt").to(device)
        
        # Generate the answer with enhanced beam search and diversity settings
        summary_ids = model.generate(
            inputs,
            max_length=250,
            num_beams=8,               # Increased beams for more exploration
            num_beam_groups=4,         # Divides beams into groups for diversity
            diversity_penalty=1.0,     # Apply diversity to improve answer richness
            early_stopping=True        # Stops early when answer is complete
        )
        
        # Decode and return the answer
        answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        if not answer.strip():
            return "I'm sorry, but the model couldn't generate a relevant answer. Please try again."

        return answer
    except Exception as e:
        return f"Error: {str(e)}"
