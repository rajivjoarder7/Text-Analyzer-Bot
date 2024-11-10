

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
        
        # Generate the answer with complex parameters
        summary_ids = model.generate(
            inputs,
            max_length=250,
            temperature=0.7,  # Adjust temperature for varied responses
            top_p=0.9, 
            num_beams=12,              # Increase beams for more exploration
            num_beam_groups=6,         # Increase beam groups for diversity
            diversity_penalty=1.5,     # Encourage diversity
            repetition_penalty=2.0,    # Penalize repetition to avoid extractive answers
            length_penalty=1.2,        # Slightly increase length for completeness
            early_stopping=True        # Stops early when the answer is complete
        )
        
        # Decode and return the answer
        answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        if not answer.strip():
            return "I'm sorry, but the model couldn't generate a relevant answer. Please try again."

        return answer
    except Exception as e:
        return f"Error: {str(e)}"
