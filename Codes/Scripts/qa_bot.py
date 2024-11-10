from transformers import pipeline
import torch
from datetime import datetime

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1  # -1 for CPU, 0 for GPU

# Load the question-answering pipeline with Flan-T5
model_name = "google/flan-t5-large"
qa_pipeline = pipeline("question-answering", model=model_name, device=device)

# Confidence threshold to handle out-of-context questions
CONFIDENCE_THRESHOLD = 0.5  # Adjusted threshold for improved context accuracy

def answer_question(question, context):
    """
    Generate an answer based on a question and context, with handling for out-of-context questions.
    """
    try:
        # Append current date and weekday to context for additional relevance
        today = datetime.now().strftime("%d-%b-%Y")
        day_of_week = datetime.now().strftime("%A")
        context += f"\nToday's date is {today}, today is {day_of_week}."

        # Use the QA pipeline
        result = qa_pipeline(question=question, context=context)

        # Verify if the score meets the confidence threshold
        if result['score'] < CONFIDENCE_THRESHOLD:
            return "I'm sorry, but that question seems unrelated to the provided context. Please try again."
        return result['answer']
    except Exception as e:
        return f"Error: {str(e)}"
