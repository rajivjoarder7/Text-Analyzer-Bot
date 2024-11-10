from transformers import BertForQuestionAnswering, BertTokenizer, pipeline
import torch
from datetime import datetime

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1  # -1 for CPU, 0 for GPU

# Load the tokenizer and model for SQuAD 2.0 (BERT)
model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

# Initialize the QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer, device=device)

# Confidence threshold to handle out-of-context questions
CONFIDENCE_THRESHOLD = 0.6  # Adjust this threshold based on your needs

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

        # Check if the confidence score is above the threshold
        if result['score'] < CONFIDENCE_THRESHOLD:
            return "I'm sorry, but that question seems unrelated to the provided context. Please try again."

        return result['answer']
    except Exception as e:
        return f"Error: {str(e)}"
