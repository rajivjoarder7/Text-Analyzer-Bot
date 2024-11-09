from transformers import pipeline
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1  # -1 for CPU, 0 for GPU

# Load the question-answering pipeline
model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=model_name, device=device)

CONFIDENCE_THRESHOLD = 0.6

def answer_question(question, context):
    """
    Generate an answer based on a question and context, with handling for out-of-context questions.
    """
    try:
        result = qa_pipeline(question=question, context=context)
        if result['score'] < CONFIDENCE_THRESHOLD:
            return "I'm sorry, but that question seems unrelated to the provided context. Please try again."
        return result['answer']
    except Exception as e:
        return f"Error: {str(e)}"