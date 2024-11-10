from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1  # -1 for CPU, 0 for GPU

# Load the question-answering pipeline
model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=model_name, device=device)

# Load a semantic similarity model for checking question-context relevance
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define thresholds
CONFIDENCE_THRESHOLD = 0.2
SIMILARITY_THRESHOLD = 0.2

def answer_question(question, context):
    """
    Generate an answer based on a question and context, with handling for out-of-context questions.
    """
    # Add today's date to the end of the context
    today_date = datetime.now().strftime("%d-%b-%Y")
    today_day = datetime.now().strftime("%A")
    context += f"\n\nToday's date is {today_date}, today is {today_day}."

    # Check question-context relevance
    question_embedding = similarity_model.encode(question, convert_to_tensor=True)
    context_embedding = similarity_model.encode(context, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(question_embedding, context_embedding).item()

    # If similarity is low, return an out-of-context message
    if similarity_score < SIMILARITY_THRESHOLD:
        return "The question appears to be unrelated to the provided context. Please ask a question relevant to the context."

    try:
        # Get answer from QA model
        result = qa_pipeline(question=question, context=context)

        # If the confidence score is low, treat as out-of-context
        if result['score'] < CONFIDENCE_THRESHOLD:
            return "I'm sorry, but that question seems unrelated to the provided context. Please try again."

        return result['answer']
    except Exception as e:
        return f"Error: {str(e)}"
