from transformers import BartForConditionalGeneration, BartTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the BART model and tokenizer for Question Generation
model_name = "facebook/bart-large-qg"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Function to answer questions
def answer_question(question, context, confidence_threshold=0.5):
    # Tokenize the question and context together
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    
    # Generate the answer using the model
    outputs = model.generate(inputs['input_ids'], max_length=100, num_beams=4, early_stopping=True)
    
    # Decode the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Compute confidence score using cosine similarity (can be refined further)
    question_embedding = tokenizer(question, return_tensors="pt")['input_ids']
    context_embedding = tokenizer(context, return_tensors="pt")['input_ids']
    answer_embedding = tokenizer(answer, return_tensors="pt")['input_ids']
    
    # Calculate cosine similarity between question and context embeddings
    sim_score = cosine_similarity(question_embedding.detach().numpy(), context_embedding.detach().numpy())
    answer_score = cosine_similarity(answer_embedding.detach().numpy(), context_embedding.detach().numpy())
    
    # Confidence threshold check
    if answer_score[0][0] < confidence_threshold:
        return "I'm sorry, but that question seems unrelated to the provided context. Please try again."
    
    return answer

