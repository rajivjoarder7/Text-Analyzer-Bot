from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Load the BART model and tokenizer for Question Generation
model_name = "facebook/bart-large"  # We use bart-large, a well-tested and robust model
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

    # Here, instead of generating an extractive answer, we output the model's generative answer directly

    # Compute confidence score (this is a simplistic approach, further optimization could be done)
    answer_score = torch.nn.functional.cosine_similarity(inputs['input_ids'], outputs[0], dim=-1).mean().item()

    if answer_score < confidence_threshold:
        return "I'm sorry, but that question seems unrelated to the provided context. Please try again."
    
    return answer
