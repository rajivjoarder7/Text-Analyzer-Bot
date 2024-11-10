from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from datetime import datetime

# Load pre-trained T5 model fine-tuned on Question Answering
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "t5-large"  # Use the large model for improved QA performance
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

CONFIDENCE_THRESHOLD = 0.5  # Threshold for answer confidence (optional for further enhancements)

def chunk_text_by_token_length(text, max_tokens=100):
    """
    Splits text into chunks based on max token length without unnecessary sentence breaks.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        token_length = len(tokenizer.tokenize(word))
        if current_length + token_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = token_length
        else:
            current_chunk.append(word)
            current_length += token_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def answer_question(question, context):
    """
    Answer the question using T5 with controlled context chunking and beam diversity.
    """
    try:
        # Append current date info to the context for added relevance
        today = datetime.now().strftime("%d-%B-%Y")
        day_of_week = datetime.now().strftime("%A")
        context += f"\nToday's date is {today}, today is {day_of_week}."
        
        # Use token-length based chunking
        context_chunks = chunk_text_by_token_length(context, max_tokens=100)
        answers = []
        
        for chunk in context_chunks:
            # Tokenize question and chunked context
            inputs = tokenizer.encode(f"question: {question} context: {chunk}", return_tensors="pt").to(device)
            
            # Generate answer with diverse beams and stopping criteria
            summary_ids = model.generate(
                inputs,
                max_length=250,
                num_beams=8,
                num_beam_groups=4,
                diversity_penalty=1.0,
                early_stopping=True
            )
            
            # Decode the generated answer from each chunk
            answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            if answer:
                answers.append(answer)
        
        # Filter for redundant additions
        combined_answer = " ".join(answers)
        for word in ["Chinese", "China"]:  # Example keywords to remove if redundant
            if combined_answer.count(word) > 1:
                combined_answer = combined_answer.replace(f" and {word}", "")
                combined_answer = combined_answer.replace(f", {word}", "")

        if not combined_answer.strip():
            return "I'm sorry, but the model couldn't generate a relevant answer. Please try again."

        return combined_answer

    except Exception as e:
        return f"Error: {str(e)}"
