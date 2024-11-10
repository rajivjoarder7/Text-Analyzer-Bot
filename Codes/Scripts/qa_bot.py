from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from datetime import datetime
import textwrap  # Useful for splitting context into chunks

# Load pre-trained T5 model fine-tuned on Question Answering
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "t5-large"  # Use the large model for improved QA performance
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
model.to(device)

CONFIDENCE_THRESHOLD = 0.5  # Threshold for answer confidence (optional for further enhancements)

def chunk_text(text, max_tokens=100):
    """
    Splits text into chunks with a max token length.
    This helps model focus on smaller context sections.
    """
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        token_length = len(tokenizer.tokenize(sentence))
        if current_length + token_length > max_tokens:
            chunks.append('. '.join(current_chunk) + '.')  # Finish the chunk with a period
            current_chunk = [sentence]
            current_length = token_length
        else:
            current_chunk.append(sentence)
            current_length += token_length
    
    # Add any remaining sentences to the final chunk
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
        
    return chunks

def answer_question(question, context):
    """
    Answer the question using T5 with context chunking and beam diversity.
    """
    try:
        # Append current date info to the context for added relevance
        today = datetime.now().strftime("%d-%B-%Y")
        day_of_week = datetime.now().strftime("%A")
        context += f"\nToday's date is {today}, today is {day_of_week}."
        
        # Chunk the context into smaller pieces
        context_chunks = chunk_text(context, max_tokens=100)
        answers = []
        
        for chunk in context_chunks:
            # Tokenize question and chunked context
            inputs = tokenizer.encode(f"question: {question} context: {chunk}", return_tensors="pt").to(device)
            
            # Generate answer with diverse beams and stopping criteria
            summary_ids = model.generate(
                inputs,
                max_length=250,
                num_beams=8,               # Increased beams for more exploration
                num_beam_groups=4,         # Divides beams into groups for diversity
                diversity_penalty=1.0,     # Apply diversity to improve answer richness
                early_stopping=True        # Stops early when answer is complete
            )
            
            # Decode and add the generated answer from each chunk
            answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            if answer:  # Only add non-empty answers
                answers.append(answer)
        
        # Combine answers for a more comprehensive response
        final_answer = " ".join(answers)
        
        if not final_answer.strip():
            return "I'm sorry, but the model couldn't generate a relevant answer. Please try again."

        return final_answer

    except Exception as e:
        return f"Error: {str(e)}"
