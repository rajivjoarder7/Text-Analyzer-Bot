# qa_bot.py
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")

# Model checkpoint: google/flan-t5-large (780M parameters)
# For even higher reasoning on an A10/T4 16GB, use "google/flan-t5-xl"
MODEL_NAME = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
model.eval()

def answer_question(question: str, context: str) -> str:
    """
    Answers a question grounded strictly in the provided context using Flan-T5.
    """
    try:
        if not question.strip() or not context.strip():
            return "Please provide both a valid question and context paragraph."

        # Flan-T5 performs best with explicit instruction framing
        prompt = (
            f"Answer the following question based only on the provided context.\n\n"
            f"Context:\n{context.strip()}\n\n"
            f"Question: {question.strip()}\n\n"
            f"Answer:"
        )

        # Tokenize with truncation to avoid exceeding Flan-T5's 512-token position limit
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(device)

        # Deterministic generation: Beam search without diversity penalty or sampling noise
        with torch.no_grad():
            output_tokens = model.generate(
                **inputs,
                max_new_tokens=64,        # QA answers are typically concise (1-3 sentences)
                num_beams=4,              # Standard beam search for optimal token probability
                do_sample=False,          # Greedy / deterministic decoding prevents hallucination
                early_stopping=True,
                no_repeat_ngram_size=3,   # Prevents repetitive looping
                length_penalty=1.0
            )

        answer = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()

        if not answer:
            return "The model could not find an answer in the provided context."

        return answer

    except Exception as e:
        return f"Error during inference: {str(e)}"


if __name__ == "__main__":
    sample_context = (
        "Apollo 11 launched from Cape Kennedy on July 16, 1969, carrying Commander Neil Armstrong, "
        "Command Module Pilot Michael Collins, and Lunar Module Pilot Edwin 'Buzz' Aldrin. "
        "An estimated 650 million people watched Armstrong's televised image and heard his voice "
        "describe the event as 'one small step for [a] man, one giant leap for mankind' on July 20, 1969."
    )
    sample_question = "Who was the command module pilot on Apollo 11?"
    
    print("Question:", sample_question)
    print("Answer:", answer_question(sample_question, sample_context))
