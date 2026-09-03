import os
import gradio as gr
from huggingface_hub import InferenceClient

# Uses Meta's industry-leading Llama-3-8B-Instruct via Hugging Face's free Serverless Inference
# You can also use "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Optional: Add your HF token if you hit rate limits, or leave blank for public free tier
HF_TOKEN = os.getenv("HF_TOKEN", None)
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)

SYSTEM_PROMPT = """You are an elite, executive-level Question Answering engine.
Your objective is to answer the user's question based EXCLUSIVELY on the provided context.

CRITICAL RULES:
1. Provide the complete, fully-qualified answer including organizations, roles, and entities (e.g., instead of just "head coach", say "Real Madrid head coach").
2. Answer concisely and directly (1 to 2 sentences maximum).
3. Do not assume or extrapolate anything outside the context.
4. If the answer cannot be determined from the context, state: "The provided context does not contain sufficient information to answer this question."
"""

def answer_question(question: str, context: str):
    if not question.strip() or not context.strip():
        return "Please provide both a question and reference context."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{context.strip()}\n\nQuestion:\n{question.strip()}\n\nAnswer:"
        }
    ]

    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=80,
            temperature=0.1,  # Near-zero temperature ensures strict factual consistency
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback to text_generation if chat_completion endpoint warms up
        try:
            prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\nContext:\n{context}\n\nQuestion:\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
            res = client.text_generation(prompt, max_new_tokens=80, temperature=0.1)
            return res.strip()
        except Exception as inner_e:
            return f"Service notification: {str(inner_e)}"

with gr.Blocks(theme=gr.themes.Soft(primary_hue="slate")) as demo:
    gr.Markdown("""
    # 🏛️ Enterprise Contextual Intelligence Bot
    ### Designed for Executive Precision & Zero Hallucination
    """)

    with gr.Row():
        with gr.Column(scale=3):
            context_input = gr.Textbox(
                lines=8,
                label="Source Document / Business Context",
                placeholder="Paste paragraph or case study context here..."
            )
            question_input = gr.Textbox(
                lines=2,
                label="Executive Question",
                placeholder="e.g. Who is Jose Mourinho?"
            )
            submit_btn = gr.Button("Generate Qualified Answer", variant="primary", size="lg")

        with gr.Column(scale=2):
            output_answer = gr.Textbox(label="Precise Answer", lines=4, interactive=False)

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, context_input],
        outputs=[output_answer]
    )

    gr.Examples(
        examples=[
            [
                "Who is Jose Mourinho?",
                "Real Madrid head coach José Mourinho publicly criticised defender Raúl Asencio after another off-field controversy involving the 23-year-old Spaniard. The Portuguese coach stressed that he has no complaints about Asencio's attitude at the training ground, but made it clear that a player represents the club at all times."
            ]
        ],
        inputs=[question_input, context_input]
    )

if __name__ == "__main__":
    demo.launch()
