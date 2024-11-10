import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from datetime import datetime
from qa_bot import answer_question  # Importing the QA logic
from text_summarizer import summarize_text 

def answer_question_gradio(question, context):
    return answer_question(question, context)

# Define Gradio functions for summarization
def summarize_text_gradio(text):
    # Assume you already have a text summarization function
    return summarize_text(text)

# Define the Gradio UI
with gr.Blocks() as demo:
    # Title of the app
    gr.Markdown("<h1>Text Analysis</h1>")
    
    # Instruction for minimum character input
    instruction_text = gr.Markdown("Please enter at least 250 characters for analysis to start.")
    
    # Textbox for the paragraph/context input
    context_input = gr.Textbox(label="Enter paragraph here:", placeholder="Type or paste your paragraph here...", lines=10)
    word_count = gr.Markdown("Word Count: 0")
    char_count = gr.Markdown("Character Count: 0")
    
    # Function to update word and character counts
    def update_counters(context):
        words = len(context.split())
        chars = len(context)
        return f"Word Count: {words}", f"Character Count: {chars}"
    
    context_input.change(fn=update_counters, inputs=context_input, outputs=[word_count, char_count])
    
    # Summarize button appears after 2000 characters
    summarize_button = gr.Button("Summarize", visible=False)
    summary_output = gr.Textbox(label="Summary", visible=False)
    
    def check_show_summarize(context):
        # Show Summarize button when character count is 2000 or more
        return gr.update(visible=len(context) >= 2000)
    
    # Show Summarize button based on character count
    context_input.change(fn=check_show_summarize, inputs=context_input, outputs=summarize_button)
    
    # Summarize button functionality
    def handle_summarize(context):
        summary = summarize_text_gradio(context)
        return gr.update(visible=True, value=summary)
    
    summarize_button.click(fn=handle_summarize, inputs=context_input, outputs=summary_output)
    
    # Ask Questions button appears after 250 characters
    question_button = gr.Button("Ask Questions", visible=False)
    question_input = gr.Textbox(label="Type your question here:", visible=False)
    ask_button = gr.Button("Ask", visible=False)
    answer_output = gr.Textbox(label="Answer", visible=False)
    
    def check_show_question_button(context):
        # Show Ask Questions button when character count is 250 or more
        return gr.update(visible=len(context) >= 250)
    
    # Check if Ask Questions button should be visible based on character count
    context_input.change(fn=check_show_question_button, inputs=context_input, outputs=question_button)
    
    # Show question input and ask button when Ask Questions button is clicked
    question_button.click(fn=lambda: gr.update(visible=True), outputs=question_input)
    question_button.click(fn=lambda: gr.update(visible=True), outputs=ask_button)
    
    # Handle the ask question functionality
    def handle_ask_click(question, context):
        answer = answer_question_gradio(question, context)
        return gr.update(visible=True, value=answer)
    
    ask_button.click(fn=handle_ask_click, inputs=[question_input, context_input], outputs=answer_output)

    # Reset button to clear all fields and reset visibility
    reset_button = gr.Button("Reset", elem_id="reset-button")
    def reset_all():
        return "", "", "Word Count: 0", "Character Count: 0", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
    
    reset_button.click(fn=reset_all, outputs=[context_input, question_input, word_count, char_count, summarize_button, question_input, answer_output, summary_output])

# Launch the app with a shareable link
demo.launch(share=True)
