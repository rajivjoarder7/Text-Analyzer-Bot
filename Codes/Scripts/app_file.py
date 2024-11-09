import gradio as gr
from qa_bot import answer_question
from text_summarizer import summarize_text
import time

# Helper function to get word and character count
def get_counts(text):
    words = len(text.split())
    chars = len(text)
    return words, chars

# Define the Gradio UI
def create_ui():
    with gr.Blocks(css=".container { color: #333; background-color: #f3f6fb; }") as demo:
        gr.Markdown("<h1 style='text-align: center; color: #0056b3;'>Text Analysis</h1>")
        
        # Text area for paragraph input
        paragraph = gr.Textbox(placeholder="Type or paste your paragraph here...", lines=5, interactive=True)
        
        # Word and character counters
        word_count = gr.Markdown(value="Words: 0", visible=True)
        char_count = gr.Markdown(value="Characters: 0", visible=True)
        
        # Button for summarization and question answering
        summarize_button = gr.Button("Summarize", visible=False)
        ask_question_button = gr.Button("Ask Questions", visible=False)
        
        # Question input and answer display
        question_input = gr.Textbox(placeholder="Enter your question here", visible=False)
        ask_button = gr.Button("Ask", visible=False)
        answer_display = gr.Markdown(visible=False)
        
        # Home and Reset buttons
        home_button = gr.Button("Home", visible=False)
        reset_button = gr.Button("Reset", visible=True)
        
        # Word/character count and button logic based on text length
        def update_ui(text):
            words, chars = get_counts(text)
            word_count.update(f"Words: {words}")
            char_count.update(f"Characters: {chars}")
            
            if chars >= 200:
                ask_question_button.update(visible=True)
            else:
                ask_question_button.update(visible=False)

            if chars >= 2000:
                summarize_button.update(visible=True)
            else:
                summarize_button.update(visible=False)
            
            return gr.update()
        
        paragraph.change(update_ui, paragraph)
        
        # Summarization functionality
        def summarize_text_gradio(text):
            summary = summarize_text(text)
            summary_display.update(summary)
            home_button.update(visible=True)
        
        summarize_button.click(summarize_text_gradio, inputs=paragraph, outputs=None)

        # Question-answering functionality
        def ask_question_gradio(question, context):
            answer = answer_question(question, context)
            answer_display.update(answer)
            question_input.update(value="")
        
        ask_button.click(ask_question_gradio, inputs=[question_input, paragraph], outputs=None)

        # Reset functionality
        def reset_ui():
            paragraph.update(value="")
            word_count.update("Words: 0")
            char_count.update("Characters: 0")
            summarize_button.update(visible=False)
            ask_question_button.update(visible=False)
            question_input.update(visible=False)
            ask_button.update(visible=False)
            answer_display.update(visible=False)
            home_button.update(visible=False)

        reset_button.click(reset_ui)

    return demo

# Run the Gradio interface
ui = create_ui()
ui.launch(share=True)
