import streamlit as st
from qa_bot import answer_question
from text_summarizer import summarize_text

def main():
    st.title("AI Q&A and Text Summarizer")

    # Displaying options to the user for choosing the functionality
    st.sidebar.title("Choose Functionality")
    app_mode = st.sidebar.selectbox("Select Mode", ["Question Answering", "Text Summarizer"])

    if app_mode == "Question Answering":
        st.header("Question Answering")
        # Input context for the QA bot
        context = st.text_area("Enter context for question answering", "", help="Provide context for the question.")
        question = st.text_input("Enter your question", "", help="Ask a question based on the provided context.")

        # Button to submit and get the answer
        if st.button("Get Answer"):
            if context and question:
                answer = answer_question(question, context)
                st.write(f"**Answer:** {answer}")
            else:
                st.write("Please provide both context and a question.")
    
    elif app_mode == "Text Summarizer":
        st.header("Text Summarizer")
        # Input text for summarizing
        text_to_summarize = st.text_area("Enter text to summarize", "", help="Provide a long text to summarize.")

        # Button to submit and get the summary
        if st.button("Summarize Text"):
            if text_to_summarize:
                summary = summarize_text(text_to_summarize)
                st.write(f"**Summary:** {summary}")
            else:
                st.write("Please provide text to summarize.")

if __name__ == "__main__":
    main()
