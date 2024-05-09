import streamlit as st
import pdfplumber
import os
import tempfile
import openai
from llamaindex import Index  # Make sure to install and setup llamaindex

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize your LlamaIndex
index = Index("openai")

def extract_text_from_pdf(file):
    """Extracts text from a PDF file."""
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return "\n".join(filter(None, pages))

def process_pdf_files(uploaded_files):
    """Process each uploaded PDF file and extract text."""
    all_texts = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
            all_texts.append(text)
    return all_texts

def generate_embeddings(texts):
    """Generate embeddings for a list of texts."""
    embeddings = index.embed(texts)
    return embeddings

def query_model(embeddings):
    """Query the OpenAI GPT-4 Turbo model based on embeddings."""
    responses = []
    for embedding in embeddings:
        response = openai.Completion.create(
            model="gpt-4-turbo",
            prompt="",  # You can adjust the prompt if needed
            max_tokens=50,
            user="llama-user",  # Replace with your user identifier
            embeddings={
                "model": "text-embedding-ada-002",
                "data": embedding
            }
        )
        responses.append(response.choices[0].text.strip())
    return responses

def main():
    st.title("PDF Processor with LlamaIndex and GPT-4 Turbo")

    # File uploader allows user to add multiple files
    uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type=['pdf'])

    if st.button("Process PDFs"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF file.")
            return

        # Extract text from all the uploaded PDF files
        extracted_texts = process_pdf_files(uploaded_files)

        if not extracted_texts:
            st.error("No text could be extracted from the uploaded files.")
            return

        # Generate embeddings for the extracted text
        embeddings = generate_embeddings(extracted_texts)

        # Query the model with embeddings
        responses = query_model(embeddings)

        # Display the responses
        for i, response in enumerate(responses):
            st.subheader(f"Response for Document {i + 1}")
            st.write(response)

if __name__ == "__main__":
    main()
