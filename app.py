import streamlit as st
import pdfplumber
import openai
from llamaindex import LlamaIndex
import os
import tempfile

# Initialize Streamlit app
st.title('RAG Application with Streamlit, LlamaIndex, and OpenAI')

# OpenAI API Key (Ensure to set your OPENAI_API_KEY in your environment variables)
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    st.warning('OpenAI API key is not set in environment variables.')
    st.stop()

openai.api_key = api_key

# LlamaIndex Embeddings Model
try:
    index = LlamaIndex(model="text-embedding-ada-002")
except Exception as e:
    st.error(f"Failed to load LlamaIndex model: {str(e)}")
    st.stop()

# File Uploader
uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])
if not uploaded_files:
    st.warning('Please upload at least one PDF file.')
    st.stop()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        pages = [page.extract_text() for page in pdf.pages]
    return "\n".join(filter(None, pages))

# Process uploaded files and extract text
texts = []
for uploaded_file in uploaded_files:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        texts.append(extract_text_from_pdf(temp_file.name))
        os.unlink(temp_file.name)

# Display extracted texts (optional, can be commented out in production)
# st.subheader("Extracted Text from PDFs")
# for i, text in enumerate(texts):
#     st.write(f"### PDF {i+1}")
#     st.write(text[:500])  # Display first 500 characters of each text

# Join all texts to form the context
complete_text = " ".join(texts)

# Generate embeddings for the complete text
try:
    embeddings = index.encode(complete_text)
except Exception as e:
    st.error(f"Failed to generate embeddings: {str(e)}")
    st.stop()

# Retrieve relevant context based on embeddings
# For simplicity, we use the complete text as the context
context = complete_text[:4000]  # Limit context to 4000 characters

# Form the prompt (you can customize this as needed)
prompt = f"Summarize the following content:\n\n{context}"

# Use GPT-4 Turbo to generate the response
try:
    response = openai.Completion.create(
        model="gpt-4-turbo",
        prompt=prompt,
        max_tokens=150,
        temperature=0.7
    )
    generated_text = response.get('choices')[0].get('text').strip()
except Exception as e:
    st.error(f"Failed to call OpenAI GPT-4 Turbo: {str(e)}")
    st.stop()

# Display the generated text
st.subheader("Generated Summary")
st.write(generated_text)
