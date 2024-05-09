# app.py

import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
import tempfile
import os
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from openai import ChatCompletion
from openai import api_key as OPENAI_API_KEY
import openai
from pathlib import Path

# Set OpenAI API Key
openai.api_key = "YOUR_OPENAI_API_KEY"

# Define Streamlit app
def main():
    st.title("RAG System with Streamlit, LLaMA-Index, and GPT-4")
    st.write("Upload multiple PDF files to merge and query using GPT-4.")

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type='pdf')

    if uploaded_files:
        st.write(f"{len(uploaded_files)} PDF files uploaded.")
        merged_pdf_path = merge_pdfs(uploaded_files)
        st.write("PDF files merged successfully!")

        with open(merged_pdf_path, "rb") as file:
            st.download_button(
                label="Download Merged PDF",
                data=file,
                file_name="merged_document.pdf",
                mime="application/pdf"
            )

        # Indexing the merged PDF via llama-index
        index, storage_dir = index_pdf(merged_pdf_path)

        st.write("PDF indexed successfully! You can now ask questions.")

        # Chat functionality
        user_input = st.text_input("Ask a question about the merged PDF:")
        if user_input:
            response = query_index(index, user_input)
            st.write(f"Answer: {response}")

def merge_pdfs(files):
    pdf_writer = PdfWriter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_merged_pdf:
        for uploaded_file in files:
            pdf_reader = PdfReader(uploaded_file)
            for page_num in range(len(pdf_reader.pages)):
                pdf_writer.add_page(pdf_reader.pages[page_num])
        pdf_writer.write(temp_merged_pdf)
        temp_merged_pdf_path = temp_merged_pdf.name
    return temp_merged_pdf_path

def index_pdf(pdf_path):
    # Create a temporary directory for storing indexed data
    storage_dir = Path(tempfile.mkdtemp())
    pdf_dir = storage_dir / "pdfs"
    pdf_dir.mkdir()

    # Copy the merged PDF to the indexing directory
    merged_pdf_name = Path(pdf_path).name
    target_pdf_path = pdf_dir / merged_pdf_name
    os.rename(pdf_path, target_pdf_path)

    # Read and index the PDF
    documents = SimpleDirectoryReader(pdf_dir).load_data()
    llm_predictor = LLMPredictor(temperature=0)
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
    index = GPTVectorStoreIndex(documents, service_context=service_context)

    # Persist the index to disk
    index.set_index_id("pdf_index")
    index.storage_context.persist(storage_dir)

    return index, storage_dir

def query_index(index, query):
    response = index.query(query)
    return response.response

if __name__ == "__main__":
    main()
