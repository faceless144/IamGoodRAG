import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
import tempfile
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext, StorageContext
from openai import Completion
import openai
from llama_index.llms.openai import OpenAI
from pathlib import Path
import shutil
from io import BytesIO

# Set OpenAI API Key
openai.api_key = st.secrets["openai_key"]

# Define Streamlit app
def main():
    st.title("RAG System with Streamlit, LLaMA-Index, and GPT-4")
    st.write("Upload multiple PDF files to merge and query using GPT-4.")

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
        st.write(f"{len(uploaded_files)} PDF files uploaded.")
        merged_pdf_path = merge_pdfs(uploaded_files)
        if merged_pdf_path:
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

            if index:
                st.write("PDF indexed successfully! You can now ask questions.")

                # Chat functionality
                user_input = st.text_input("Ask a question about the merged PDF:")
                if user_input:
                    response = query_index(index, user_input)
                    st.write(f"Answer: {response}")

            # Clean up temporary files
            shutil.rmtree(storage_dir)

def merge_pdfs(files):
    pdf_writer = PdfWriter()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_merged_pdf:
            for uploaded_file in files:
                # Read the uploaded file as a stream
                reader = PdfReader(BytesIO(uploaded_file.getvalue()))
                for page_num in range(len(reader.pages)):
                    pdf_writer.add_page(reader.pages[page_num])
            pdf_writer.write(temp_merged_pdf)
            return temp_merged_pdf.name
    except Exception as e:
        st.error(f"An error occurred while merging PDFs: {e}")
        return None

def index_pdf(pdf_path):
    try:
        # Create a temporary directory for storing indexed data
        storage_dir = Path(tempfile.mkdtemp())
        pdf_dir = storage_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        # Copy the merged PDF to the indexing directory
        shutil.copy(pdf_path, pdf_dir)

        # Read and index the PDF
        documents = SimpleDirectoryReader(pdf_dir).load_data()
        service_context = ServiceContext(llm=OpenAI(model="gpt-4-turbo", temperature=0.1))
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)

        # Persist the index to disk
        index.set_index_id("pdf_index")
        index.storage_context.persist(storage_dir)

        return index, storage_dir
    except Exception as e:
        st.error(f"An error occurred while indexing PDF: {e}")
        return None, None

def query_index(index, query):
    try:
        response = index.query(query)
        return response.response
    except Exception as e:
        st.error(f"An error occurred while querying the index: {e}")
        return "Error in querying the index"

if __name__ == "__main__":
    main()
