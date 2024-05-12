import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
import tempfile
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from openai import Completion
import openai
from llama_index.llms.openai import OpenAI
from pathlib import Path
import shutil
from io import BytesIO

# Set OpenAI API Key
openai.api_key = st.secrets["openai_key"]

def main():
    st.title("RAG System with Streamlit, LLaMA-Index, and GPT-4")
    st.write("Upload multiple PDF files to merge and query using GPT-4.")

    uploaded_files = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=['pdf'])

    if uploaded_files:
        st.write(f"{len(uploaded_files)} PDF files uploaded.")
        merged_pdf_path = merge_pdfs(uploaded_files)
        
        if merged_pdf_path:
            st.write("PDF files merged successfully!")
            try:
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

                    if "messages" not in st.session_state:
                        st.session_state.messages = []

                    if "chat_engine" not in st.session_state:  # Initialize the chat engine
                        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
                    
                    if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
                        st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    for message in st.session_state.messages:  # Display the prior chat messages
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
                    
                    # If last message is not from assistant, generate a new response
                    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                response = st.session_state.chat_engine.chat(prompt)
                                st.write(response.response)
                                st.session_state.messages.append({"role": "assistant", "content": response.response})
                
                # Clean up temporary files
                if storage_dir:
                    shutil.rmtree(storage_dir)
            finally:
                os.remove(merged_pdf_path)

def merge_pdfs(files):
    pdf_writer = PdfWriter()
    temp_merged_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
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
    finally:
        temp_merged_pdf.close()

def index_pdf(pdf_path):
    try:
        # Create a temporary directory for storing indexed data
        storage_dir = Path(tempfile.mkdtemp())
        pdf_dir = storage_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        # Copy the merged PDF to the indexing directory
        shutil.copy(pdf_path, pdf_dir)

        with st.spinner(text="Loading and indexing the docs – hang tight! This should take 2-10 minutes."):
            docs = SimpleDirectoryReader(pdf_dir).load_data()
            service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-turbo", temperature=0.1, system_prompt="You are an upbeat, encouraging tutor..."))
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        
        # Persist the index to disk
        index.set_index_id("pdf_index")
        index.storage_context.persist(storage_dir)

        return index, storage_dir
    except Exception as e:
        st.error(f"An error occurred while indexing PDF: {e}")
        return None, None

if __name__ == "__main__":
    main()
