import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
import tempfile
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
import openai
from llama_index.llms.openai import OpenAI
from pathlib import Path
import shutil
from io import BytesIO

# Set OpenAI API Key
openai.api_key = st.secrets["openai_key"]

def main():
    st.title("DocTalk, talk to your docs  - Developed by Abhyas Manne")
    st.write("Upload one or more multiple PDF files")

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

                index, storage_dir = index_pdf(merged_pdf_path)

                if index:
                    st.write("PDF indexed successfully! You can now ask questions.")
                    summary = st.session_state.chat_engine.chat(Summarize the document)
                    if "messages" not in st.session_state.keys(): # Initialize the chat messages history
                        st.session_state.messages = [
                            {"role": "assistant", "content": summary}
                        ]
                    
                    if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
                            st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
                    
                    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
                        st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    for message in st.session_state.messages: # Display the prior chat messages
                        with st.chat_message(message["role"]):
                            st.write(message["content"])
                    
                    # If last message is not from assistant, generate a new response
                    if st.session_state.messages[-1]["role"] != "assistant":
                        with st.chat_message("assistant"):
                            with st.spinner("Thinking..."):
                                response = st.session_state.chat_engine.chat(prompt)
                                st.write(response.response)
                                message = {"role": "assistant", "content": response.response}
                                st.session_state.messages.append(message) # Add response to message history

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
            reader = PdfReader(BytesIO(uploaded_file.getvalue()))
            for page_num in range(len(reader.pages)):
                pdf_writer.add_page(reader.pages[page_num])
        pdf_writer.write(temp_merged_pdf)
        temp_merged_pdf.close()
        return temp_merged_pdf.name
    except Exception as e:
        st.error(f"An error occurred while merging PDFs: {e}")
        return None
    finally:
        temp_merged_pdf.close()
@st.cache_resource(show_spinner=False)
def index_pdf(pdf_path):
    try:
        storage_dir = Path(tempfile.mkdtemp())
        pdf_dir = storage_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(pdf_path, pdf_dir / "merged_document.pdf")

        with st.spinner("Indexing documents..."):
            docs = SimpleDirectoryReader(pdf_dir).load_data()
            service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-turbo", temperature=0.1, system_prompt="You are assistant researcher who is helping young scholars read scientific articles. Initially provide a concise summary of the uploaded documents, then ask what the user wants to know more about. If the information is not within the context provided, then reply that you could not find the relavent information in the context, do not hallucinate." ))
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            index.set_index_id("pdf_index")
            index.storage_context.persist(storage_dir)

        return index, storage_dir
    except Exception as e:
        st.error(f"An error occurred while indexing PDF: {e}")
        return None, None

if __name__ == "__main__":
    main()
