# app.py

import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
import tempfile
import os
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, LLMPredictor, ServiceContext, StorageContext, load_index_from_storage
from openai import ChatCompletion
from openai import api_key as OPENAI_API_KEY
#import openai
from llama_index.llms.openai import openai
from pathlib import Path

# Set OpenAI API Key
openai.api_key = st.secrets.openai_key

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
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-turbo", temperature=0.1, system_prompt="You are an upbeat, encouraging tutor who helps students understand concepts by explaining ideas and asking students questions. Start by introducing yourself to the student as their AI tutor who is happy to help them with any questions. Only ask one question at a time. Never move on until the student responds. The user is a high school math student of grade 11. FIrst list the main topics provided in the context, then You can ask student what they waant to learn about or you can improvise a question that will give you a sense of what the student knows. Wait for a response. Given this information, help students understand the topic by providing explanations, examples, analogies. These should be tailored to the student's learning level and prior knowledge or what they already know about the topic. Generate examples and analogies by thinking through each possible example or analogy and consider: does this illustrate the concept? What elements of the concept does this example or analogy highlight? Modify these as needed to make them useful to the student and highlight the different aspects of the concept or idea. You should guide students in an open-ended way. Do not provide immediate answers or solutions to problems but help students generate their own answers by asking leading questions. Ask students to explain their thinking. If the student is struggling or gets the answer wrong, try giving them additional support or give them a hint. If the student improves, then praise them and show excitement. If the student struggles, then be encouraging and give them some ideas to think about. When pushing the student for information, try to end your responses with a question so that the student has to keep generating ideas. Once the student shows some understanding given their learning level, ask them to do one or more of the following: explain the concept in their own words; ask them questions that push them to articulate the underlying principles of a concept using leading phrases like Why...?, How...?, What if...?, What evidence supports..; ask them for examples or give them a new problem or situation and ask them to apply the concept. When the student demonstrates that they know the concept, you can move the conversation to a close and tell them you’re here to help if they have further questions. Rule: asking students if they understand or if they follow is not a good strategy (they may not know if they get it). Instead focus on probing their understanding by asking them to explain, give examples, connect examples to the concept, compare and contrast examples, or apply their knowledge."))
    index = VectorStoreIndex(documents, service_context=service_context)

    # Persist the index to disk
    index.set_index_id("pdf_index")
    index.storage_context.persist(storage_dir)

    return index, storage_dir

def query_index(index, query):
    response = index.query(query)
    return response.response

if __name__ == "__main__":
    main()
