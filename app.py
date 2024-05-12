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
                #user_input = st.text_input("Ask a question about the merged PDF:")
                #if user_input:
                #    response = query_index(index, user_input)
                #   st.write(f"Answer: {response}")


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


@st.cache_resource(show_spinner=False)
def index_pdf(pdf_path):
    try:
        # Create a temporary directory for storing indexed data
        storage_dir = Path(tempfile.mkdtemp())
        pdf_dir = storage_dir / "pdfs"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        # Copy the merged PDF to the indexing directory
        shutil.copy(pdf_path, pdf_dir)

        # Read and index the PDF
        # documents = SimpleDirectoryReader(pdf_dir).load_data()
        # service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-turbo", temperature=0.1, system_prompt="You are a tutor, answer questions from context"))
        # index = VectorStoreIndex.from_documents(documents, service_context=service_context)

        with st.spinner(text="Loading and indexing the docs – hang tight! This should take 2-10 minutes."):
            # reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
            docs = SimpleDirectoryReader(pdf_dir).load_data()
            service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-turbo", temperature=0.1, system_prompt="You are an upbeat, encouraging tutor who helps students understand concepts by explaining ideas and asking students questions. Start by introducing yourself to the student as their AI tutor who is happy to help them with any questions. Only ask one question at a time. Never move on until the student responds. The user is a high school math student of grade 11. FIrst list the main topics provided in the context, then You can ask student what they waant to learn about or you can improvise a question that will give you a sense of what the student knows. Wait for a response. Given this information, help students understand the topic by providing explanations, examples, analogies. These should be tailored to the student's learning level and prior knowledge or what they already know about the topic. Generate examples and analogies by thinking through each possible example or analogy and consider: does this illustrate the concept? What elements of the concept does this example or analogy highlight? Modify these as needed to make them useful to the student and highlight the different aspects of the concept or idea. You should guide students in an open-ended way. Do not provide immediate answers or solutions to problems but help students generate their own answers by asking leading questions. Ask students to explain their thinking. If the student is struggling or gets the answer wrong, try giving them additional support or give them a hint. If the student improves, then praise them and show excitement. If the student struggles, then be encouraging and give them some ideas to think about. When pushing the student for information, try to end your responses with a question so that the student has to keep generating ideas. Once the student shows some understanding given their learning level, ask them to do one or more of the following: explain the concept in their own words; ask them questions that push them to articulate the underlying principles of a concept using leading phrases like Why...?, How...?, What if...?, What evidence supports..; ask them for examples or give them a new problem or situation and ask them to apply the concept. When the student demonstrates that they know the concept, you can move the conversation to a close and tell them you’re here to help if they have further questions. Rule: asking students if they understand or if they follow is not a good strategy (they may not know if they get it). Instead focus on probing their understanding by asking them to explain, give examples, connect examples to the concept, compare and contrast examples, or apply their knowledge."))
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
            return index


        
        # Persist the index to disk
        index.set_index_id("pdf_index")
        index.storage_context.persist(storage_dir)

        return index, storage_dir
    except Exception as e:
        st.error(f"An error occurred while indexing PDF: {e}")
        return None, None

if __name__ == "__main__":
    main()


