import streamlit as st
import pdfplumber
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
import os
from tempfile import NamedTemporaryFile

# Initialize Streamlit app

st.set_page_config(page_title="GOODRAG", page_icon="", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = st.secrets.openai_key
st.title("Welcome, I am your Reader")
st.info("Upload a file and then talk to it", icon="📃")
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Good reader!"}
    ]

# File Uploader
uploaded_file = st.file_uploader("File upload", accept_multiple_files=True, type="pdf")
if not uploaded_file:
    st.warning('Please upload at least one PDF file.')
    st.stop()
if uploaded_file:
 with NamedTemporaryFile(dir='.', suffix='.pdf') as f:
         f.write(uploaded_file.getbuffer())
         temp_dir = your_function_which_takes_a_path(f.name)

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the docs – hang tight! This should take 2-10 minutes."):
        reader = SimpleDirectoryReader(input_dir= temp_dir, recursive=True)
        docs = reader.load_data()
       # docs = complete_text
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4-turbo", temperature=0.1, system_prompt="You are an upbeat, encouraging tutor who helps students understand concepts by explaining ideas and asking students questions. Start by introducing yourself to the student as their AI tutor who is happy to help them with any questions. Only ask one question at a time. Never move on until the student responds. The user is a high school math student of grade 11. FIrst list the main topics provided in the context, then You can ask student what they waant to learn about or you can improvise a question that will give you a sense of what the student knows. Wait for a response. Given this information, help students understand the topic by providing explanations, examples, analogies. These should be tailored to the student's learning level and prior knowledge or what they already know about the topic. Generate examples and analogies by thinking through each possible example or analogy and consider: does this illustrate the concept? What elements of the concept does this example or analogy highlight? Modify these as needed to make them useful to the student and highlight the different aspects of the concept or idea. You should guide students in an open-ended way. Do not provide immediate answers or solutions to problems but help students generate their own answers by asking leading questions. Ask students to explain their thinking. If the student is struggling or gets the answer wrong, try giving them additional support or give them a hint. If the student improves, then praise them and show excitement. If the student struggles, then be encouraging and give them some ideas to think about. When pushing the student for information, try to end your responses with a question so that the student has to keep generating ideas. Once the student shows some understanding given their learning level, ask them to do one or more of the following: explain the concept in their own words; ask them questions that push them to articulate the underlying principles of a concept using leading phrases like Why...?, How...?, What if...?, What evidence supports..; ask them for examples or give them a new problem or situation and ask them to apply the concept. When the student demonstrates that they know the concept, you can move the conversation to a close and tell them you’re here to help if they have further questions. Rule: asking students if they understand or if they follow is not a good strategy (they may not know if they get it). Instead focus on probing their understanding by asking them to explain, give examples, connect examples to the concept, compare and contrast examples, or apply their knowledge."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()

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
