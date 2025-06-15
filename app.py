import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
from main_chat import process_file, process_url, ask_question, get_filetype
import tempfile

# Load OpenAI embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"))

# Page configuration
st.set_page_config(page_title="RAG ChatBot", layout="wide")

# Apply custom CSS for color scheme and layout
st.markdown("""
    <style>
    body {
        background-color: #f4f4f4;
    }
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    .stChatMessage {
        background-color: #e0f7fa;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
    }
    .user-message {
        background-color: #d1c4e9;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
    }
    .bot-message {
        background-color: #c8e6c9;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 RAG ChatBot with Memory")

# Session state for conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar for external info input
with st.sidebar:
    st.header("Add External Information")
    url = st.text_input("Enter Website URL")
    uploaded_file = st.file_uploader("Upload PDF or TXT file")

    if st.button("Add Information"):
        if url:
            with st.spinner("Processing..."):
                process_url(url)
            st.success("URL has been successfully loaded and added")
        elif uploaded_file:
            file_type = get_filetype(uploaded_file.name)
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:  
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            with st.spinner("Processing..."):
                process_file(temp_file_path, file_type)
            st.success(f"{uploaded_file.name} has been successfully loaded and added")
        else:
            st.warning("Please enter a valid URL or upload a file")

    if st.button("Clear chat history"):
        st.session_state.history = []  
        st.rerun()

# Display chat messages (top to bottom)
st.subheader("Chat")
for role, message in st.session_state.history:
    with st.chat_message(role):
        st.markdown(message)

# Text input at the bottom
with st.container():
    if "input_query" not in st.session_state:
        st.session_state.input_query = ""

    query = st.text_input("Ask a question regarding the document", key="input_query", label_visibility="collapsed")

    # Only run if query is non-empty and hasn't been processed yet
    if query and st.session_state.get("last_processed_query") != query:
        st.session_state.history.append(("user", query))

        with st.spinner("Generating..."):
            answer, source = ask_question(query, st.session_state.history)

        st.session_state.history.append(("bot", answer))
        st.session_state.last_processed_query = query  # Mark it as processed

        # Clear input
        st.session_state.input_query = ""
        st._rerun()

