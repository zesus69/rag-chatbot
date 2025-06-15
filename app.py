import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
from main_chat import process_file, process_url, ask_question, get_filetype
import tempfile

# Load OpenAI API key
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"))

# Page configuration
st.set_page_config(page_title="RAG ChatBot", layout="wide")

# Custom CSS for sidebar hover and visual improvements
st.markdown("""
    <style>
    /* Hide sidebar by default and show on hover */
    [data-testid="stSidebar"] {
        transition: all 0.3s ease-in-out;
        width: 0;
        overflow-x: hidden;
        opacity: 0;
        z-index: 9999;
        position: fixed;
        left: 0;
        top: 0;
        bottom: 0;
        background-color: #f8f9fa;
        border-right: 1px solid #ddd;
    }
    [data-testid="stSidebar"]:hover {
        width: 300px !important;
        opacity: 1;
        padding: 1rem;
    }
    .stApp {
        background-color: #ffffff;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 RAG ChatBot with Memory")

# Session state variables
if "history" not in st.session_state:
    st.session_state.history = []
if "input_query" not in st.session_state:
    st.session_state.input_query = ""
if "last_processed_query" not in st.session_state:
    st.session_state.last_processed_query = ""

# Hidden Sidebar (shows on hover)
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
        st.session_state.input_query = ""
        st.session_state.last_processed_query = ""
        st.rerun()

# Display chat history
st.subheader("Chat")
for role, message in st.session_state.history:
    with st.chat_message(role):
        st.markdown(message)

# Input area at bottom of page
with st.container():
    st.markdown("---")
    input_query = st.chat_input("Ask a question regarding the document")

# Process when user submits via Enter
if input_query and input_query.strip():
    input_query = input_query.strip()
    st.session_state.history.append(("user", input_query))

    with st.spinner("Generating..."):
        answer, source = ask_question(input_query, st.session_state.history)

    st.session_state.history.append(("bot", answer))
    
    # Optionally add source
    if source:
        st.session_state.history.append(("bot", f"**Source:**\n{source}"))

    st.experimental_rerun()
