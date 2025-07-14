import streamlit as st
from langchain_openai import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
from main_chat import process_file, process_url, ask_question, get_filetype, clear_all, is_valid_url
import tempfile
from pydantic import SecretStr
 
embeddings = AzureOpenAIEmbeddings(
    api_key=SecretStr(os.environ.get("AZURE_API_KEY","")),
    model = os.environ.get("AZURE_DEPLOYMENT_NAME2"),
    azure_endpoint = os.environ.get("AZURE_ENDPOINT"),
    chunk_size = 2048,              
)
 
st.set_page_config(page_title="RAGBot", layout="wide")
 
st.markdown("""
    <style>
    html, body, [data-testid="stApp"] {
        height: 100%;
        margin: 0;
        padding: 0;
    }
 
    .chat-container {
        max-height: calc(100vh - 120px);
        overflow-y: auto;
        padding-bottom: 1rem;
    }
 
    [data-testid="stSidebar"] {
        transition: all 0.2s ease-in-out;
        width: 0;
        overflow-x: hidden;
        opacity: 0;
        z-index: 9999;
        position: fixed;
        left: 0;
        top: 0;
        bottom: 0;
        background-color: #f8f9fa;
       
    }
 
    [data-testid="stSidebar"] {
        width: 300px !important;
        opacity: 1;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)
 
st.title("RAGBot")
 
if "history" not in st.session_state:
    st.session_state.history = []
if "input_query" not in st.session_state:
    st.session_state.input_query = ""
if "input_url" not in st.session_state:
    st.session_state.clear_triggered = False
 
if st.session_state.clear_triggered:
    st.session_state.history = []
    st.session_state.input_query = ""
    st.session_state.input_url = ""
    st.session_state.clear_triggered = False
 
with st.sidebar:
    st.header("Add External Information")
    url = st.text_input("Enter Website URL",key="input_url")
    uploaded_file = st.file_uploader("Upload PDF or TXT file")
 
    if st.button("Add Information"):
        if url:
            with st.spinner("Processing..."):
                if not is_valid_url(url):
                    st.error("Please enter a valid URL")
                else:
                    process_url(url)
                    st.success("URL has been successfully loaded and added")
        elif uploaded_file:
            file_type = get_filetype(uploaded_file.name)
            file_bytes = uploaded_file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}", mode="wb") as temp_file:
                temp_file.write(file_bytes)
                temp_file_path = temp_file.name
            with st.spinner("Processing..."):
                process_file(temp_file_path, file_type)
            st.success(f"{uploaded_file.name} has been successfully loaded and added")
        else:
            st.warning("Please enter a valid URL or upload a file")
 
    if st.button("Clear chat history"):
        st.session_state.history = []
        st.session_state.input_query = ""
        st.session_state.clear_triggered = True
        clear_all()
        st.rerun()
 
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for role, message in st.session_state.history:
        with st.chat_message(role):
            st.markdown(message)
    st.markdown('</div>', unsafe_allow_html=True)
 
user_input = st.chat_input("Ask a question...")
 
if user_input and user_input.strip():
    user_input = user_input.strip()
    st.session_state.history.append(("user", user_input))
 
    with st.spinner("Generating..."):
        answer, source = ask_question(user_input, st.session_state.history)
 
    st.session_state.history.append(("bot", answer))
 
    if source:
        st.session_state.history.append(("bot", f"**Source:**\n{source}"))
 
    st.rerun()