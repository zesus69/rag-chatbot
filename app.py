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

# Custom CSS
st.markdown("""
    <style>
    /* Sidebar hover effect */
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

    /* Fix input form at bottom */
    .chat-input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 10px 20px;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 9999;
    }

    /* Make chat container scrollable */
    .chat-container {
        padding-bottom: 100px;  /* leave space for input */
    }

    textarea {
        font-size: 16px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 RAG ChatBot with Memory")

# Session state
if "history" not in st.session_state:
    st.session_state.history = []
if "input_query" not in st.session_state:
    st.session_state.input_query = ""

# Sidebar input
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
        st.rerun()

# Display chat history
with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for role, message in st.session_state.history:
        with st.chat_message(role):
            st.markdown(message)
    st.markdown('</div>', unsafe_allow_html=True)

# Input form fixed at bottom
st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
with st.form("chat_input_form", clear_on_submit=True):
    user_input = st.text_area("Ask a question regarding the document", key="input_box", label_visibility="collapsed", height=60)
    submitted = st.form_submit_button("Send")

    if submitted and user_input.strip():
        user_input = user_input.strip()
        st.session_state.history.append(("user", user_input))

        with st.spinner("Generating..."):
            answer, source = ask_question(user_input, st.session_state.history)

        st.session_state.history.append(("bot", answer))
        if source:
            st.session_state.history.append(("bot", f"**Source:**\n{source}"))

        st.experimental_rerun()
st.markdown('</div>', unsafe_allow_html=True)
