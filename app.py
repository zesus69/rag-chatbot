import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
from main_chat import process_file, process_url, ask_question, get_filetype
import tempfile

# Load embeddings
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"))

# Page setup
st.set_page_config(page_title="RAG ChatBot", layout="wide")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []
if "show_sidebar" not in st.session_state:
    st.session_state.show_sidebar = False
if "input_query" not in st.session_state:
    st.session_state.input_query = ""

# Custom CSS for layout and toggle
st.markdown("""
    <style>
    .sidebar-container {
        width: 300px;
        background-color: #f8f9fa;
        border-right: 1px solid #ddd;
        padding: 1rem;
        height: 100vh;
        position: fixed;
        top: 0;
        left: 0;
        z-index: 1000;
        overflow-y: auto;
    }
    .chat-main {
        transition: margin-left 0.3s ease;
        padding: 2rem;
    }
    .chat-full {
        margin-left: 0;
    }
    .chat-collapsed {
        margin-left: 300px;
    }
    .toggle-btn {
        position: fixed;
        top: 1rem;
        left: 1rem;
        z-index: 1100;
        background: #ffffff;
        border: 1px solid #ccc;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        cursor: pointer;
        font-weight: 600;
    }
    .chat-container {
        max-height: calc(100vh - 120px);
        overflow-y: auto;
        padding-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Toggle button
if st.button("🧾 Click here to add external information", key="toggle_btn", help="Toggle upload/info panel"):
    st.session_state.show_sidebar = not st.session_state.show_sidebar

# Sidebar Panel
if st.session_state.show_sidebar:
    with st.container():
        st.markdown('<div class="sidebar-container">', unsafe_allow_html=True)
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

        st.markdown('</div>', unsafe_allow_html=True)

# Main Chat Area
main_class = "chat-main chat-collapsed" if st.session_state.show_sidebar else "chat-main chat-full"
st.markdown(f'<div class="{main_class}">', unsafe_allow_html=True)

st.title("🧠 RAG ChatBot with Memory")

with st.container():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for role, message in st.session_state.history:
        with st.chat_message(role):
            st.markdown(message)
    st.markdown('</div>', unsafe_allow_html=True)

# Input Box at bottom
user_input = st.chat_input("Ask a question...")

# Handle user query
if user_input and user_input.strip():
    user_input = user_input.strip()
    st.session_state.history.append(("user", user_input))

    with st.spinner("Generating..."):
        answer, source = ask_question(user_input, st.session_state.history)

    st.session_state.history.append(("bot", answer))
    if source:
        st.session_state.history.append(("bot", f"**Source:**\n{source}"))

    st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
