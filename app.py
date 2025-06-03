import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
embeddings = OpenAIEmbeddings(
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"))
from main_chat import process_file, process_url, ask_question, get_filetype
import tempfile

st.set_page_config(page_title="RAG ChatBot", layout="wide")
st.title("RAG ChatBot with Memory")

if "history" not in st.session_state:
    st.session_state.history = []

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
st.subheader("Chat")

query = st.text_input("Ask a question regarding the document")  

if query:
    st.session_state.history.append(("user", query))

    with st.spinner("Generating..."):
        answer, source = ask_question(query, st.session_state.history)

    st.session_state.history.append(("bot", answer))

for role, message in st.session_state.history:
    with st.chat_message(role):  
        st.markdown(message)

    if role == "bot" and source:  
        st.markdown("**Source:**")
        st.markdown(source)