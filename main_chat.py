import os
from langchain.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from typing import List, Tuple

# Global vectorstore and retriever
doc_chunks: List[Document] = []
vectorstore = None

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# api
api_deployment = os.getenv("AZURE_DEPLOYMENT_NAME")
api_key = os.getenv("AZURE_API_KEY")
api_endpoint = os.getenv("AZURE_ENDPOINT")
api_version = os.getenv("AZURE_API_VERSION")

# embeddings
embed_deployment = os.getenv("AZURE_DEPLOYMENT_NAME2")
embed_version = os.getenv("AZURE_API_VERSION2")

# Create embeddings object
embeddings = AzureOpenAIEmbeddings(azure_deployment=embed_deployment,
azure_endpoint=api_endpoint,chunk_size=2048)

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

# Prompt Template for better output control
PROMPT_TEMPLATE = """
You are an assistant with access to external documents. Answer the user's question using the provided context. 
Be precise and mention the source if relevant. If unsure, say "I don't know."

Context:
{context}

Question:
{question}

Answer:
"""

QA_CHAIN_PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])


def get_filetype(filename: str) -> str:
    ext = os.path.splitext(filename)[-1].lower()
    if ext == ".pdf":
        return "pdf"
    elif ext == ".txt":
        return "txt"
    else:
        raise ValueError("Unsupported file type")


def load_documents_from_file(filepath: str, filetype: str) -> List[Document]:
    if filetype == "pdf":
        loader = PyPDFLoader(filepath)
    elif filetype == "txt":
        loader = TextLoader(filepath)
    else:
        raise ValueError("Unsupported file type")
    return loader.load()


def process_file(filepath: str, filetype: str):
    global vectorstore, doc_chunks
    docs = load_documents_from_file(filepath, filetype)
    chunks = splitter.split_documents(docs)
    doc_chunks.extend(chunks)
    update_vectorstore()


def process_url(url: str):
    global vectorstore, doc_chunks
    loader = WebBaseLoader(url,verify_ssl=False)
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    doc_chunks.extend(chunks)
    update_vectorstore()


def update_vectorstore():
    global vectorstore
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)


def ask_question(query: str, history: List[Tuple[str, str]]) -> Tuple[str, str]:
    global vectorstore

    llm = AzureChatOpenAI(
        azure_deployment=api_deployment,
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=api_endpoint,
        temperature=0.2
    )

    # If we have a vectorstore, do Retrieval-Augmented Generation (RAG)
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        result = qa(query)
        answer = result["result"]
        source_docs = result.get("source_documents", [])[:1]
        sources = "\n".join([f"- {doc.metadata.get('source', 'Unknown')}" for doc in source_docs])
        return answer.strip(), sources.strip()

    # If no external data, just use the LLM as a normal chatbot
    else:
        response = llm.predict(query)
        return response.strip(), ""  # No sources if not using external docs
    
def clear_all():
    global doc_chunks, vectorstore
    doc_chunks = []
    vectorstore = None