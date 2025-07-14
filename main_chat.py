import os
import re
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from pydantic import SecretStr
from typing import List, Tuple
 
doc_chunks: List[Document] = []
vectorstore = None
 
from dotenv import load_dotenv
load_dotenv()
 
client = AzureChatOpenAI(
        api_key= os.environ.get("AZURE_API_KEY",""),
        api_version= os.environ.get("AZURE_API_VERSION",""),
        azure_endpoint= os.environ.get("AZURE_ENDPOINT",""),
        temperature=0.1,
        model= os.environ.get("AZURE_DEPLOYMENT_NAME",""),
        #streaming=True,
)
                       
embeddings = AzureOpenAIEmbeddings(
    api_key= SecretStr(os.environ.get("AZURE_API_KEY","")),
    model = os.environ.get("AZURE_DEPLOYMENT_NAME2"),
    azure_endpoint = os.environ.get("AZURE_ENDPOINT"),
    chunk_size = 2048,          
)
 
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
 
PROMPT_TEMPLATE = """
You are an  AI assistant called RagBot with access to external documents. Answer the user's question using the provided context.
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
    loader = WebBaseLoader(url, verify_ssl=False)
    docs = loader.load()
    chunks = splitter.split_documents(docs)
    doc_chunks.extend(chunks)
    update_vectorstore()
 
 
def update_vectorstore():
    global vectorstore
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
 
 
def ask_question(query: str, history: List[Tuple[str, str]]) -> Tuple[str, str]:
    global vectorstore
 
    if vectorstore:
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        qa = RetrievalQA.from_chain_type(
            llm=client,
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
 
    else:
        response = client.predict(query)
        return response.strip(), ""
   
def clear_all():
    global doc_chunks, vectorstore
    doc_chunks = []
    vectorstore = None
   
def is_valid_url(url: str) -> bool:
    url_regex = re.compile(
        r'^(https?://)?'
        r'([a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}'  
        r'(:\d+)?'  
        r'(\/\S*)?$',
        re.IGNORECASE
    )
    return re.match(url_regex, url.strip()) is not None