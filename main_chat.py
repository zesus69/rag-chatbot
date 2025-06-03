import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader
from langchain_openai import AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

llm = AzureChatOpenAI(
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key="e11b852a42dc4318ae635d91234bd972",
    api_version="2023-05-15",
    azure_endpoint="https://ai-aistudio-ailearn.cognitiveservices.azure.com/",
    chunk_size=2048
)

vectorstore = None
index_path = "faiss_local"
if os.path.exists(index_path):
    vectorstore = FAISS.load_local(index_path,embeddings,allow_dangerous_deserialization=True)

qa_chain = None
def init_qachain():
    global qa_chain
    if vectorstore is not None:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )
    else:
        print("Error: Vectorstore is not initialized. Cannot create QA chain.")

if vectorstore:
    init_qachain()

def add_doc(texts):
    global vectorstore
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = []
    for text in texts:
        chunks = text_splitter.split_text(text)
        docs.extend([Document(page_content = chunk) for chunk in chunks])
    if vectorstore is None:
        vectorstore = FAISS.from_documents(docs,embedding = embeddings)
    else:
        vectorstore.add_documents(docs)
    vectorstore.save_local("faiss_index")
    init_qachain()

def ask_question(query, chat_history):
    if qa_chain is None:
        return "No documents have been added. Please upload a document or provide a URL.", None

    result = qa_chain({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    sources = result.get("source_documents", [])
    
    source_name = None
    if sources:
        metadata = sources[0].metadata  # Get metadata of the first source
        if "file_path" in metadata:  # For files
            source_name = os.path.basename(metadata["file_path"])
        elif "url" in metadata:  # For URLs
            source_name = metadata["url"]
    
    return answer, source_name

def process_file(file, file_type):
    if file_type == "pdf":
        loader = PyPDFLoader(file_path=file)
    else:
        loader = TextLoader(file_path=file)
    docs = loader.load()
    try:
        texts = [doc.page_content for doc in docs]
    except AttributeError as e:
        print(f"{e}")
    add_doc(texts)

def process_url(url):
    loader = WebBaseLoader(url)
    loader.requests_kwargs = {'verify':False}
    docs = loader.load()
    try:    
        texts = [doc.page_content for doc in docs]
    except AttributeError as e:
        print(f"{e}")
    add_doc(texts)

def get_filetype(filename):
    if filename.endswith(".pdf"):
        return "pdf"
    else:
        return "txt"

if os.path.exists(index_path):
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
else:
    # Initialize an empty vectorstore with a placeholder document
    placeholder_text = "No Documents have been added. Please upload a document or provide a URL."
    vectorstore = FAISS.from_texts([placeholder_text], embedding=embeddings)
    try:
        vectorstore.delete([placeholder_text])  # Remove the placeholder after initialization
    except Exception as e:
        print(f"Error removing placeholder: {e}")

if vectorstore:
    init_qachain()

