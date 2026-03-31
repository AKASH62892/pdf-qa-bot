import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os

CHROMA_PATH = "chroma_db"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def build_vector_store(pdf_path, session_id):
    text = extract_text_from_pdf(pdf_path)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    db_path = f"{CHROMA_PATH}/{session_id}"
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    
    return len(chunks)

def get_relevant_chunks(query, session_id, k=5):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    db_path = f"{CHROMA_PATH}/{session_id}"
    vectorstore = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])