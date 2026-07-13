import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

file_path = "data/sample.pdf"
loader = PyMuPDFLoader(file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

print(f"Loaded and split PDF into {len(chunks)} chunks.")


print("Initializing LOCAL Embedding Model (HuggingFace)...")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Saving chunks to Vector Database (Please wait...)")
vector_db_dir = "data/chroma_db"

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=vector_db_dir
)

print("-" * 50)
print(f"Success! Vector database created and saved securely at: {vector_db_dir}")
