from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = "data/sample.pdf"
print(f"Loading PDF From: {file_path}...")
loader = PyMuPDFLoader(file_path)
docs = loader.load()

print(f"Success! PDF Has {len(docs)} Pages.")
print("-" * 50)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(docs)

print(f"The Document Was Split Into {len(chunks)} smaller chunks.")
print("-" * 50)
print("Preview Of The Very First Chunk:")
print(chunks[0].page_content)

