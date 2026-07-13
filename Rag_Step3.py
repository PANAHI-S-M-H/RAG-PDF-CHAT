import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")

print("Loading Local Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("Connecting To Local Vector Database...")
vector_db_dir = "data/chroma_db"
vectorstore = Chroma(persist_directory=vector_db_dir, embedding_function=embeddings)


retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

template = """
You Are an expert technical assistant. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say "I cannot find the answer in the provided document." DO NOT make up an answer.

Context:
{context}

Question: {question}

Answer in a clear and concise way:
"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("-" * 50)
print("PDF Assistant is Ready!")

question = "What is the TX current at 0dBm output power?"
print(f"Question: {question}")
print("Searching document...")


print("\n" + "="*20 + " DEBUG INFO " + "="*20)
found_docs = retriever.invoke(question)
for i, doc in enumerate(found_docs):
    print(f"--- Chunk {i+1} ---")
    
    print(doc.page_content.strip()[:150] + "...\n")
print("="*52 + "\n")


print("Generating answer...")
try:
    response = rag_chain.invoke(question)
    print(f"Answer: {response}")
except Exception as e:
    print(f"Error: {e}")
