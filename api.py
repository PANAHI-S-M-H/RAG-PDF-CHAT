from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="Applied AI Portfolio API",
              description="My First AI-Powered RAG API For PDF Q&A",
              version="1.0"
              )

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

template = """
You Are An Expert Technical Assistant. Use The Following Pieces Of Retrieved Context To Answer The Question.
If You Don't Know The Answer, Just Say "I Cannot Find The Answer In The Provided Document." DO NOT Make Up An Answer.

Context:
{context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n" .join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

class UserRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_pdf(request: UserRequest):
    try:

        answer = rag_chain.invoke(request.question)


        return {
            "status": "success",
            "question": request.question,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    