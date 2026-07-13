import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Dynamic AI Assistant", page_icon="🚀", layout="wide")
st.title("Dynamic PDF AI Assistant")
st.caption("Upload, Process, And Chat With Any Document You Like!")
st.divider()


load_dotenv()
Vector_db_dir = "faiss_index"

with st.sidebar:
    st.header("Document Upload")
    st.info("Upload A PDF File For The AI ​​To Read.")

    uploaded_file = st.file_uploader("Upload Your PDF Here", type="pdf")
    process_button = st.button("Process Document", use_container_width=True)
    st.divider()
    st.markdown("Developed By SMHP")


def process_pdf(file):

    file_path = os.path.join("data", "temp_upload.pdf")
    with open(file_path, "wb") as f:
        f.write(file.getbuffer())

    loader = PyMuPDFLoader(file_path)
    docs = loader.load
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)


    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    FAISS.from_documents(documents=chunks, embedding=embeddings, persist_directory=Vector_db_dir)
    return True


if process_button and uploaded_file is not None:
    with st.spinner("Reading, Splitting, And Saving To Database..."):
        process_pdf(uploaded_file)
        st.session_state.messages = []
        st.sidebar.success("Document Processed Successfully! Ready To Chat.")
elif process_button and uploaded_file is None:
    st.sidebar.warning("Please Upload A PDF File First.")


def get_rag_chain():
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS(persist_directory=Vector_db_dir, embedding_function=embeddings)
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
    
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask A Question About Your Uploaded Document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking And Searching..."):
            try:
                rag_chain = get_rag_chain()
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")