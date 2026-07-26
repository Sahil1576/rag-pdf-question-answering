import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Q&A", page_icon="📄", layout="centered")

st.markdown("""
<style>
.st-key-top_section {
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: #0e1117;
    padding-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

PERSIST_DIR = "chroma_store"

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings()

embedding_model = get_embedding_model()

if "vectorstore" not in st.session_state:
    if os.path.exists(PERSIST_DIR):
        st.session_state.vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embedding_model,
        )
    else:
        st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
You are a Retrieval-Augmented Generation (RAG) assistant.

Your ONLY source of information is the retrieved context provided below.

Rules:
1. Answer ONLY using the retrieved context.
2. Do NOT use your own knowledge, memory, or pre-trained information.
3. Do NOT make assumptions or infer facts that are not explicitly mentioned in the context.
4. If the answer cannot be found completely in the retrieved context, reply EXACTLY:
I couldn't find the answer in the provided document.
5. If the user's question is unrelated to the uploaded document, reply EXACTLY:
This question is outside the scope of the uploaded document.
6. Never answer greetings, casual conversation, general knowledge questions, or any question that is not supported by the retrieved context.

------------------------
Conversation History:
{chat_history}

------------------------
Retrieved Context:
{context}
------------------------
"""),
    ("human", "Question:\n{question}\n\nAnswer:")
])

with st.container(key="top_section"):
    st.title("📄 RAG Q&A")
    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

    file_key = f"{pdf_file.name}_{pdf_file.size}" if pdf_file is not None else None

    if pdf_file is not None and file_key not in st.session_state.processed_files:
        with st.spinner(f"Reading and indexing {pdf_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.getvalue())
                tmp_path = tmp.name

            docs = PyPDFLoader(tmp_path).load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embedding_model,
                    persist_directory=PERSIST_DIR,
                )
            else:
                st.session_state.vectorstore.add_documents(chunks)

            st.session_state.processed_files.add(file_key)
            os.unlink(tmp_path)

        st.success(f"Added {pdf_file.name} ({len(chunks)} chunks)")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.display_history = []
        st.rerun()

for msg in st.session_state.display_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the document...")

if question:
    if st.session_state.vectorstore is None:
        st.warning("Please upload a PDF first.")
    else:
        st.session_state.display_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                )
                context_docs = retriever.invoke(question)
                context = "\n\n".join(doc.page_content for doc in context_docs)

                llm = ChatMistralAI(model="mistral-small-2506")

                final_prompt = PROMPT.invoke({
                    "chat_history": st.session_state.chat_history,
                    "context": context,
                    "question": question,
                })
                response = llm.invoke(final_prompt)

            st.markdown(response.content)

        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=response.content))
        st.session_state.display_history.append({"role": "assistant", "content": response.content})