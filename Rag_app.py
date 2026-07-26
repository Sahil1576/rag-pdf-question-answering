from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import os

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

embedding_model = HuggingFaceEmbeddings()

persist_directory = "ChromaDB"

print("="*80)
print("Welcome to The Rag Application")
print("="*80)

if os.path.exists(persist_directory):
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )

    print("Existing Vectorstore Loaded..")
    print("-"*80)

    if vectorstore._collection.count == 0:
        print("Vectorstore is Empty. Please Upload a document.")
        print("-"*80)

        upload_pdf = input("Upload PDF Path : ")
        print("-"*80)
        
        data = PyPDFLoader(upload_pdf)
        
        docs = data.load()
        
        chunks = splitter.split_documents(docs)
        
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )

        print("Documents Loaded Successfully.")
        print("-"*80)

else:
    upload_pdf = input("Upload PDF Path : ")
    print("-"*80)

    data = PyPDFLoader(upload_pdf)

    docs = data.load()

    chunks = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    print("Documents Loaded Successfully.")
    print("-"*80)

llm = ChatMistralAI(model="mistral-small-2506")

prompts = ChatPromptTemplate.from_messages([
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
7. Never generate information that is not present in the retrieved context.
8. Treat the retrieved context as the single source of truth.

------------------------
Conversation History:
{chat_history}

------------------------
Retrieved Context:
{context}

------------------------
"""),

("human", """
Question:
{question}

Answer:
""")
])

chat_history = []

while True:
    question = input("Ask : ")
    print("-"*80)

    if question.lower() == "exit":
        print("Closed Application..")
        break

    chat_history.append(HumanMessage(content=question))

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":4,
            "fetch_k":10,
            "lambda_mult":0.5
        }
    )

    context = retriever.invoke(question)

    final_prompt = prompts.invoke({
        "chat_history":chat_history,
        "context":context,
        "question":question
    })

    responce = llm.invoke(final_prompt)

    chat_history.append(AIMessage(content=responce.content))

    print("AI : ", responce.content)
    print("="*80)