# 📄 RAG PDF Question Answering System

## 🎥 Project Demo

A complete walkthrough of the application is available on LinkedIn.

👉 **Watch Demo:**  
https://www.linkedin.com/posts/YOUR-LINK-HERE

A Retrieval-Augmented Generation (RAG) application built with **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, and **Mistral AI**. The application allows users to upload PDF documents, stores them in a persistent vector database, and answers questions strictly from the uploaded documents.

---

## 🚀 Features

- 📄 Upload PDF documents
- ✂️ Automatic document chunking
- 🔎 Semantic search using Chroma Vector Database
- 💾 Persistent ChromaDB (No need to re-index documents every time)
- 🧠 HuggingFace Embeddings
- 🤖 Mistral AI for answer generation
- 💬 Conversation history support
- 📚 Retrieval-Augmented Generation (RAG)
- 🚫 Prevents hallucinations by answering only from retrieved context
- 🎯 MMR (Max Marginal Relevance) retrieval for diverse and relevant chunks

---

## 🛠️ Tech Stack

- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Mistral AI
- PyPDF
- Python Dotenv

---

## 📂 Project Structure

```
.
├── Rag_app.py
├── .env
├── requirements.txt
├── ChromaDB/
└── README.md
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/rag-pdf-qa.git

cd rag-pdf-qa
```

### Create Virtual Environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a **.env** file.

```env
MISTRAL_API_KEY=YOUR_API_KEY
```

---

## ▶️ Run Application

```bash
python Rag_app.py
```

---

## 📖 How It Works

### First Run

1. Checks whether a persistent ChromaDB exists.
2. If not found:
   - Loads the PDF.
   - Splits it into chunks.
   - Generates embeddings.
   - Stores embeddings inside ChromaDB.
3. Creates a retriever.

### Next Runs

1. Loads the existing ChromaDB.
2. Skips embedding generation.
3. Retrieves the most relevant chunks directly.

---

## 🔄 Retrieval Pipeline

```
User Question
      │
      ▼
Query Embedding
      │
      ▼
ChromaDB
      │
Cosine Similarity + MMR
      │
Top Relevant Chunks
      │
      ▼
Mistral AI
      │
      ▼
Final Answer
```

---

## 📌 Prompt Rules

The assistant follows strict Retrieval-Augmented Generation principles:

- Uses only retrieved document context
- Never relies on model knowledge
- Does not hallucinate
- Rejects unrelated questions
- Returns a predefined response if the answer is unavailable

---

## 📦 Dependencies

Major libraries used:

- LangChain
- ChromaDB
- HuggingFace Embeddings
- Mistral AI
- PyPDF
- Python Dotenv

---

## 🎯 Future Improvements

- Support multiple PDF documents
- Source citation with page numbers
- Streamlit UI
- Document deletion
- Hybrid Search (BM25 + Vector Search)
- Metadata filtering
- Multi-file conversations

---

## 👨‍💻 Author

**Sahil Katve**

If you found this project useful, consider giving it a ⭐ on GitHub.
