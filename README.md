# Healthcare Information Chatbot Using RAG

A healthcare information chatbot built using Retrieval-Augmented Generation (RAG). The system retrieves relevant information from a healthcare dataset using semantic search and FAISS, then provides the retrieved context to a Large Language Model (LLM) to generate a response.

This project is intended for educational purposes and is not a replacement for professional medical advice.

---

## Overview

Traditional LLMs rely primarily on knowledge learned during training. This project uses RAG to provide the LLM with relevant information from a dedicated healthcare knowledge base.

### Workflow

```text
User Question
      |
      v
Question Embedding
      |
      v
FAISS Similarity Search
      |
      v
Relevant Healthcare Information
      |
      v
LLM + Retrieved Context
      |
      v
Generated Response
```

This allows the chatbot to answer questions using information retrieved from the project's dataset rather than relying entirely on the LLM's internal knowledge.

---

## Features

* Retrieval-Augmented Generation (RAG)
* Semantic search using embeddings
* Local FAISS vector database
* Hugging Face embedding model
* Healthcare and medication-related dataset
* CPU-based execution
* FastAPI backend
* Local application interface
* Retrieval testing and verification
* Basic healthcare safety instructions

---

## Tech Stack

| Component       | Technology                  |
| --------------- | --------------------------- |
| Language        | Python                      |
| Backend         | FastAPI                     |
| Vector Database | FAISS                       |
| RAG Framework   | LangChain                   |
| Embeddings      | BAAI/bge-small-en-v1.5      |
| LLM             | Hugging Face-compatible LLM |
| Data Processing | Pandas                      |
| Interface       | Streamlit                   |

---

## Project Structure

```text
healthcare-rag-chatbot/
│
├── app.py
├── main.py
├── build_faiss_index.py
├── test_faiss_retrieval.py
├── verify.py
├── check.py
├── commands.txt
├── Guaranteed_10Q_Expanded_Dataset.csv
├── vector_db/
├── data/
├── .gitignore
├── requirements.txt
└── README.md
```

### Important Files

**`build_faiss_index.py`**
Processes the dataset, generates embeddings, and creates the FAISS index.

**`test_faiss_retrieval.py`**
Tests whether relevant information can be retrieved from FAISS.

**`main.py`**
Contains the backend/API logic.

**`app.py`**
Contains the application interface.

---

## Embedding Model

The project uses:

```text
BAAI/bge-small-en-v1.5
```

The model converts healthcare text and user questions into numerical vectors. FAISS then compares these vectors to find semantically similar information.

The embedding model is configured to run on CPU.

---

## Dataset

The project uses a healthcare and medication-related dataset, including:

```text
Guaranteed_10Q_Expanded_Dataset.csv
```

The data is processed before being converted into embeddings.

```text
Dataset
   |
   v
Data Processing
   |
   v
Embeddings
   |
   v
FAISS Index
```

When the dataset is updated, the FAISS index can be rebuilt without retraining the LLM.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-rag-chatbot.git
cd healthcare-rag-chatbot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Build the Vector Database

Run:

```bash
python build_faiss_index.py
```

This processes the dataset, generates embeddings, and creates the FAISS vector index.

---

## Test Retrieval

Run:

```bash
python test_faiss_retrieval.py
```

This checks whether the system can retrieve relevant information from the vector database.

---

## Run the Application

For the FastAPI backend:

```bash
uvicorn main:app --reload
```

For the Streamlit interface:

```bash
streamlit run app.py
```

---

## LLM

The project is designed to work with a Hugging Face-compatible LLM.

The LLM receives:

```text
User Question
+
Retrieved Healthcare Context
+
System Instructions
```

The system instructions are designed to reduce unsupported answers and prevent the chatbot from presenting itself as a medical professional.

---

## Healthcare Safety

The chatbot is intended to provide general healthcare information.

It should not be used to:

* Diagnose medical conditions
* Prescribe medication
* Recommend personalized treatment
* Provide unsupported dosage instructions
* Replace a doctor or other qualified healthcare professional

For medical emergencies or serious health concerns, users should contact an appropriate healthcare professional or emergency service.

---

## Hardware

The project is designed to support local CPU-based development.

Example development hardware:

```text
CPU: Intel Core i5-12500H
RAM: 16 GB
GPU: Integrated Graphics
```

The relatively lightweight embedding model makes the retrieval component suitable for CPU execution. Larger LLMs may require more powerful hardware or a cloud-based inference service.

---

## GitHub

Initialize Git:

```bash
git init
```

Add files:

```bash
git add .
```

Commit:

```bash
git commit -m "Initial Healthcare RAG project"
```

Connect the GitHub repository:

```bash
git remote add origin https://github.com/YOUR_USERNAME/healthcare-rag-chatbot.git
```

Push:

```bash
git branch -M main
git push -u origin main
```

Do not commit `.env` files, API keys, virtual environments, or other private credentials.

---

## Future Improvements

* Better document chunking
* Improved retrieval accuracy
* Reranking
* Source citations
* Conversation history
* Better hallucination detection
* RAG evaluation
* Model quantization
* Docker deployment
* Cloud deployment

---

## Disclaimer

This project is developed for educational and demonstration purposes. It is not a clinically validated medical system and should not be used as a substitute for professional medical diagnosis, treatment, or advice.
