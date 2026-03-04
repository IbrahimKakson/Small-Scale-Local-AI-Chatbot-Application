#  Local RAG Chatbot

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Mistral](https://img.shields.io/badge/Mistral_7B-black?style=for-the-badge)

A powerful, privacy-first Retrieval-Augmented Generation (RAG) chatbot that runs entirely on your local machine. Chat with your PDF documents without your data ever leaving your system.

## 🌟 Key Features

- **100% Local Execution**: Your data stays private. No API keys or cloud services required.
- **Quantized Performance**: Optimized for consumer hardware using `Mistral-7B-Instruct-v0.2` via LlamaCpp.
- **Document Ingestion**: Easily process all PDF files in your `data/` directory.
- **Persistent Vector Store**: Uses ChromaDB to index and store document embeddings for lightning-fast retrieval.
- **Sleek UI**: Built with Streamlit for a clean, intuitive chat experience.

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **Orchestration**: LangChain
- **LLM**: Mistral 7B (GGUF Quantized)
- **Vector Database**: ChromaDB
- **Embeddings**: HuggingFace (`all-MiniLM-L6-v2`)

## 🚀 Getting Started

### 1. Prerequisite Environments
Ensure you have Python 3.9+ installed. It is recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/your-username/LocalAIChatbot.git
cd LocalAIChatbot
pip install -r requirements.txt
```

### 3. Model Setup
Download the model and place it in the `models/` directory:
- **Model**: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`
- **Path**: `LocalAIChatbot/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf`

### 4. Prepare Your Data
Place your PDF files in the `data/` folder.

## 📖 Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```
2. Open your browser to the URL provided (usually `http://localhost:8501`).
3. Click "Process Documents" in the sidebar to index your PDFs.
4. Start chatting with your local AI!

## 🔒 Privacy & Security

This project is built with a focus on absolute privacy. 
- **No Cloud API calls**: All processing happens on your CPU/GPU.
- **Local Vectors**: Your document embeddings are stored locally in the `chroma_db/` folder.
- **Local Logs**: No chat history is sent to external servers.

---
