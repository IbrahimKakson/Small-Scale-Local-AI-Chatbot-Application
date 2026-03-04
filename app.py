import os
import streamlit as st
# Document Loading: Loads PDFs from a directory
from langchain_community.document_loaders import PyPDFLoader
# Vector Store: Stores the embeddings locally for retrieval
from langchain_community.vectorstores import Chroma
# LLM: The local Large Language Model (Mistral via LlamaCpp)
from langchain_community.llms import LlamaCpp
# Embeddings: Converts text into vector numbers
from langchain_community.embeddings import HuggingFaceEmbeddings
# Text Splitting: Chunks long documents into smaller pieces
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Chains: Orchestrates the Retrieval-Augmented Generation process
from langchain.chains import RetrievalQA


# --- Configuration Constants ---
# Path to the quantized model file
MODEL_PATH = "models/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
# Directory containing PDF documents to ingest
DATA_PATH = "data/"
# Directory to persist the Chroma vector database
DB_PATH = "chroma_db/"

class LocalRAG:
    """
    Main class for the Local RAG System.
    Handles LLM initialization, document ingestion, and question answering.
    """
    def __init__(self):
        self.llm = None
        self.vector_store = None
        self.qa_chain = None
        
        # --- Initialize LLM ---
        # Checks if the model file exists before trying to load it
        if os.path.exists(MODEL_PATH):
            try:
                # Load the GGUF model using LlamaCpp
                # n_ctx: Context window size (how much text it can remember at once)
                # n_gpu_layers: Set to 0 for CPU, -1 for full GPU offload
                self.llm = LlamaCpp(
                    model_path=MODEL_PATH,
                    n_ctx=2048,
                    temperature=0.1, # Low temperature for more factual, less creative answers
                    verbose=False
                )
            except Exception as e:
                st.error(f"Failed to load model: {e}")
        else:
            st.error(f"Model not found at {MODEL_PATH}. Please upload the .gguf model to the models/ directory.")

    def ingest(self):
        """
        Ingestion Pipeline:
        1. Reads PDF files from DATA_PATH
        2. Splits text into chunks
        3. Creates embeddings
        4. Stores vectors in ChromaDB
        """
        # Ensure data directory exists
        if not os.path.exists(DATA_PATH):
            st.error(f"Data directory {DATA_PATH} not found.")
            return

        documents = []
        # List all PDF files in the directory
        pdf_files = [f for f in os.listdir(DATA_PATH) if f.endswith('.pdf')]
        
        if not pdf_files:
            st.warning("No PDF files found in data directory.")
            return

        status_text = st.empty()
        status_text.text("Loading documents...")
        
        # Load each PDF
        for file in pdf_files:
            file_path = os.path.join(DATA_PATH, file)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            
        status_text.text("Splitting text...")
        # Split text to ensuring context is kept with overlap
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        
        status_text.text("Creating embeddings and vector store...")
        # Generate embeddings using a lightweight local model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Create and persist the vector database
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=DB_PATH
        )
        status_text.success("Ingestion complete! Vector store created.")

    def query(self, question):
        """
        Retrieval and Generation:
        1. Finds relevant document chunks for the question
        2. Sends context + question to the LLM
        3. Returns the answer
        """
        if not self.vector_store:
            # Try to load existing vector store from disk if not already in memory
            if os.path.exists(DB_PATH):
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                self.vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
            else:
                st.error("Vector store not found. Please process documents first.")
                return "Error: Vector store not initialized."
        
        if not self.llm:
             st.error("LLM not initialized. Cannot process query.")
             return "Error: LLM not initialized."

        # creating a retriever interface from the vector store
        retriever = self.vector_store.as_retriever()
        
        # Create the RetrievalQA chain
        # chain_type="stuff": Simplest method, stuffs all retrieved context into one prompt
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever
        )
        
        # Run the chain to get the answer
        return self.qa_chain.run(question)

def main():
    """
    Main Streamlit UI Loop
    """
    st.set_page_config(page_title="Local RAG Chatbot", page_icon="🤖")
    st.title("🤖 Local RAG Chatbot")

    # --- Session State Management ---
    # Keeps chat history and the RAG system alive across re-runs
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = LocalRAG()

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.header("Configuration")
        # Trigger the heavy ingestion process only on button click
        if st.button("Process Documents"):
            with st.spinner("Processing documents..."):
                st.session_state.rag_system.ingest()

    # --- Chat Interface ---
    # Display previous chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle new user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_system.query(prompt)
                st.markdown(response)
        
        # Save assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
