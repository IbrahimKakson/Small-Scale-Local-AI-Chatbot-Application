try:
    from langchain_community.document_loaders import PyPDFLoader
    print("PyPDFLoader imported")
    from langchain_community.vectorstores import Chroma
    print("Chroma imported")
    from langchain_community.llms import LlamaCpp
    print("LlamaCpp imported")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("HuggingFaceEmbeddings imported")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("RecursiveCharacterTextSplitter imported")
    from langchain.chains import RetrievalQA
    print("RetrievalQA imported")
    import streamlit
    print("Streamlit imported")
    print("All imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
