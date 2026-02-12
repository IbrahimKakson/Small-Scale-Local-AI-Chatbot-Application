import sys
print(sys.path)
try:
    import langchain
    print(f"langchain version: {langchain.__version__}")
    print(f"langchain file: {langchain.__file__}")
    from langchain.chains import RetrievalQA
    print("RetrievalQA imported successfully")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
