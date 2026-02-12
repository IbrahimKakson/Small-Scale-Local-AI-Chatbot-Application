import os
import langchain
print(f"Langchain path: {os.path.dirname(langchain.__file__)}")
try:
    print(os.listdir(os.path.dirname(langchain.__file__)))
except Exception as e:
    print(e)
