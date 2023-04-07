# -*- coding: utf-8 -*-
"""Chat-with-Document-s-using-ChatGPT-API-and-Text-Embedding.ipynb

This Jupyter notebook demonstrates how to use the ChatGPT API and text embedding to chat with documents. 
The notebook begins by installing the necessary packages (openai, chromadb, langchain, and tiktoken) and 
importing required Python packages. The version of Python being used is printed and Google Drive is 
mounted on Colab for saving data. The OpenAI API key is set and Chroma is configured.

The notebook includes a function to retrieve text from a Gutenberg URL and uses it to load Romeo and 
Juliet from Project Gutenberg. The text is then split into chunks of 1000 tokens with 0 token overlap. 
OpenAI embeddings are generated and LangChain QA is configured with OpenAI as the LLM, using a temperature 
of 0 and the gpt-3.5-turbo model. """

# Install necessary packages
#!pip install openai
#!pip install chromadb
#!pip install langchain
#!pip install tiktoken


# Import required Python packages
import os
import platform
import textwrap
import requests
from typing import List

import openai
import chromadb
import langchain

# Import specific modules from langchain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import GutenbergLoader
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

print('Python: ', platform.python_version())


# Print the version of Python being used
print("Python: ", platform.python_version())

# Mount Google Drive on Colab for saving data
from google.colab import drive
drive.mount("/content/drive")

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "your api key"

# Configure Chroma
persist_directory = "/content/drive/My Drive/Colab Notebooks/chroma/romeo"


# Function to retrieve text from a Gutenberg URL
class GutenbergLoader(BaseLoader):
    """Loader that uses urllib to load .txt web files."""

    def __init__(self, file_path: str):
        """Initialize with file path."""
        if not file_path.startswith("https://open-academy.github.io"):
            raise ValueError("file path must start with 'https://open-academy.github.io'")

        if not file_path.endswith(".md"):
            raise ValueError("file path must end with '.md'")

        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load file."""
        from urllib.request import urlopen

        elements = urlopen(self.file_path)
        text = "\n\n".join([str(el.decode("utf-8-sig")) for el in elements])
        metadata = {"source": self.file_path}
        return [Document(page_content=text, metadata=metadata)]

def get_gutenberg(url):
    loader = GutenbergLoader(url)
    data = loader.load()
    return data

def markdown_to_python(markdown_text):
    # Escape quotes and backslashes in the input
    escaped_input = markdown_text.replace("\\", "\\\\").replace("'", "\\'")

    # Generate the Python string
    python_string = f"'{escaped_input}'"

    return python_string


# Downloading the text data from Project Open-academy
modelDeployment_md = 'https://open-academy.github.io/machine-learning/_sources/machine-learning-productionization/model-deployment.md'
modelDeployment_data = get_gutenberg(modelDeployment_md)

# Initializing a TokenTextSplitter object to split the text into chunks of 1000 tokens with 0 token overlap
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)

# Splitting the Romeo and Juliet text into chunks using the TokenTextSplitter object
modelDeployment_doc = text_splitter.split_documents(modelDeployment_data)

# Initializing an OpenAIEmbeddings object for word embeddings
embeddings = OpenAIEmbeddings()

# Generating Chroma vectors from the text chunks using the OpenAIEmbeddings object and persisting them to disk
vectordb = Chroma.from_documents(modelDeployment_doc, embeddings, persist_directory=persist_directory)
# This can be used to explicitly persist the data to disk. It will also be called automatically when the object is destroyed.
vectordb.persist() 

# Configure LangChain QA with OpenAI as the LLM, using a temperature of 0, and the gpt-3.5-turbo model
romeoandjuliet_qa = ChatVectorDBChain.from_llm(
    OpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    vectordb,
    return_source_documents=True,
)


chat_history = ''
query = "Have Romeo and Juliet spent the night together? Provide a verbose answer, referencing passages from the book."
result = romeoandjuliet_qa({"question": query, "chat_history": chat_history})
result["source_documents"] # Vector search engine
result["answer"]

markdown_text = "Generating questions and answers from the book is a straightforward process. To assess the accuracy of the results, I will be comparing the answers with those from SparkNotes. > *SparkNotes editors.* [“Romeo and Juliet” SparkNotes.com](https://www.sparknotes.com/shakespeare/romeojuliet/key-questions-and-answers/), *SparkNotes LLC, 2005* >"
query = markdown_to_python(markdown_text);
result = romeoandjuliet_qa({"question": query, "chat_history": chat_history})
chat_history = chat_history + result["answer"]
result["answer"]

##########

# restart the conversation
chat_history = [("hello", "hello")]

markdown_text = "我得到了一个字符串，写python代码，需要将字符串自动识别，输出array[str]。中文回答"

query = markdown_to_python(markdown_text)
result = romeoandjuliet_qa({"question": query, "chat_history": chat_history})
chat_history = chat_history + [(query, result["answer"])]
formatted_history = "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in chat_history])
wrapped_history = textwrap.fill(formatted_history, width=120)
print(wrapped_history + "\n")
result["answer"]




# restart the conversation
chat_history = [("", "")]
count = 0

# while loop for typing
while 1:
  markdown_text = input("\nQuery[{}]:".format(count))
  query = markdown_to_python(markdown_text)
  result = romeoandjuliet_qa({"question": query, "chat_history": chat_history})
  chat_history = chat_history + [(query, result["answer"])]
  formatted_history = "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in chat_history])
  wrapped_history = textwrap.fill(formatted_history, width=120)
  print(wrapped_history + "\n")
  result["answer"]