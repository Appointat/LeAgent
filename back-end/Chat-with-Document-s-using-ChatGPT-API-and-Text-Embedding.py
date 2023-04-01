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


# Load Romeo and Juliet from Project Gutenberg
romeoandjuliet_data = get_gutenberg(
    "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
)

# Split the text into chunks of 1000 tokens with 0 token overlap
text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
romeoandjuliet_doc = text_splitter.split_documents(romeoandjuliet_data)

# Generate OpenAI embeddings
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    romeoandjuliet_doc, embeddings, persist_directory=persist_directory
)
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