import os
import platform

import openai
import chromadb
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import GutenbergLoader

# from google.colab import drive
# drive.mount('/content/drive')

# Set up the OpenAI API key
os.environ["OPENAI_API_KEY"] = "your openai api key"

# Set up Chroma to use a persistent directory on disk
persist_directory = "/content/drive/My Drive/Colab Notebooks/chroma/romeo"


# Define a function to load the data from Project Gutenberg
def get_gutenberg(url):
    loader = GutenbergLoader(url)
    data = loader.load()
    return data


# Load the data from Project Gutenberg and split it into chunks using the TokenTextSplitter class:
romeoandjuliet_data = get_gutenberg(
    "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
)

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
romeoandjuliet_doc = text_splitter.split_documents(romeoandjuliet_data)

# Convert the document to vector embedding and store it in a vector search engine using Chroma
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    romeoandjuliet_doc, embeddings, persist_directory=persist_directory
)
vectordb.persist()

romeoandjuliet_qa = ChatVectorDBChain.from_llm(
    OpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    vectordb,
    return_source_documents=True,
)


# Define a function to get a response from ChatGPT
def chat_response(prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llms = OpenAI(engine="text-davinci-002", temperature=0.7)
    chain = ChatVectorDBChain(
        llms=llms,
        vectorstore=vectordb,
        top_k=1,
        similarity_threshold=0.5,
        maximum_iterations=20,
    )
    response = chain(prompt)
    return response


if __name__ == "__main__":
    print("Python: ", platform.python_version())
    # Use the chat_response function to get a response from ChatGPT
    prompt = "What is Romeo and Juliet about?"

    response = chat_response(prompt)

    print(response)
