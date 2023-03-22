# Chat-with-Document-s-using-ChatGPT-API-and-Text-Embedding
## Main idea

The short answer is that they convert documents that are over 100 or even 1,000 pages long into a numeric representation of data and related context (**vector embedding**) and store them in a vector search engine. When a user chats with the document (i.e., asks questions), the system searches and returns similar text to a question (i.e., chat) using an algorithm called **Approximate Nearest Neighbor search (ANN)**. It looks something like this.

![****[What is vector search?](https://www.elastic.co/cn/what-is/vector-search)****](https://user-images.githubusercontent.com/65004114/226753565-e2230d59-5750-4d77-840f-4f777441a4dc.png)
Fig.1 ****[What is vector search?](https://www.elastic.co/cn/what-is/vector-search)****

- Resources (images, documents, audio)
- vector representation
- Nearest neighbor
- Output/results

The program then includes the returned text that is similar to a question (i.e., chat) in a prompt and asks the same question again to the OpenAI GPT-3 API. This returns a nice response that you are used to with ChatGPT. The benefits of this approach are that the prompt is much smaller, not 1,000 pages of documents, and the user gets the answer they are looking for.

On a side note, if you are worried about something like ChatGPT providing incorrect answers to a question, then you can point it to your organization’s knowledge base and provide answers from there. The accuracy of the answers will be as good as your knowledge base.

## Tutorial

### Tools

We will be using three tools in this tutorial:

- OpenAI GPT-3, specifically the new ChatGPT API (gpt-3.5-turbo). Not because this model is any better than other models, but because it is cheaper ($0.002 / 1K tokens) and good enough for this use case.
- [**Chroma**](https://www.trychroma.com/), the AI-native open-source embedding database (i.e., vector search engine). Chroma is an easy-to-use vector database when used in conjunction with LangChain; otherwise, it’s kind of unusable. If you want to deploy these types of applications in production, I recommend using [Elasticsearch](https://www.elastic.co/) because it has wide adoption and has been around for years. Not because Elasticsearch is better than competitors, but because not many organizations like to add a new technology stack*.*
- [**LangChain**](https://github.com/hwchase17/langchain), is a library that aims to assist developers in building applications that use Large Language Models (LLMs) by allowing them to integrate these models with other sources of computation or knowledge.

### Data

We will be using the data from Project Gutenberg’s “[Romeo and Juliet by William Shakespeare”](https://www.gutenberg.org/ebooks/1513), which consists of 55,985 tokens. This makes it a nicely sized dataset.

### Python code

**Installation of packages:**

```python
$ writefile requirements.txt
openai
chromadb
langchain
tiktoken
```

```python
$ pip install -r requirements.txt
```

**Import Python Packages**

```python
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

print('Python: ', platform.python_version())
```

**Mount Google Drive on Colab**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**OpenAI API Key**

```python
os.environ["OPENAI_API_KEY"] = 'your openai api key'
```

**Configure Chroma**

Chroma uses both of my favorite technologies for their backend — [DuckDB](https://duckdb.org/) and [Apache Parquet](https://parquet.apache.org/) — but by default, it uses an **in-memory database**. This is fine for this tutorial, but I want to give you the option of storing the database file on a disk so you can reuse the database without paying for embedding it every single time.

```python
persist_directory="/content/drive/My Drive/Colab Notebooks/chroma/romeo"
```

**Convert Document to Embedding**

Convert the document, i.e., the book, to vector embedding and store it in a vector search engine, i.e., a vector database.

```python
def get_gutenberg(url):
    loader = GutenbergLoader(url)
    data = loader.load()
    return data
```

```python
romeoandjuliet_data = get_gutenberg('https://www.gutenberg.org/cache/epub/1513/pg1513.txt')

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
romeoandjuliet_doc = text_splitter.split_documents(romeoandjuliet_data)

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(romeoandjuliet_doc, embeddings, persist_directory=persist_directory)
vectordb.persist()
```

- The first step is a bit self-explanatory, but it involves using *‘from langchain.document_loaders import GutenbergLoader’* to load a book from Project Gutenberg.
- The second step is more involved. To obtain an embedding, we need to send the text string, i.e., the book, to OpenAI’s embeddings API endpoint along with a choice of embedding model ID, e.g., *text-embedding-ada-002*. The response will contain an embedding. However, since the book consists of 55,985 tokens and the token limit for the *text-embedding-ada-002* model is 2,048 tokens, we use the *‘text_splitter’* utility (from *‘langchain.text_splitter import TokenTextSplitter’*) to split the book into manageable 1,000-token chunks. The following is an illustration of a sample embedding response from OpenAI. If you’re wondering, the pricing for the embedding model is $0.0004 / 1K tokens.
- The third step is pretty straightforward: we store the embedding in Chroma, our vector search engine, and persist it on a file system.

**Configure LangChain QA**

To configure LangChain QA with Chroma, use the OpenAI GPT-3 model (*`model_name='gpt-3.5-turbo'`*) and ensure that the response includes the intermediary step of a result from a vector search engine, i.e., Chroma (set *`return_source_documents=True`*).

```python
romeoandjuliet_qa = ChatVectorDBChain.from_llm(OpenAI(temperature=0, model_name="gpt-3.5-turbo"), vectordb, return_source_documents=True)
```

**Deployment of API**

```python
import openai
import time

# Set up the API client
openai.api_key = "YOUR_API_KEY"

# Set up the prompt for the conversation
prompt = "Let's have a conversation with an AI. Ask me anything!"

# Set up the initial variables for the conversation
conversation_history = ""
end_conversation = False

# Loop for the conversation
while not end_conversation:
    # Generate a response to the prompt using the GPT-3 API
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt + conversation_history,
        temperature=0.7,
        max_tokens=150,
        n=1,
        stop=None,
    )

    # Extract the generated response text from the API response
    message = response.choices[0].text.strip()

    # Print the response to the console
    print("AI: " + message)

    # Check if the response indicates the end of the conversation
    if "Goodbye" in message:
        end_conversation = True
    else:
        # Get the user's response and add it to the conversation history
        user_response = input("You: ")
        conversation_history += "\nYou: " + user_response

    # Add a delay to prevent exceeding API usage limits
    time.sleep(1.0)
```

This code sets up a conversation loop where the user enters their message and the AI responds with a generated message. The **`prompt`** variable is used to set up the initial conversation prompt, and the **`conversation_history`** variable is used to keep track of the previous messages in the conversation.

Inside the loop, the OpenAI API is used to generate a response to the current conversation prompt (**`prompt + conversation_history`**). The generated response is then printed to the console, and the user is prompted to enter their response. The user's response is added to the **`conversation_history`** variable, and the loop continues.

The conversation ends when the AI generates a message containing the word "Goodbye". At that point, the **`end_conversation`** variable is set to **`True`**, and the loop exits. Note that there's a 1-second delay after each API call to prevent exceeding the API usage limits. Also, you need to replace "YOUR_API_KEY" with your actual API key.
