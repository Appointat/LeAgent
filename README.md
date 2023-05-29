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


## Introducing _TraceTalk_ 🤖: Harnessing the Power of AI for Interactive Learning 📚

Welcome to TraceTalk 🎉, a cutting-edge chatbot designed to encapsulate the power of advanced AI technology 🧠, with a special focus on data science 💽, machine learning 📈, and deep learning 🌐. Leveraging the GPT-3.5 API and the high-performance Qdrant database 💾, our chatbot is a testament to the symbiosis of technology and thoughtful engineering 🛠️.

Drawing inspiration from New Bing’s interactive solution 💡, TraceTalk has a unique feature—it can insert relevant links 🔗 into its responses. These links serve as reference points leading to detailed and accurate sources of knowledge on [Ocademy](https://open-academy.github.io/machine-learning/intro.html) 🏫. This feature not only allows users to delve deeper into the subject matter but also inspires them to broaden their learning scope 🌍.

Our data, drawn from the expansive and ever-growing [Ocademy](https://open-academy.github.io/machine-learning/intro.html)—an open-source initiative 🔄, undergoes continuous updates and refinements. As we evolve, our plans include integrating TraceTalk with an educational data science website on AWS ☁️, thereby expanding the user's access to a wide range of services.

But what sets TraceTalk apart isn't just the advanced API it uses—it's the intricate engineering and meticulous management of the system 💻. We've put a lot of effort into optimizing the chatbot, from ensuring high-quality data and refined data processing to the development of various prompt projects 📝. The potential for further optimization and iterations on the current version is truly limitless 🔝.

To cater to diverse conversation scenarios, we've established multiple GPT-3.5 pipeline/chain types 📚:

1. Stuff Chain Type: Designed to retrieve and present relevant information succinctly 📊.
2. Map_reduce Chain Type: Specializes in summarization, offering users a brief overview of a specific topic or concept 🗺️.
3. Refine Chain Type: Enhances the coherence and fluency of the chatbot's responses, ensuring a smooth and engaging conversation 🎙️.

In addition, TraceTalk supports what we call 'selfish data.' Our database currently hosts over 400 documents in private domains 🔐, and yet we're just scratching the surface of the model's capacity.

We also acknowledge that interactions may occasionally veer off-topic or include inappropriate content 🙅. To manage such situations, we've built in mechanisms to filter out these interactions and guide the conversation back to its intended course 🧭.

In essence, TraceTalk is more than a chatbot—it's the next step in interactive learning 🎓. Our commitment to comprehensive implementation and meticulous project management ensures not only accurate and relevant responses but a seamless, informative, and engaging user experience 🌟. Welcome to the future of conversation, welcome to TraceTalk 🚀.

**Query 01**: "Introduce TraceTalk, please."

<center>
<img src="https://github.com/Appointat/Chat-with-Document-s-using-ChatGPT-API-and-Text-Embedding/assets/65004114/4c68ad40-54ee-4d64-be66-f7de445b10bd" alt="Answer 01">
</center>

**Query 02**: "I am a new learner, give me some sources of data science."

<center>
<img src="https://github.com/Appointat/Chat-with-Document-s-using-ChatGPT-API-and-Text-Embedding/assets/65004114/f4d4da99-0246-4dbd-96ee-f6dcc4180689" alt="Answer 02">
</center>

An answer from TraceTalk. As we do not provide the UI, so we reform the answer in ChatGPT to better present the quality of the original answer.

<center>
<img src="https://github.com/Appointat/Chat-with-Document-s-using-ChatGPT-API-and-Text-Embedding/assets/65004114/b5498664-462d-4be3-adb0-9357176c0472" alt="Answer with python code">
</center>



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

```powershell
$ writefile requirements.txt
openai
chromadb
langchain
tiktoken
```

```powershell
$ pip install -r requirements.txt
```

Note: administrator privileges may be required to install these packages.

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

**Questions & Answers with “Romeo and Juliet” Book**

Generating questions and answers from the book is a straightforward process. To assess the accuracy of the results, I will be comparing the answers with those from SparkNotes.

> *SparkNotes editors.* [“Romeo and Juliet” SparkNotes.com](https://www.sparknotes.com/shakespeare/romeojuliet/key-questions-and-answers/), *SparkNotes LLC, 2005*
> 

**I hope you have enjoyed this simple tutorial. If you have any questions or comments, please provide them here.**

# **Resources**

- [ChatPDF](https://www.chatpdf.com/)
- [ArvixGPT](https://chrome.google.com/webstore/detail/arxivgpt/fbbfpcjhnnklhmncjickdipdlhoddjoh)
- [GPT for Sheets and Docs](https://workspace.google.com/marketplace/app/gpt_for_sheets_and_docs/677318054654)
- [What is vector search?](https://www.elastic.co/what-is/vector-search)
- [Chroma](https://www.trychroma.com/)
- [Elasticsearch](https://www.elastic.co/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Romeo and Juliet by William Shakespeare](https://www.gutenberg.org/ebooks/1513)
- [DuckDB](https://duckdb.org/)
- [Apache Parquet](https://parquet.apache.org/)
- [What are embeddings?](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)
- [Sparknotes’ Romeo and Juliet: Questions & Answers](https://www.sparknotes.com/shakespeare/romeojuliet/key-questions-and-answers/)


# **Modules of LangChain**

There are several main modules that LangChain provides support for. For each module we provide some examples to get started, how-to guides, reference docs, and conceptual guides. These modules are, in increasing order of complexity:

- [Models](https://python.langchain.com/en/latest/modules/models.html): LLMs, Chat models, Text embedding models
- [Prompts](https://python.langchain.com/en/latest/modules/prompts.html): LLM prompts templates, Chat prompt templates, Example selectors, Output Parers
- [Indexes](https://python.langchain.com/en/latest/modules/indexes.html): The primary index and retrieval types supported by LangChain are currently centered around vector databases, and therefore a lot of the functionality we dive deep on those topics.
- [Memory](https://python.langchain.com/en/latest/modules/memory.html): LangChain provides a standard interface for memory, a collection of memory implementations, and examples of chains/agents that use memory.
- [Chains](https://python.langchain.com/en/latest/modules/chains.html): Chains go beyond just a single LLM call, and are sequences of calls (whether to an LLM or a different utility). LangChain provides a standard interface for chains, lots of integrations with other tools, and end-to-end chains for common applications.
- [Agents](https://python.langchain.com/en/latest/modules/agents.html): Tools, Agents, Toolkits, Agent exucutor
