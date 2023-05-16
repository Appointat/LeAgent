## Import Python Packages
from collections import deque
import os

import openai
import langchain

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ChatVectorDBChain
from langchain.chains import RetrievalQA

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from prep_data import prep_data
from prep_data import get_emmbedings



# chatbot agent
class ChatbotAgent:
    def __init__(self, _openai_api_key: str):
        """
        Initializes an instance of the ChatbotAgent class.

        Args:
            openai_api_key (str): OpenAI API key.
        """
        # Set OpenAI API key.
        self.__openai_api_key = _openai_api_key
        os.environ["OPENAI_API_KEY"] = self.__openai_api_key
        self.llm = OpenAI(temperature=0.8, model_name="gpt-3.5-turbo")

        self.client = QdrantClient(path=r'chatbot\vector-db-persist-directory\Qdrant')
        self.client.get_collections()
        #pirnt("\nNumber of the collection: ".format(client.count(collection_name='Articles')))

        # Initialize the chat history
        self._max_chat_history_length = 20
        self.chat_history = deque(maxlen=self._max_chat_history_length)
        self.query = ""
        self.answer = ""
        self.count = 1 # count the number of times the chatbot has been called


    # Search agent based on Qdrant.
    def search_context_qdrant(self, query, collection_name, vector_name='content', top_k=10):
            
        # Creates embedding vector from user query.
        embedded_query = get_emmbedings(query)
    
        query_results = self.client.search(
            collection_name=collection_name,
            query_vector=(
                vector_name, embedded_query
            ),
            limit=top_k,
        )

        return query_results


    # Prompt the chatbot.
    def prompt_the_chatbot(self):
        template="""
Context information is below: 
{context}
=========
Chat_history:
{chat_history}
=========
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES", the refernces do not include links). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
Respond in English.

QUESTION: {qury}
=========
{summaries}
=========
FINAL ANSWER IN ENGLISH:
            """
        prompt = PromptTemplate(template=template, input_variables=["context", "chat_history", "summaries", "qury"]) # Parameter the prompt template
        chain = LLMChain(
            llm=self.llm, 
            prompt=prompt,
            verbose=True,
        )
        return chain

    # Combine prompt.
    def combine_prompt(self):
        template = """
{context}

=========
FINAL ANSWER IN ENGLISH:
"""
        prompt = PromptTemplate(template=template, input_variables=["context"]) # Parameter the prompt template
        chain = LLMChain(
            llm=self.llm, 
            prompt=prompt,
            verbose=True,
        )
        return chain


    # Update chat history.  
    def update_chat_history(self, query, answer):
        if (len(self.chat_history) == self._max_chat_history_length):
            self.chat_history.popleft()
        self.chat_history.append({
            "role": "system",
            "content": f"You are now chatting with your AI assistant. The session id is {self.count}."
        })
            
        self.chat_history.append({
            "role": "user",
            "content": query
        })
        self.chat_history.append({
            "role": "chatbot",
            "content": answer
        })
        self.count += 1


    # Convert chat history to string.
    def convert_chat_history_to_string(self, new_query="", new_answser=""):
        chat_string = ""
        if len(self.chat_history) > 0:
            for message in self.chat_history:
                chat_string += f"{message['role']}: {message['content']}\n"
        if new_query and new_answser:
            chat_string += f"user: {new_query}\n"
            chat_string += f"chatbot: {new_answser}\n"
        return chat_string

    # Convert Markdown to Python
    def markdown_to_python(self, markdown_text):
        # Escape quotes and backslashes in the input
        escaped_input = markdown_text.replace("\\", "\\\\").replace("'", "\\'")

        # Generate the Python string
        python_string = f"'{escaped_input}'"

        return python_string


    def chatbot_pipeline(self, query_pipeline, choose_GPTModel = False, updateChatHistory = False):
        # choose which GPT model
        if choose_GPTModel:
            result_pipeline = openai.Completion.create(
                engine="davinci",
                prompt = query_pipeline,
                temperature=0.7,
                max_tokens=150,
                n=1,
                stop=None,
            ).choice[0].text.strip() # choose the first answer whose score/probability is the highest
        else:
            result_pipeline = self.chatbot_qa({"question": query_pipeline, "chat_history": self.chat_history})
        
        if updateChatHistory:
            self.query = query_pipeline
            self.result = result_pipeline
            self.chat_history = self.chat_history + [(self.query, self.reslut["answer"])]
            return self.reslut
        else:
            return result_pipeline
            


    # Prompt the chatbot for map_reduce chain type
    def promtp_engineering_for_map_reduce_chain_type(self):
        self.chatbot_qchatbot_qa_retrieval_map_reduce_chain_type({"input_documents": self.vectordb, "question": self.query}, return_only_outputs=True)
        # {'output_text': ' The president thanked Justice Breyer for his service.\nSOURCES: 30-pl'}


    # Prompt the chatbot for non libary content
    def promtp_engineering_for_non_library_content(self, query): # please do not modify the value of query
        query_prompted = query + " Please provide a verbose answer."

        result_prompted = self.chatbot_pipeline(query_prompted)
        result_not_know_answer = []
        result_non_library_query = []
        result_official_keywords = []
        result_cheeting = []
        # 
        #
        #
        # Return the prompted query
        return result_prompted