import os
import re
from typing import List
from collections import deque
from jinja2 import Template

import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)
from qdrant_client import QdrantClient

from prep_data import get_emmbedings

from prompts.basic_prompt import basic_prompt
from prompts.combine_prompt import combine_prompt



# chatbot agent
class ChatbotAgent:
    def __init__(self, openai_api_key: str, qdrant_url: str, qdrant_api_key: str, messages: List[str]):
        """
        Initializes an instance of the ChatbotAgent class.

        Args:
            openai_api_key (str): OpenAI API key.
        """
        # Set OpenAI API key.
        self._openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = self._openai_api_key
        self.llm = OpenAI(temperature=0.8, model_name="gpt-3.5-turbo")
        self.llm_streaming = OpenAI(streaming=True, callbacks=[FinalStreamingStdOutCallbackHandler()], temperature=0.8, model_name="gpt-3.5-turbo")

        # self.client = QdrantClient(path=r'TraceTalk\vector-db-persist-directory\Qdrant')
        self.client = QdrantClient(
                url=qdrant_url,
                prefer_grpc=False,
                api_key=qdrant_api_key,
        )
        self.client.get_collections()

        # Initialize the chat history.
        self.count = 1 # Count the number of times the chatbot has been called.
        self._max_chat_history_length = 20
        self.chat_history = deque(maxlen=self._max_chat_history_length)
        for i in range(len(messages)):
            if i % 2 == 0:
                tmp_query = messages[i]
            else:
                tmp_answer = messages[i]
                self.update_chat_history(query=tmp_query, answer=tmp_answer)
        
        self.query = ""
        self.answer = ""


    # Search agent based on Qdrant.
    def search_context_qdrant(self, query, collection_name, vector_name='content', top_k=10):
        # Create embedding vector from user query.
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
    def prompt_chatbot(self):
        prompt = basic_prompt()
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt, 
            # verbose=True,
            return_final_only=True,
        )
        return chain


    # Combine prompt.
    def prompt_combine_chain(self, query, answer_list, link_list):
        n = len(answer_list)

        if n == 0:
            return "I'm sorry, there is not enough information to provide a meaningful answer to your question. Can you please provide more context or a specific question?"
        else:
            chat_history = self.convert_chat_history_to_string()

            prompt = combine_prompt(chat_history=chat_history, query=query, answer_list=answer_list, link_list=link_list)
            # chain = LLMChain(
            #     llm=self.llm_streaming,
            #     prompt=prompt,
            #     verbose=True,
            # )
            # combine_answer = self.llm_streaming(prompt)
            # combine_answer = self.convert_links_in_text(combine_answer)
            if (prompt.count("sorry") >= 2):
                combine_answer = ""
                for answer in answer_list:
                    combine_answer += answer
                return combine_answer
                # return answer_list[0]
            return self.convert_links_in_text(prompt)
            # return combine_answer


    # Update chat history.
    def update_chat_history(self, query, answer):
        if (len(self.chat_history) == self._max_chat_history_length):
            self.chat_history.popleft()
        self.chat_history.append({
            "role": "system",
            "content": f"The session id of conversation is {self.count}."
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
    def convert_chat_history_to_string(self, new_query="", new_answser="", user_only=False, chatbot_only=False):
        if sum([bool(new_query), bool(new_answser)]) == 2:
            raise ValueError("user_only and chatbot_only cannot be True at the same time.")
        chat_string = "[chatbot]: I am TraceTalk, a cutting-edge chatbot designed to encapsulate the power of advanced AI technology, with a special focus on data science, machine learning, and deep learning. (https://github.com/Appointat/Chat-with-Document-s-using-ChatGPT-API-and-Text-Embedding)\n"
        if len(self.chat_history) > 0:
            for message in self.chat_history:
                if message['role'] == "chatbot" and ~user_only:
                    # Deleet the text (the text until to end) begin with "REFERENCE:" in the message['content'], because we do not need it.
                    chat_string += f"[{message['role']}]: {message['content'].split('REFERENCE:', 1)[0]} \n"
                elif message['role'] == "user" and ~chatbot_only:
                    chat_string += f"[{message['role']}]: {message['content']} \n"
        if new_query and new_answser:
            chat_string += f"[user]: {new_query} \n"
            chat_string += f"[chatbot]: {new_answser} \n"
        return chat_string


    def convert_links_in_text(self, text):
        links = re.findall('https://open-academy.github.io/machine-learning/[^\s]*', text)
        for link in links:
            converted_link = (
                link.replace("_sources/", "")
                .replace(".md", ".html")
                .replace("open-machine-learning-jupyter-book/", "")
            )
            text = text.replace(link, converted_link)
        return text


    # Convert Markdown to Python.
    def markdown_to_python(self, markdown_text):
        # Escape quotes and backslashes in the input
        escaped_input = markdown_text.replace("\\", "\\\\").replace("'", "\\'")

        # Generate the Python string
        python_string = f"'{escaped_input}'"

        return python_string


    def chatbot_pipeline(self, query_pipeline, choose_GPTModel = False, updateChatHistory = False):
        # choose which GPT model.
        if choose_GPTModel:
            result_pipeline = openai.Completion.create(
                engine="davinci",
                prompt = query_pipeline,
                temperature=0.7,
                max_tokens=150,
                n=1,
                stop=None,
            ).choice[0].text.strip() # Choose the first answer whose score/probability is the highest.
        else:
            result_pipeline = self.chatbot_qa({"question": query_pipeline, "chat_history": self.chat_history})

        if updateChatHistory:
            self.query = query_pipeline
            self.result = result_pipeline
            self.chat_history = self.chat_history + [(self.query, self.reslut["answer"])]
            return self.reslut
        else:
            return result_pipeline


    # Prompt the chatbot for non libary content.
    def promtp_engineering_for_non_library_content(self, query): # please do not modify the value of query
        query_prompted = query + " Please provide a verbose answer."

        result_prompted = self.chatbot_pipeline(query_prompted)
        result_not_know_answer = [] # TBD
        result_non_library_query = [] # TBD
        result_official_keywords = [] # TBD
        result_cheeting = [] # TBD
        return result_prompted