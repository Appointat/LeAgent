# Import basic libraries.
import os
import re
from typing import List
from collections import deque
from dotenv import load_dotenv
from src import get_emmbedings, get_tokens_number

# Import OpenAI API and Langchain libraries.
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout_final_only import (
    FinalStreamingStdOutCallbackHandler,
)

# Import Qdrant client (vector database).
from qdrant_client import QdrantClient

# Importing prompts.
from prompts.basic_prompt import basic_prompt
from prompts.combine_prompt import combine_prompt


class ChatbotAgent:
    """
    A class used to represent a chatbot agent.
    """

    def __init__(
        self,
        openai_api_key: str,
        qdrant_url: str,
        qdrant_api_key: str,
        messages: List[str],
    ):
        """
        Initializes an instance of the ChatbotAgent class.

        Args:
            openai_api_key (str): The API key provided by OpenAI to authenticate requests.
            qdrant_url (str): The URL for the Qdrant service to connect with.
            qdrant_api_key (str): The API key for the Qdrant service to authenticate requests.
            messages (List[str]): A list of messages that the chatbot agent will process.

        """
        # Set OpenAI API key.
        self._openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = self._openai_api_key
        self.llm = OpenAI(temperature=0.8, model_name="gpt-3.5-turbo")
        self.llm_dot3 = OpenAI(temperature=0.3, model_name="gpt-3.5-turbo")
        self.llm_streaming = OpenAI(
            streaming=True,
            callbacks=[FinalStreamingStdOutCallbackHandler()],
            temperature=0.8,
            model_name="gpt-3.5-turbo",
        )

        # self.client = QdrantClient(path=r'TraceTalk\vector-db-persist-directory\Qdrant')
        self.client = QdrantClient(
            url=qdrant_url,
            prefer_grpc=False,
            api_key=qdrant_api_key,
        )
        self.client.get_collections()

        # Initialize the chat history.
        self.count = 1  # Count the number of times the chatbot has been called.
        self._max_chat_history_length = 20
        self.chat_history = deque(maxlen=self._max_chat_history_length)
        init_prompt = "I am TraceTalk, a cutting-edge chatbot designed to encapsulate the power of advanced AI technology, with a special focus on data science, machine learning, and deep learning. (https://github.com/Appointat/Chat-with-Document-s-using-ChatGPT-API-and-Text-Embedding)\n"
        self.chat_history.append({"role": "chatbot", "content": init_prompt})
        for i in range(len(messages)):
            if i % 2 == 0:
                self.chat_history.append({"role": "user", "content": messages[i]})
            else:
                self.chat_history.append({"role": "chatbot", "content": messages[i]})

        self.query = ""
        self.answer = ""

    # Search agent based on Qdrant.
    def search_context_qdrant(
        self, query, collection_name, vector_name="content", top_k=10
    ):
        """
        Search the Qdrant database for the top k most similar vectors to the query.

        Args:
            query (str): The query to search for.
            collection_name (str): The name of the collection to search in.
            vector_name (str): The name of the vector to search for.
            top_k (int): The number of results to return.

        Returns:
            query_results (list): A list of the top k most similar vectors to the query.
        """
        # Create embedding vector from user query.
        embedded_query = get_emmbedings(query)

        query_results = self.client.search(
            collection_name=collection_name,
            query_vector=(vector_name, embedded_query),
            limit=top_k,
        )

        return query_results

    # Prompt the chatbot.
    def prompt_chatbot(self):
        """
        Prompt the chatbot to generate a response.

        Returns:
            chatbot_answer (str): The chatbot's response to the user's query.
        """
        prompt = basic_prompt()
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt,
            # verbose=True,
            return_final_only=True,
        )
        return chain

    # Combine prompt.
    def prompt_combine_chain(self, query, answer_list, link_list_list):
        """
        Prompt the chatbot to generate a response.

        Args:
            query (str): The user's query.
            answer_list (list): A list of answers to the user's query.
            link_list (list): A list of links to the user's query.
        Returns:
            chatbot_answer (str): The chatbot's response to the user's query.
        """
        MAX_TOKENS_CHAT_HISTORY = 800
        n = len(answer_list)

        if n == 0:
            return "I'm sorry, there is not enough information to provide a meaningful answer to your question. Can you please provide more context or a specific question?"
        else:
            chat_history = self.convert_chat_history_to_string()
            if (
                get_tokens_number(chat_history) >= MAX_TOKENS_CHAT_HISTORY
            ):  # Max token length for GPT-3 is 4096.
                print(
                    f"Warning: chat history is too long, tokens: {get_tokens_number(chat_history)}."
                )
                chat_history = self.convert_chat_history_to_string(
                    user_only=True, remove_resource=True
                )

            prompt = combine_prompt(
                chat_history=chat_history,
                query=query,
                answer_list=answer_list,
                link_list_list=link_list_list,
                MAX_TOKENS=4096 - 700 - MAX_TOKENS_CHAT_HISTORY,
            )
            prompt = self.convert_links_in_text(prompt)

            # responses = openai.Completion.create(
            #     engine="davinci",
            #     prompt=prompt,
            #     max_tokens=500,
            #     stream=True
            # )

            if get_tokens_number(prompt) > 4096 - 500:
                return "Tokens number of the prompt is too long: {}.".format(
                    get_tokens_number(prompt)
                )
            else:
                print("Tokens number of the prompt: {}.".format(get_tokens_number(prompt)))

            # return responses[0]["text"]
            return prompt

    # Update chat history.
    def update_chat_history(self, query, answer):
        """
        Update the chat history with the user's query and the chatbot's response.

        Args:
            query (str): The user's query.
            answer (str): The chatbot's response to the user's query.
        """
        if len(self.chat_history) == self._max_chat_history_length:
            self.chat_history.popleft()

        self.chat_history.append({"role": "user", "content": query})
        self.chat_history.append({"role": "chatbot", "content": answer})
        self.count += 1

    # Convert chat history to string.
    def convert_chat_history_to_string(
        self,
        new_query="",
        new_answser="",
        user_only=False,
        chatbot_only=False,
        remove_resource=False,
    ):
        """
        Convert the chat history to a string.

        Args:
            new_query (str): The user's query.
            new_answser (str): The chatbot's response to the user's query.
            user_only (bool): If True, only return the user's queries.
            chatbot_only (bool): If True, only return the chatbot's responses.

        Returns:
            chat_string (str): The chat history as a string.
        """
        if sum([bool(user_only), bool(chatbot_only)]) == 2:
            raise ValueError(
                "user_only and chatbot_only cannot be True at the same time."
            )
        chat_string = ""
        if len(self.chat_history) > 0:
            for message in self.chat_history:
                if message["role"] == "chatbot" and ~user_only:
                    # Deleet the text (the text until to end) begin with "REFERENCE:" in the message['content'], because we do not need it.
                    if remove_resource:
                        chat_string += f"[{message['role']}]: {message['content'].split('RESOURCE:', 1)[0].split('REFERENCE', 1)[0]} \n"
                    else:
                        chat_string += f"[{message['role']}]: {message['content']} \n"
                elif message["role"] == "user" and ~chatbot_only:
                    chat_string += f"[{message['role']}]: {message['content']} \n"
        if new_query:
            chat_string += f"[user]: {new_query} \n"
        if new_answser:
            chat_string += f"[chatbot]: {new_answser} \n"

        if (
            get_tokens_number(chat_string) >= 3000
        ):  # Max token length for GPT-3 is 4096.
            print(
                f"Chat history is too long: {get_tokens_number(chat_string)} tokens. Truncating chat history."
            )
        return chat_string

    def convert_links_in_text(self, text):
        """
        Convert links in the text to the correct format.

        Args:
            text (str): The text to convert.

        Returns:
            text (str): The text with the links converted.
        """
        links = re.findall(
            "https://open-academy.github.io/machine-learning/[^\s]*", text
        )
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
        """
        Convert Markdown text to Python string.

        Args:
            markdown_text (str): The Markdown text to convert.

        Returns:
            python_string (str): The Python string.
        """
        # Escape quotes and backslashes in the input.
        escaped_input = markdown_text.replace("\\", "\\\\").replace("'", "\\'")

        # Generate the Python string
        python_string = f"'{escaped_input}'"

        return python_string

    def chatbot_pipeline(
        self, query_pipeline, choose_GPTModel=False, updateChatHistory=False
    ):
        """
        Chat with the chatbot using the pipeline.

        Args:
            query_pipeline (str): The user's query.
            choose_GPTModel (bool): If True, choose the GPT model.
            updateChatHistory (bool): If True, update the chat history.

        Returns:
            result_pipeline (str): The chatbot's response to the user's query.
        """
        # choose which GPT model.
        if choose_GPTModel:
            result_pipeline = (
                openai.Completion.create(
                    engine="davinci",
                    prompt=query_pipeline,
                    temperature=0.7,
                    max_tokens=150,
                    n=1,
                    stop=None,
                )
                .choice[0]
                .text.strip()
            )  # Choose the first answer whose score/probability is the highest.
        else:
            result_pipeline = self.chatbot_qa(
                {"question": query_pipeline, "chat_history": self.chat_history}
            )

        if updateChatHistory:
            self.query = query_pipeline
            self.result = result_pipeline
            self.chat_history = self.chat_history + [
                (self.query, self.reslut["answer"])
            ]
            return self.reslut
        else:
            return result_pipeline

    # Prompt the chatbot for non libary content.
    def promtp_engineering_for_non_library_content(self, query):
        """
        Prompt the chatbot for non libary content.

        Args:
            query (str): The user's query.

        Returns:
            result_prompted (str): The chatbot's response to the user's query.
        """
        # Please do not modify the value of query.
        query_prompted = query + " Please provide a verbose answer."

        result_prompted = self.chatbot_pipeline(query_prompted)
        result_not_know_answer = []  # TBD
        result_non_library_query = []  # TBD
        result_official_keywords = []  # TBD
        result_cheeting = []  # TBD
        return result_prompted
