# Import basic libraries.
import os
import re
from collections import deque
from typing import List

# Import OpenAI API and Langchain libraries.
from openai import OpenAI
from langchain.prompts import PromptTemplate

# Importing prompts.
from prompts.basic_prompt import basic_prompt
from prompts.combine_prompt import combine_prompt

# Import Qdrant client (vector database).
from qdrant_client import QdrantClient
from src import get_emmbeddings, get_tokens_number


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
        # Set OpenAI API key and initialize client.
        self._openai_api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = self._openai_api_key
        self.client = OpenAI(api_key=self._openai_api_key)

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            prefer_grpc=False,
            api_key=qdrant_api_key,
        )
        self.qdrant_client.get_collections()

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
        embedded_query = get_emmbeddings(query)

        query_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=(vector_name, embedded_query),
            limit=top_k,
        )

        return query_results

    def prompt_chatbot(self, context, chat_history, resource, query):
        """
        Prompt the chatbot to generate a response.

        Args:
            context (str): The context for the query.
            chat_history (str): The chat history.
            resource (str): The resource string.
            query (str): The user's query.

        Returns:
            str: The chatbot's response to the user's query.
        """
        prompt = f"Context: {context}\nChat History: {chat_history}\nResources: {resource}\nQuestion: {query}\nPlease provide an answer based on the given context and resources."
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8
        )
        return response.choices[0].message.content

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
        MAX_TOKENS_CHAT_HISTORY = 1000
        n = len(answer_list)

        if n == 0:
            return "I'm sorry, there is not enough information to provide a meaningful answer to your question. Can you please provide more context or a specific question?"
        else:
            chat_history = self.convert_chat_history_to_string()
            if get_tokens_number(chat_history) >= MAX_TOKENS_CHAT_HISTORY:
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
                MAX_TOKENS=4096 - 1000,
            )
            prompt = self.convert_links_in_text(prompt)

            if get_tokens_number(prompt) > 4096 - 1000:
                return "Tokens number of the prompt is too long: {}.".format(
                    get_tokens_number(prompt)
                )
            else:
                print(
                    "Tokens number of the prompt: {}.".format(get_tokens_number(prompt))
                )

            # return prompt
            # Use the OpenAI API to generate a response based on the prompt
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=1000,
                n=1,
                stop=None,
            )

            # Extract and return the generated response
            print(f"Chatbot response:\n{response.choices[0].message.content.strip()}")
            return response.choices[0].message.content.strip()

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
            remove_resource (bool): If True, remove the resource from the chatbot's responses.

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
                    # Delete the text (the text until to end) begin with "REFERENCE:" in the message['content'], because we do not need it.
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

        if get_tokens_number(chat_string) >= 3000:  # Max token length for GPT-3 is 4096.
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
            response = self.client.completions.create(
                model="davinci",
                prompt=query_pipeline,
                temperature=0.7,
                max_tokens=150,
                n=1,
                stop=None,
            )
            result_pipeline = response.choices[0].text.strip()
        else:
            result_pipeline = self.chatbot_qa(
                {"question": query_pipeline, "chat_history": self.chat_history}
            )

        if updateChatHistory:
            self.query = query_pipeline
            self.result = result_pipeline
            self.chat_history = self.chat_history + [
                (self.query, self.result["answer"])
            ]
            return self.result
        else:
            return result_pipeline

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
        # result_not_know_answer = []  # TBD
        # result_non_library_query = []  # TBD
        # result_official_keywords = []  # TBD
        # result_cheeting = []  # TBD
        return result_prompted

    def chatbot_qa(self, input_data):
        """
        Process the input data and generate a response using the chatbot.

        Args:
            input_data (dict): A dictionary containing the question and chat history.

        Returns:
            dict: A dictionary containing the chatbot's response.
        """
        question = input_data["question"]
        chat_history = input_data["chat_history"]

        # Prepare the messages for the ChatCompletion API
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
        ]
        
        # Add chat history to the messages
        for entry in chat_history:
            if isinstance(entry, dict):
                messages.append({"role": entry["role"], "content": entry["content"]})
            elif isinstance(entry, tuple):
                messages.append({"role": "user", "content": entry[0]})
                messages.append({"role": "assistant", "content": entry[1]})

        # Add the current question
        messages.append({"role": "user", "content": question})

        # Call the OpenAI API
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=150,
            n=1,
            stop=None,
        )

        # Extract the response
        answer = response.choices[0].message.content.strip()

        return {"answer": answer}
