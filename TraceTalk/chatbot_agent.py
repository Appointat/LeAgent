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
        template = """
CONTEXT information is below:
{{context}}
=========
Chat_history:
{{chat_history}}
=========
Given the following extracted parts of a long document and a QUESTION, create a final answer with references ("SOURCES", the refernces do not include links).
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
If the QUESTION is not associate with the CONTEXT, you can say "SORRY" and say that you need more information to answer it, or you can enven refuse to answer it.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: {{query}}
=========
{{summaries}}
=========
FINAL ANSWER THE QUESTION {{query}}, language used for answers is CONSISTENT with QUESTION:
"""
        prompt = PromptTemplate(template=template, input_variables=["context", "chat_history", "summaries", "query"], template_format="jinja2", validate_template=False) # Parameter the prompt template
        chain = LLMChain(
            llm=self.llm,
            prompt=prompt, 
            verbose=True,
        )
        return chain


    # Combine prompt.
    def prompt_combine_chain(self, query, answer_list, link_list):
        n = len(answer_list)
        if n == 0:
            return "I'm sorry, there is not enough information to provide a meaningful answer to your question. Can you please provide more context or a specific question?"
        else:
            chat_history = self.convert_chat_history_to_string()
            template = f"""
Now I will provide you with {n} chains, here is the definition of chain: each chain contains an answer and a link. The answers in the chain are the results from the links.
In theory, each chain should produce a paragraph with links as references. It means that you MUST tell me from which references you make the summery.
The smaller the number of the chain, the more important the information contained in the chain.
Your final answer is verbose.
But if the meaning of an answer in a certain chain is similar to 'I am not sure about your question' or 'I refuse to answer such a question', it means that this answer chain is deprecated, and you should actively ignore the information in this answer chain.

You now are asked to COMBINE these {n} chains (combination means avoiding repetition, writing logically, smooth writing, giving verbose answer), and divide it into 2-4 paragraphs appropriately.
The final answer is ALWAYS in the form of TEXT WITH MD LINK. If no refernce for one sentence, you do not need to attach the link to that sentence.
In addition, ALWAYS return "TEXT WITH MD LINK", and ALSO ALWAYs return a "REFERENCE" part in your answer (they are two parts).
ReFERENCE can ONLY be a list of links, each link is a reference for a sentence in the answer.

For exmaple:
    I provide the input text:
		CHAIN 1:
			CONTEXT:
                Text of chain 1. ABCDEFGHIJKLMNOPQRSTUVWXYZ
			REFERENCE:
				https://link1.com
		CHAIN 2:
			CONTEXT: 
                Text of chain 2. ABCDEFGHIJKLMNOPQRSTUVWXYZ
			REFERENCE:
                https://link2.com
    Your output should be:
		COMBINATION:
            Text of combined chain 1 and chain 2. blablabla.
        REFERENCE:
			[1] https://link1.com
			[2] https://link2.com
=========
"""
            template += """
QUESTION: {{query}}
Chat_history: 
{{chat_history}}[user]: {{query}}

=========
"""
            for i in range(n):
                template += f"""
### CHAIN {i+1}:
CONTEXT:
    {answer_list[i]}
REFERENCE:
    {link_list[i]}
    """
            template += """
=========
ANSWER THE QUESTION {{query}}, FINAL A VERBOSE ANSWER, language used for answers is CONSISTENT with QUESTION:
"""
            # prompt = PromptTemplate(template=template, input_variables=["query", "chat_history"], template_format="jinja2", validate_template=False) # Parameter the prompt template
            prompt = Template(template).render(query=query, chat_history=chat_history)
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