import os
import openai
import tiktoken
from dotenv import load_dotenv


def get_emmbedings(text):
    """
    Get the embeddings of the text.

    Args:
        text (str): The text to get the embeddings of.

    Returns:
        embedded_query (list): The embeddings of the text.
    """
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    embedded_query = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002",
    )["data"][0]["embedding"]

    return embedded_query  # It is a vector of numbers.


def get_tokens_number(text="", encoding_type="cl100k_base", model_name="gpt-3.5-turbo"):
    """
    Get the number of tokens in the text.

    Args:
        text (str): The text to get the number of tokens of.
        encoding_type (str): The encoding type.
        model_name (str): The model name.

    Returns:
        tokens_number (int): The number of tokens in the text.
    """
    encoding = tiktoken.get_encoding(encoding_type)
    encoding = tiktoken.encoding_for_model(model_name)
    return len(encoding.encode(text))
