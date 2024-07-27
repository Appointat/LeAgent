import os

from openai import OpenAI
import tiktoken


def get_emmbeddings(text):
    """
    Get the embeddings of the text.

    Args:
        text (str): The text to get the embeddings of.

    Returns:
        embedded_query (list): The embeddings of the text.
    """
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    embedded_query = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002",
    ).data[0].embedding

    return embedded_query  # It is a vector of numbers.


def get_tokens_number(text="", encoding_type="cl100k_base", model_name="gpt-4o-mini"):
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
