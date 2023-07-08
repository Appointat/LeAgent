# Import basic libraries.
import os
import warnings
import ast
from dotenv import load_dotenv
import pandas as pd
import re
import requests
import openai

# Import Qdrant libraries.
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest


def prep_book_data(
    csv_file_path=r"TraceTalk\vector-db-persist-directory\book dada\book data.csv",
):
    """
    This function prepares the book data for the vector database.

    Args:
        csv_file_path (str): The path to the CSV file containing the book data.

    Returns:
        book_data (list): A list of dictionaries containing the book data.
    """
    book_data = []  # Create an empty list to store book data.
    id = 0  # Set an initial value for the ID counter.

    input_directory = (
        r"TraceTalk\vector-db-persist-directory\resources"  # Set the defualt directory.
    )

    for file in os.listdir(input_directory):
        if file.endswith(".txt"):
            with open(os.path.join(input_directory, file), "r") as f:
                txt_content = f.read()
                # Link all Markdown files extracted from the text file.
                md_links = re.findall(r"'(https://[\w\d\-_/.]+\.md)',", txt_content)

            for link in md_links:
                md_file = link.rsplit("/", 1)[-1]
                md_title = md_file[:-3]  # Remove the .md suffix.

                # Get the contents of the .md file.
                converted_link = (
                    link.replace("github.com/open-academy", "ocademy-ai.github.io")
                    .replace("tree/main", "_sources")
                    .replace("open-machine-learning-jupyter-book/", "")
                )
                md_content_request = requests.get(converted_link)
                md_content = (
                    md_content_request.text
                    if md_content_request.status_code == 200
                    else ""
                )
                print(f"Processing {md_title}: {link}...")

                md_content_split = split_text_into_chunks(
                    md_content, chunk_size=500
                )  # Split the text into chunks.
                for text in md_content_split:
                    if not text:
                        continue
                    id = id + 1

                    # Add the book data to the list.
                    book_data.append(
                        {
                            "id": id,
                            "title": md_title,
                            "title_vector": get_emmbedings(md_title),
                            "content": text,  # Further optimization can be done by splitting files to reduce text volume.
                            "content_vector": get_emmbedings(
                                text
                            ),  # Further optimization can be done by splitting files to reduce text volume.
                            "link": converted_link,
                        }
                    )

    # Convert the book_data list into a pandas DataFrame.
    book_data_df = pd.DataFrame(book_data)
    print("Shape of the book data DataFrame:", book_data_df.shape)
    book_data_df.to_csv(csv_file_path, index=False)


def update_collection_to_database(
    csv_file_path=r"TraceTalk\vector-db-persist-directory\book dada\book data.csv",
):
    """
    This function updates the Qdrant database collection with the book data.

    Args:
        csv_file_path (str, optional): The path to the CSV file containing the book data. Defaults to r"TraceTalk\vector-db-persist-directory\book dada\book data.csv".
    """
    # Load the book data DataFrame.
    book_data_df = pd.read_csv(csv_file_path)

    def convert_string_to_list(s):
        return ast.literal_eval(s)

    book_data_df["title_vector"] = book_data_df["title_vector"].apply(
        convert_string_to_list
    )
    book_data_df["content_vector"] = book_data_df["content_vector"].apply(
        convert_string_to_list
    )

    # Idex client.
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable not set.")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY environment variable not set.")
    # client = QdrantClient(path=r'TraceTalk\vector-db-persist-directory\Qdrant')
    client = QdrantClient(
        url=qdrant_url,
        prefer_grpc=False,
        api_key=qdrant_api_key,
    )

    # Create a new collection of the Qdrant database.
    vector_size = 1536
    client.recreate_collection(
        collection_name="Articles",
        vectors_config={
            "title": rest.VectorParams(
                distance=rest.Distance.COSINE,
                size=vector_size,
            ),
            "content": rest.VectorParams(
                distance=rest.Distance.COSINE,
                size=vector_size,
            ),
        },
        optimizers_config=rest.OptimizersConfigDiff(
            indexing_threshold=0,
        ),
    )

    # Upsert the data into the collection of the Qdrant database.
    batch_size = 50  # Adjust this value to fit within Qdrant's size limits.
    # Divide data into batches.
    batches = [
        book_data_df[i : i + batch_size]
        for i in range(0, book_data_df.shape[0], batch_size)
    ]

    for batch in batches:
        points = []
        for _, row in batch.iterrows():
            point = rest.PointStruct(
                id=row["id"],
                vector={
                    "title": row["title_vector"],
                    "content": row["content_vector"],
                },
                payload=row.to_dict(),
            )
            points.append(point)
            print(f"Upserting point with id: {row['id']}")

        client.upsert(
            collection_name="Articles",
            points=points,
        )


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


def split_text_into_chunks(text, delimiter="\n# ", chunk_size=500):
    begin_pattern = r"---.*?---"
    text = re.sub(begin_pattern, "", text, flags=re.DOTALL)
    text = "\n" + text

    # Remove code cells from text.
    code_pattern = r"(```{code-cell}.*?```)"
    code_cells = re.findall(code_pattern, text, flags=re.DOTALL)
    text = re.sub(code_pattern, "TEMPLATE_CODE_CELL\n", text, flags=re.DOTALL)

    chunks = re.split(
        "((?:^|\n)(?={}(?!#)))".format(delimiter), text, flags=re.MULTILINE
    )
    chunks = [chunk for chunk in chunks if chunk.strip()]

    final_chunks = []
    for chunk in chunks:
        words = re.split(
            r"(\s)", chunk
        )  # Split a string using regex, preserving spaces and newlines.
        current_chunk_words = []
        current_word_count = 0
        for word in words:
            # If word is a space or a newline, don't count the word; otherwise, add 1 to the word count.
            if word.strip() != "":
                current_word_count += 1
            current_chunk_words.append(word)

            # When the number of words reaches chunk_size, add a new chunk.
            if current_word_count >= chunk_size:
                final_chunks.append("".join(current_chunk_words))
                current_chunk_words = []
                current_word_count = 0
        # Add the last chunk.
        if current_chunk_words:
            final_chunks.append("".join(current_chunk_words))

    # Count the number of words in a string.
    count_words_in_string = lambda s: len(s.split())

    for i, chunk in enumerate(final_chunks):
        try:
            while "TEMPLATE_CODE_CELL" in chunk and code_cells:
                code_cell = code_cells.pop(0)

                # Split the code cell into lines.
                code_cell_lines = code_cell.split("\n")

                # If the code cell can be inserted without exceeding the chunk size, do it.
                if (
                    count_words_in_string(
                        chunk.replace("TEMPLATE_CODE_CELL", code_cell, 1)
                    )
                    <= 1000
                ):  # chunk_size
                    chunk = chunk.replace("TEMPLATE_CODE_CELL", code_cell, 1)
                # If not, insert as many lines as possible.
                else:
                    temp_chunk = chunk
                    for line in code_cell_lines:
                        # If the next line can be inserted without exceeding the chunk size, do it.
                        if (
                            count_words_in_string(
                                temp_chunk.replace("TEMPLATE_CODE_CELL", line, 1)
                            )
                            <= 1000
                        ):  # chunk_size
                            temp_chunk = temp_chunk.replace(
                                "TEMPLATE_CODE_CELL", line, 1
                            )
                        # If not, put the remaining lines back into code_cells and stop.
                        else:
                            break
                    chunk = temp_chunk

            final_chunks[i] = chunk
        except IndexError:
            warnings.warn(
                "Code cells mismatch. The number of 'TEMPLATE_CODE_CELL' placeholders and actual code cells do not match."
            )

    return final_chunks