# Import basic libraries.
import ast
import os
import re
import warnings

import pandas as pd
import requests

# Import Qdrant libraries.
from qdrant_client import QdrantClient, models
from src import get_emmbeddings, get_tokens_number


def prep_book_data(
    csv_file_path=r"vector-db-persist-directory/book data/book data.csv",
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
        r"vector-db-persist-directory/resources"  # Set the defualt directory.
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
                    md_content, chunk_max_tokens=300
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
                            "title_vector": get_emmbeddings(md_title),
                            "content": text,  # Further optimization can be done by splitting files to reduce text volume.
                            "content_vector": get_emmbeddings(
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
    csv_file_path=r"vector-db-persist-directory/book data/book data.csv",
):
    """
    This function updates the Qdrant database collection with the book data.

    Args:
        csv_file_path (str, optional): The path to the CSV file containing the book data.
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

    # Initialize client.
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        raise ValueError("QDRANT_URL environment variable not set.")
    print("QDRANT_URL:", qdrant_url)
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_api_key:
        raise ValueError("QDRANT_API_KEY environment variable not set.")
    print("QDRANT_API_KEY:", qdrant_api_key)

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Create a new collection of the Qdrant database.
    vector_size = 1536
    client.recreate_collection(
        collection_name="Articles",
        vectors_config={
            "title": models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
            "content": models.VectorParams(
                size=vector_size, distance=models.Distance.COSINE
            ),
        },
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
            point = models.PointStruct(
                id=row["id"],
                vector={
                    "title": row["title_vector"],
                    "content": row["content_vector"],
                },
                payload=row.to_dict(),
            )
            points.append(point)
            print(f"Upserting point with id: {row['id']}")

        client.upsert(collection_name="Articles", points=points)


def split_text_into_chunks(
    text, delimiter="\n# ", chunk_max_tokens=600, MAX_TOKENS=4096
):
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
        # Split the chunk into the sentences.
        sentences = re.split(r"([.?!])", chunk)
        current_n_sentences = []
        for sentence in sentences:
            current_n_sentences.append(sentence)

            # When the number of words reaches chunk_size, add a new chunk.
            if get_tokens_number("".join(current_n_sentences)) >= chunk_max_tokens:
                final_chunks.append("".join(current_n_sentences))
                current_n_sentences = []
        # Add the last chunk.
        if current_n_sentences:
            final_chunks.append("".join(current_n_sentences))

    for i, chunk in enumerate(final_chunks):
        try:
            while "TEMPLATE_CODE_CELL" in chunk and code_cells:
                code_cell = code_cells.pop(0)

                # Split the code cell into lines.
                code_cell_lines = code_cell.split("\n")

                # If the code cell can be inserted without exceeding the chunk size, do it.
                if (
                    get_tokens_number(chunk.replace("TEMPLATE_CODE_CELL", code_cell, 1))
                    <= chunk_max_tokens * 2
                ):
                    chunk = chunk.replace("TEMPLATE_CODE_CELL", code_cell, 1)
                # If not, insert as many lines as possible.
                else:
                    code_cell_lines = []
                    for line in code_cell_lines:
                        # If the next line can be inserted without exceeding the chunk size, do it.
                        if (
                            get_tokens_number(
                                chunk.replace(
                                    "TEMPLATE_CODE_CELL", "\n".join(code_cell_lines), 1
                                )
                            )
                            <= chunk_max_tokens * 2
                        ):  # chunk_size
                            code_cell_lines.append(line)
                        # If not, put the remaining lines back into code_cells and stop.
                        else:
                            break
                    chunk = chunk.replace(
                        "TEMPLATE_CODE_CELL", "\n".join(code_cell_lines)
                    )

            final_chunks[i] = chunk
            print(f"Tokens number of chunk {i}: {get_tokens_number(chunk)}")
        except IndexError:
            warnings.warn(
                "Code cells mismatch. The number of 'TEMPLATE_CODE_CELL' placeholders and actual code cells do not match."
            )

    return final_chunks
