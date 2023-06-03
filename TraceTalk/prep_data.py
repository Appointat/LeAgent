import os
import warnings
from dotenv import load_dotenv 

import pandas as pd
import re
import requests

import openai

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest



def prep_data():
	"""
    This function prepares the book data by extracting content from Markdown files and their inline links.

	:param DataFrame: A pandas DataFrame of book data, with columns 'id', 'title', 'content',
    'link', 'inline_link_list', 'inline_title_list', and 'inline_content_list'.
	"""
	book_data = [] # Create an empty list to store book data.
	id = 0 # Set an initial value for the ID counter.

	input_directory = r'TraceTalk\vector-db-persist-directory\resources' # Set the defualt directory.
	
	for file in os.listdir(input_directory):
		if file.endswith('.txt'):
			with open(os.path.join(input_directory, file), 'r') as f:
				txt_content = f.read()
				# Link all Markdown files extracted from the text file.
				md_links = re.findall(r"'(https://[\w\d\-_/.]+\.md)',", txt_content)

			for link in md_links:
				md_file = link.rsplit('/', 1)[-1]
				md_title = md_file[:-3]  # Remove the .md suffix.

				# Get the contents of the .md file.
				converted_link = (
					link.replace("github.com/open-academy", "open-academy.github.io")
					.replace("tree/main", "_sources")
					.replace("open-machine-learning-jupyter-book/", "")
				)
				md_content_request = requests.get(converted_link)
				md_content = md_content_request.text if md_content_request.status_code == 200 else ''
				print(f"Processing {md_title}: {link}...")

				md_content_split = split_text_into_chunks(md_content, chunk_size=1000)  # Split the text into chunks.
				for text in md_content_split:
					if not text:
						continue
					id = id + 1

					# Get the inline contents of the .md file.
					inline_title_list = []
					inline_content_list = []

					inline_link_list = []
					inline_links_list = re.findall(r'\[([^\]]+?)\]\((https?://[^\)]+?)\)', text)

					for inline_title, inline_link in inline_links_list:
						if inline_link.endswith(('.exe', '.zip', '.rar')):
							continue
						try:
							inline_content_request = requests.get(inline_link)
							inline_content = inline_content_request.text if md_content_request.status_code == 200 else ''
							inline_title_list.append(inline_title)

							inline_content_list.append(inline_content)
							inline_link_list.append(inline_link)
						except requests.exceptions.RequestException as e:
							warnings.warn(f"Failed to fetch content from link: {inline_link}\n{e}")
							continue

					# Add the book data to the list.
					book_data.append({
						'id': id,
						'title': md_title,
						'title_vector': get_emmbedings(md_title),
						'content': text, # Further optimization can be done by splitting files to reduce text volume.
						'content_vector': get_emmbedings(text), # Further optimization can be done by splitting files to reduce text volume.
						'link': converted_link,
						'inline_title_list': inline_title_list,
						'inline_content_list': inline_content_list,
						'inline_link_list': inline_link_list,
					})

	# Convert the book_data list into a pandas DataFrame.
	book_data_df = pd.DataFrame(book_data)

	# Idex client.
	client = QdrantClient(path=r'TraceTalk\vector-db-persist-directory\Qdrant')
	
	# Upsert the data into the collection of the Qdrant database.
	vector_size = 1536
	client.recreate_collection(
		collection_name='Articles',
		vectors_config={
			'title': rest.VectorParams(
				distance=rest.Distance.COSINE,
				size=vector_size,
			),
			'content': rest.VectorParams(
				distance=rest.Distance.COSINE,
				size=vector_size,
			),
			#'inline_title_vector': rest.VectorParams(
			#    distance=rest.Distance.COSINE,
			#    size=vector_size,
			#),
			# 'inline_content_vector': rest.VectorParams(
			#    distance=rest.Distance.COSINE,
			#    size=vector_size,
			#),
		}
	)

	client.upsert(
        collection_name='Articles',
        points=[
            rest.PointStruct(
                id=row['id'],
                vector={
                    'title': row['title_vector'],
                    'content': row['content_vector'],
                    #'inline_title_vector': row['inline_title_list'],
                    #'inline_content_vector': row['inline_content_list'],
                },
                payload=row.to_dict(),
            )
            for _, row in book_data_df.iterrows()
        ],
    )
	
	return book_data_df



def get_emmbedings(text):
	load_dotenv()
	openai.api_key = os.getenv('OPENAI_API_KEY')
	embedded_query = openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002",
    )['data'][0]['embedding']
	
	return embedded_query # a vector of numbers



def split_text_into_chunks(text, delimiter="###", chunk_size=500):
	pattern = r"---.*?---"
	text = re.sub(pattern, "", text, flags=re.DOTALL)
	chunks = re.split('({})'.format(delimiter), text)
	chunks = ['{}{}'.format(delimiter, chunk) if i % 2 else chunk for i, chunk in enumerate(chunks)]

	final_chunks = []
	for chunk in chunks:
		words = re.split(r'(\s)', chunk)  # Split a string using regex, preserving spaces and newlines.
		current_chunk_words = []
		current_word_count = 0
		for word in words:
            # If word is a space or a newline, don't count the word; otherwise, add 1 to the word count.
			if word.strip() != '':
				current_word_count += 1
			current_chunk_words.append(word)

            # When the number of words reaches chunk_size, add a new chunk.
			if current_word_count >= chunk_size:
				final_chunks.append(''.join(current_chunk_words))
				current_chunk_words = []
				current_word_count = 0
		# Add the last chunk.
		if current_chunk_words:
			final_chunks.append(''.join(current_chunk_words))
	
	return final_chunks



def split_text_into_chunks_2(text, chunk_size=500):
	if len(text) <= chunk_size:
		return [text]
	
	words = text.split()
	chunks = []
	current_chunk = []
	
	for word in words:
		current_chunk.append(word)
		
		if len(current_chunk) == chunk_size:
			chunks.append(' '.join(current_chunk))
			current_chunk = []
	if current_chunk:
		chunks.append(' '.join(current_chunk))
		
	return chunks