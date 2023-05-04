import openai

from typing import List, Iterator
import pandas as pd
import numpy as np
import os
import wget
from ast import literal_eval

# Qdrant's client library for Python
import qdrant_client



# Setup
# I've set this to our new embeddings model, this can be changed to the embedding model of your choice
EMBEDDING_MODEL = "text-embedding-ada-002"

# Ignore unclosed SSL socket warnings - optional in case you get these errors
import warnings

warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)



# Load data
embeddings_url = 'https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip'

# The file is ~700 MB so this will take some time
wget.download(embeddings_url)

import zipfile
with zipfile.ZipFile("vector_database_wikipedia_articles_embedded.zip","r") as zip_ref:
    zip_ref.extractall("../data")

article_df = pd.read_csv('../data/vector_database_wikipedia_articles_embedded.csv')
article_df.head()

# Read vectors from strings back into a list
article_df['title_vector'] = article_df.title_vector.apply(literal_eval)
article_df['content_vector'] = article_df.content_vector.apply(literal_eval)

# Set vector_id to be a string
article_df['vector_id'] = article_df['vector_id'].apply(str)

article_df.info(show_counts=True)



# For the local deployment, we are going to use Docker, according to the Qdrant documentation: https://qdrant.tech/documentation/quick_start/. 
# Qdrant requires just a single container, but an example of the docker-compose.yaml file is available at ./qdrant/docker-compose.yaml in this repo.
# You can start Qdrant instance locally by navigating to this directory and running docker-compose up -d

qdrant = qdrant_client.QdrantClient(host='localhost', prefer_grpc=True)
qdrant.get_collections()



# Index data
# Qdrant stores data in collections where each object is described by at least one vector and may contain an additional metadata called payload. 
# Our collection will be called "Articles" and each object will be described by both "title" and "content" vectors.
# We'll be using an official "qdrant-client" package that has all the utility methods already built-in.
from qdrant_client.http import models as rest

vector_size = len(article_df['content_vector'][0])

qdrant.recreate_collection(
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
    }
)

qdrant.upsert(
    collection_name='Articles',
    points=[
        rest.PointStruct(
            id=k,
            vector={
                'title': v['title_vector'],
                'content': v['content_vector'],
            },
            payload=v.to_dict(),
        )
        for k, v in article_df.iterrows()
    ],
)

# Check the collection size to make sure all the points have been stored
qdrant.count(collection_name='Articles')



# Search Data
# Once the data is put into Qdrant we will start querying the collection for the closest vectors. 
# We may provide an additional parameter vector_name to switch from title to content based search.
def query_qdrant(query, collection_name, vector_name='title', top_k=20):

    # Creates embedding vector from user query
    embedded_query = openai.Embedding.create(
        input=query,
        model=EMBEDDING_MODEL,
    )['data'][0]['embedding']
    
    query_results = qdrant.search(
        collection_name=collection_name,
        query_vector=(
            vector_name, embedded_query
        ),
        limit=top_k,
    )
    
    return query_results

query_results = query_qdrant('modern art in Europe', 'Articles')
for i, article in enumerate(query_results):
    print(f'{i + 1}. {article.payload["title"]} (Score: {round(article.score, 3)})')

# This time we'll query using content vector
query_results = query_qdrant('Famous battles in Scottish history', 'Articles', 'content')
for i, article in enumerate(query_results):
    print(f'{i + 1}. {article.payload["title"]} (Score: {round(article.score, 3)})')