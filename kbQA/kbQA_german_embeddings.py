from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore

from haystack.retriever.elasticsearch import EmbeddingRetriever
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.utils import convert_files_to_dicts
import pandas as pd
import requests
import logging
import subprocess
import time
import os


if __name__ == '__main__':
    # Start new server or connect to a running one. true and false respectively
    LAUNCH_ELASTICSEARCH = False
    # Determines whether the Elasticsearch Server has to be populated with data or not
    POPULATE_DOCUMENT_STORE = True

    if LAUNCH_ELASTICSEARCH:
        logging.info("Starting Elasticsearch ...")
        status = subprocess.run(
            ['docker run -d -p 9200:9200 -e "disScovery.type=single-node" elasticsearch:7.6.2'], shell=True
        )
        if status.returncode:
            raise Exception("Failed to launch Elasticsearch. If you want to connect to an existing Elasticsearch instance"
                            "then set LAUNCH_ELASTICSEARCH in the script to False.")
        time.sleep(15)

    # Init the DocumentStore
    #
    # * specify the name of our `text_field` in Elasticsearch that we want to return as an answer
    # * specify the name of our `embedding_field` in Elasticsearch where we'll store the embedding of our question and that is used later for calculating our similarity to the incoming user question
    # * set `excluded_meta_data=["question_emb"]` so that we don't return the huge embedding vectors in our search results

    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                                index="document",
                                                text_field="text",
                                                embedding_field="question_emb",
                                                embedding_dim=768,
                                                excluded_meta_data=["question_emb"])

    # Create a Retriever using embeddings
    # Instead of retrieving via Elasticsearch's plain BM25, we want to use vector similarity of the questions (user question vs. FAQ ones).
    # We can use the `EmbeddingRetriever` for this purpose and specify a model that we use for the embeddings.
    #
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model=os.getcwd()+"\\kbQA\\bert-german-model", gpu=True, model_format="transformers")

    if POPULATE_DOCUMENT_STORE:
        # set path to directory conating the text files
        doc_dir = os.getcwd() + "\\kbQA\\data\\lotr"
        # convert files to dicts containing documents that can be indexed to our datastore
        dicts = convert_files_to_dicts(
            dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

        df = pd.DataFrame.from_dict(dicts)
        # Get embeddings for our questions from the FAQs
        questions = list(df["text"].values)
        df["question_emb"] = retriever.create_embedding(texts=questions)

        # Convert Dataframe to list of dicts and index them in our DocumentStore
        docs_to_index = df.to_dict(orient="records")
        # You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
        # It must take a str as input, and return a str.

        # Now, let's write the docs to our DB.
        document_store.write_documents(docs_to_index)

    # reader = TransformersReader(
    #     model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)

    # # Init reader & and use Finder to get answer (same as in Tutorial 1)
    # finder = Finder(reader=reader, retriever=retriever)

    # prediction = finder.get_answers(
    #     question="Who is the father of Arya?", top_k_reader=3, top_k_retriever=5)

    # print_answers(prediction, details="all")
