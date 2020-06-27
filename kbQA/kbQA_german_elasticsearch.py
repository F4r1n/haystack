import logging
import subprocess
import time
import os

from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.cleaning import clean_wiki_text
from haystack.indexing.utils import convert_files_to_dicts
from haystack.reader.farm import FARMReader
from haystack.reader.transformers import TransformersReader
from haystack.utils import print_answers
from haystack.retriever.elasticsearch import ElasticsearchRetriever

# windows workaround to prevent endless recursion
if __name__ == '__main__':
    # Start new server or connect to a running one. true and false respectively
    LAUNCH_ELASTICSEARCH = False
    # Determines whether the Elasticsearch Server has to be populated with data or not
    POPULATE_DOCUMENT_STORE = False

    # Start an Elasticsearch server
    if LAUNCH_ELASTICSEARCH:
        logging.info("Starting Elasticsearch ...")
        status = subprocess.run(
            ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2'], shell=True
        )
        if status.returncode:
            raise Exception("Failed to launch Elasticsearch. If you want to connect to an existing Elasticsearch instance"
                            "then set LAUNCH_ELASTICSEARCH in the script to False.")
        time.sleep(15)

    # Connect to Elasticsearch
    document_store = ElasticsearchDocumentStore(
        host="localhost", username="", password="", index="document")

    # ## Cleaning & indexing documents

    # Initialize Elasticsearch with docs
    if POPULATE_DOCUMENT_STORE:
        # set path to directory containing the text files
        doc_dir = os.getcwd() + "\\kbQA\\data\\tesla"
        # convert files to dicts containing documents that can be indexed to our datastore
        dicts = convert_files_to_dicts(
            dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

        # write the docs to the elasticsearch database
        document_store.write_documents(dicts)

    # ## Initalize Retriever, Reader,  & Finder
    #
    # ### Retriever
    #
    # Retrievers help narrowing down the scope for the Reader to smaller units of text where a given question
    # could be answered.
    #
    # They use some simple but fast algorithm.
    # **Here:** We use Elasticsearch's default BM25 algorithm

    retriever = ElasticsearchRetriever(document_store=document_store)

    # ### Reader
    #
    # A Reader scans the texts returned by retrievers in detail and extracts the k best answers. They are based
    # on powerful, but slower deep learning models.

    reader = TransformersReader(
        model="dbmdz/bert-base-german-uncased", tokenizer="dbmdz/bert-base-german-uncased", use_gpu=-1)

    # ### Finder
    #
    # The Finder sticks together reader and retriever in a pipeline to answer our actual questions.

    finder = Finder(reader, retriever)

    # You can configure how many candidates the reader and retriever shall return
    # top_k_retriever: number of documents, top_k_reader: number of answers to be retunred
    prediction = finder.get_answers(
        question="fährt das auto wenn der stecker steckt?", top_k_retriever=5, top_k_reader=3)
    print_answers(prediction, details="all")
