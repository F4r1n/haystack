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
        # set path to directory conating the text files
        doc_dir = os.getcwd() + "\\kbQA\\data\\article_txt_got"
        # convert files to dicts containing documents that can be indexed to our datastore
        dicts = convert_files_to_dicts(
            dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)
        # You can optionally supply a cleaning function that is applied to each doc (e.g. to remove footers)
        # It must take a str as input, and return a str.

        # Now, let's write the docs to our DB.
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
    # **Alternatives:**
    # - Customize the `ElasticsearchRetriever`with custom queries (e.g. boosting) and filters
    # - Use `EmbeddingRetriever` to find candidate documents based on the similarity of
    #   embeddings (e.g. created via Sentence-BERT)
    # - Use `TfidfRetriever` in combination with a SQL or InMemory Document store for simple prototyping and debugging

    # retriever = ElasticsearchRetriever(document_store=document_store)

    # Alternative: An in-memory TfidfRetriever based on Pandas dataframes for building quick-prototypes
    # with SQLite document store.

    from haystack.retriever.tfidf import TfidfRetriever
    retriever = TfidfRetriever(document_store=document_store)

    # ### Reader
    #
    # A Reader scans the texts returned by retrievers in detail and extracts the k best answers. They are based
    # on powerful, but slower deep learning models.
    #
    # Haystack currently supports Readers based on the frameworks FARM and Transformers.
    # With both you can either load a local model or one from Hugging Face's model hub (https://huggingface.co/models).
    # **Here:** a medium sized RoBERTa QA model using a Reader based on
    #           FARM (https://huggingface.co/deepset/roberta-base-squad2)
    # **Alternatives (Reader):** TransformersReader (leveraging the `pipeline` of the Transformers package)
    # **Alternatives (Models):** e.g. "distilbert-base-uncased-distilled-squad" (fast) or
    #                            "deepset/bert-large-uncased-whole-word-masking-squad2" (good accuracy)
    # **Hint:** You can adjust the model to return "no answer possible" with the no_ans_boost. Higher values mean
    #           the model prefers "no answer possible"

    # #### TransformersReader

    # Alternative:
    reader = TransformersReader(
        model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased", use_gpu=-1)

    # ### Finder
    #
    # The Finder sticks together reader and retriever in a pipeline to answer our actual questions.

    finder = Finder(reader, retriever)

    # You can configure how many candidates the reader and retriever shall return
    # The higher top_k_retriever, the better (but also the slower) your answers.
    prediction = finder.get_answers(
        question="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=3)

    print_answers(prediction, details="minimal")
