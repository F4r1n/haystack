from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.utils import convert_files_to_dicts
from haystack.retriever.elasticsearch import EmbeddingRetriever
from haystack.utils import print_answers
import os
import pandas as pd
import time
import subprocess
from flask import Flask, json, request
import logging

api = Flask(__name__)


@api.route('/answer', methods=['GET'])
def get_answer():
    question = request.args.get('q')
    question = preprocessQuestion(question)
    prediction = finder.get_answers_via_similar_questions(
        question, top_k_retriever=5)
    return json.dumps(prediction)


@api.route('/fillDB', methods=['POST'])
def fillDatabase():
    doc_dir = os.getcwd() + "/kbQA/data/tesla_embs.json"
    # initialize dataframe with column names
    df = pd.DataFrame(
        columns=['name', 'text', 'question_emb', 'question'],)
    # open the file
    with open(doc_dir, encoding="utf-8") as file:
        # initialize question indexing
        i = 0
        # each line has multiple paragraphs and embeddings, read file line
        # by line
        for cnt, line in enumerate(file):
            # load the json string of the current line as a apython object
            data = json.loads(line)
            # add an entry in the dataframe for each paragraph in the current
            # line
            for j in range(len(data["paragraph"])):
                df = df.append(
                    {"question_emb": data["embeddings"][j],
                        "name": data["file"],
                        "text": data["paragraph"][j],
                        "question": i},
                    ignore_index=True)
                i = i + 1
                logging.info(f"lines read: {i}")

        # convert the dataframe to a dict
        docs_to_index = df.to_dict(orient="records")
        logging.info("df.to_dict done")
        # write documents to elasticsearch storage
        document_store.write_documents(docs_to_index)
        logging.info("docs written to document store")


def preprocessQuestion(text) -> str:
    text = text.lower()
    text = text.replace("ß", "ss")
    text = text.replace("?", ".")
    return text


if __name__ == '__main__':
    # Connect to Elasticsearch
    document_store = ElasticsearchDocumentStore(host="localhost",
                                                username="",
                                                password="",
                                                index="document",
                                                text_field="text",
                                                embedding_field="question_emb",
                                                embedding_dim="768",
                                                excluded_meta_data=["question_emb"])

    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model=os.getcwd() +
        "/kbQA/bert-german-model",
        gpu=True, model_format="transformers",
        emb_extraction_layer=-2)
    finder = Finder(reader=None, retriever=retriever)

    api.run(host='0.0.0.0', debug=True)
