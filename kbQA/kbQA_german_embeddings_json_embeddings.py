import os
import time
import logging
import subprocess
import pandas as pd
import json
import numpy as np
from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.utils import convert_files_to_dicts
from haystack.retriever.dense import EmbeddingRetriever
from haystack.utils import print_answers

# custom cleaning function to match the Sense.AI.Tion BERT-Model


def preprocessQuestion(text) -> str:
    text = text.lower()
    text = text.replace("ß", "ss")
    text = text.replace("?", ".")
    return text


# windows workaround to prevent endless recursion
if __name__ == '__main__':
    # Start new server or connect to a running one. true and false respectively
    LAUNCH_ELASTICSEARCH = False
    # Determines whether the Elasticsearch Server has to be populated with data
    # or not
    POPULATE_DOCUMENT_STORE = False
    # Start an Elasticsearch server
    if LAUNCH_ELASTICSEARCH:
        logging.info("Starting Elasticsearch ...")
        status = subprocess.run(
            ['docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:7.6.2'],
            shell=True
        )
        if status.returncode:
            raise Exception("Failed to launch Elasticsearch. If you want to "
                            "connect to an existing Elasticsearch instance"
                            "then set LAUNCH_ELASTICSEARCH in the script to False.")
        time.sleep(15)

    # Connect to Elasticsearch including embedding definition
    document_store = ElasticsearchDocumentStore(host="localhost",
                                                username="",
                                                password="",
                                                index="document",
                                                text_field="text",
                                                embedding_field="question_emb",
                                                embedding_dim="768",
                                                excluded_meta_data=["question_emb"])
    # ## Initalize Retriever, Reader,  & Finder
    # ### Retriever
    # gpu= True to speed up processing
    # BERT-Model is trained to use the second extraction layer
    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model=os.getcwd() +
        "/kbQA/bert-german-model",
        gpu=True, model_format="transformers",
        emb_extraction_layer=-2)
    if POPULATE_DOCUMENT_STORE:
        # set path to directory containing the embeddings
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

    questions = [
        "worauf sollte man auf Fähren achten?",
        "wird die verkehrschilderkennung für alle kommen?",
        "was beinhaltet der Autopilot?",
        "wie viel verbaucht das Model 3?",
        "fährt das auto wenn der stecker steckt?",
        "Welche dimension haben die kleinen Sommerreifen?",
        "wie viel zoll haben die Sommerreifen?",
        "Werden UV-Strahlen beim Tesla geblockt?",

        "Ich habe bei Tesla 500€ pro Rad bezahlt.",
        "Tempomat Geschwindigkeit ändern.",
        "die batterie sollte mindestens 50% haben."
    ]
    # auch hier wieder: Kleinschreibung zwingend notwendig!
    # question = question.lower()
    times = []

    # Wir können aktuell keinen Reader verwenden, da diese scheinbar QA fine tuning voraussetzen
    # Der Retriever holt anhand der embeddings die besten Treffer ran.
    # get_answers() ohne reader nicht verwendbar
    finder = Finder(reader=None, retriever=retriever)

    for question in questions:
        # Änderung: process_time() zählt nur die tatsächliche CPU time
        start_time = time.process_time()
        # Änderung: Ergebnis von preprocessingQuestion ging in "q" aber wurde nicht weiter verwendet
        question = preprocessQuestion(question)
        print(f"QUESTION: {question}")
        prediction = finder.get_answers_via_similar_questions(
            question, top_k_retriever=5)
        end_time = time.process_time()
        times.append(end_time - start_time)
        print_answers(prediction, details="minimal")

    total = 0
    for zeit in times:
        total = total + zeit
    print(total / len(times))
