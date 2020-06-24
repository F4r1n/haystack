import os
import time
import pandas as pd
import json
import numpy as np
from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.utils import convert_files_to_dicts
from haystack.retriever.elasticsearch import EmbeddingRetriever
from haystack.utils import print_answers


def clean_text(text: str) -> str:
    # wie in der Bibliothek
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    # In tokenizer_config.json ist do_lower_case = false, weil andernfalls Umlaute, etc. völlig
    # falsch gehandelt werden.
    text = text.lower()
    lines = text.split("\n")
    # \n\n wird für split_paragraphs=True benötigt
    text = "\n\n".join(lines)
    return text


def preprocessQuestion(text) -> str:
    text = text.lower()
    text = text.replace("ß", "ss")
    text = text.replace("-", " ")
    return text


def main():
    POPULATE_DOCUMENT_STORE = False

    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                                index="document",
                                                text_field="text",
                                                embedding_field="question_emb",
                                                embedding_dim="768",
                                                excluded_meta_data=["question_emb"])

    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model=os.getcwd() +
        "\\kbQA\\bert-german-model",
        gpu=True, model_format="transformers",
        emb_extraction_layer=-2)

    if POPULATE_DOCUMENT_STORE:
        doc_dir = os.getcwd() + "\\kbQA\\data\\tesla_embs.json"

        df = pd.DataFrame(
            columns=['name', 'text', 'question_emb', 'question'], )
        with open(doc_dir, encoding="utf8") as file:
            i = 0
            for cnt, line in enumerate(file):
                data = json.loads(line)
                for j in range(len(data["paragraph"])):
                    df = df.append(
                        {"question_emb": data["embeddings"][j], "name": data["file"], "text": data["paragraph"][j], "question": i}, ignore_index=True)
                    i = i + 1
        print(df.head())

        docs_to_index = df.to_dict(orient="records")
        document_store.write_documents(docs_to_index)

    # question = "Wie viele haben Angst um ihren Job?"
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
        start_time = time.time()
        q = preprocessQuestion(question)
        prediction = finder.get_answers_via_similar_questions(
            question, top_k_retriever=5)
        times.append(time.time() - start_time)
        print_answers(prediction, details="minimal")

    total = 0
    for zeit in times:
        total = total + zeit
    print(total / len(times))
    # Idee:
    # doc -> langer Text -> Paragraph -> embedding == ungenau
    # doc -> langer text -> Satz 1 Satz 2 Satz 3 -> embedding 1 embedding 2 embedding 3 == deutlich genauer


if __name__ == '__main__':
    main()
