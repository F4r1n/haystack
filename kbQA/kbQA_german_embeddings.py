import os

import pandas as pd

from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.utils import convert_files_to_dicts
from haystack.retriever.dense import EmbeddingRetriever
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


def main():
    POPULATE_DOCUMENT_STORE = True

    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                                index="document",
                                                text_field="text",
                                                embedding_field="question_emb",
                                                embedding_dim="768",
                                                excluded_meta_data=["question_emb"])

    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model=os.getcwd() +
        "\\kbQA\\bert-german-model",
        gpu=True, model_format="transformers")

    if POPULATE_DOCUMENT_STORE:
        doc_dir = os.getcwd() + "\\kbQA\\data\\Skripte\\Securplus\\txt"
        dicts = convert_files_to_dicts(
            dir_path=doc_dir, clean_func=clean_text, split_paragraphs=True)

        with open("Output.txt", "w") as text_file:
            text = ""
            for doc in dicts:
                text = text + "\n" + doc["text"]
            text_file.write(text)
        df = pd.DataFrame.from_dict(dicts)

        # Hier muss man aufpassen! Wir erzeugen an dieser Stelle keine embeddings für die questions, sondern für
        # für die Texte, d.h. die Antworten. Daher sind die Namen der Variablen etwas verwirrend gewählt.
        # dummy_questions ist einfach nur eine steigende Zahl beginnend bei eins. Wird benötigt, da sonst Exceptions
        # bei der Suche geschmissen werden.
        # Im Tutorial scheint von einem FAQ ausgegangen zu sein, bei dem Frage und Antwort
        # definiert sind und somit embeddings für die vordefinierte Frage erzeugt werden können und eigentlich nur
        # auf diese basierend, die k-besten Kandidaten zurückgegeben werden. Wir dagegen erzeugen embeddings für
        # jeden einzelnen Text.
        # todo: Da wir für jeden Text embeddings erzeugen müssen wir eventuell eine Sentence Segmentation durchführen,
        #       denn je länger die Texte werden, desto ungenauer werden auch die embeddings. Pro Satz embedding sind
        #       deutlich exakter.
        questions = list(df["text"].values)
        df["question_emb"] = retriever.create_embedding(texts=questions)
        dummy_questions = [f"{no}" for no, x in enumerate(questions, start=1)]
        df["question"] = dummy_questions
        print(df.head())

        docs_to_index = df.to_dict(orient="records")
        document_store.write_documents(docs_to_index)

    # question = "Wie viele haben Angst um ihren Job?"
    question = "welche leistungen sind ausgeschlossen?"
    # auch hier wieder: Kleinschreibung zwingend notwendig!
    question = question.lower()

    # Wir können aktuell keinen Reader verwenden, da diese scheinbar QA fine tuning voraussetzen
    # Der Retriever holt anhand der embeddings die besten Treffer ran.
    # get_answers() ohne reader nicht verwendbar
    finder = Finder(reader=None, retriever=retriever)
    prediction = finder.get_answers_via_similar_questions(
        question, top_k_retriever=5)
    print_answers(prediction, details="all")

    # Idee:
    # doc -> langer Text -> Paragraph -> embedding == ungenau
    # doc -> langer text -> Satz 1 Satz 2 Satz 3 -> embedding 1 embedding 2 embedding 3 == deutlich genauer


if __name__ == '__main__':
    main()
