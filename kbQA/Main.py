import os

import pandas as pd

from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.utils import convert_files_to_dicts
from haystack.retriever.elasticsearch import EmbeddingRetriever
from haystack.utils import print_answers


def clean_text(text: str) -> str:
    # habe ich einfach mal aus clean_wiki_text übernommen
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    # super wichtig! In tokenizer_config.json ist do_lower_case = false, weil andernfalls Umlaute, etc. völlig
    # falsch gehandelt werden.
    text = text.lower()
    lines = text.split("\n")
    # \n\n scheint irgendwie benötigt zu werden, wegen split_paragraphs=True weiter unten.
    text = "\n\n".join(lines)
    return text


def main():
    POPULATE_DOCUMENT_STORE = False

    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="",
                                                index="document",
                                                text_field="text",
                                                embedding_field="question_emb",
                                                embedding_dim="768",  # war bei dir int, aber string wird verlangt
                                                excluded_meta_data=["question_emb"])

    retriever = EmbeddingRetriever(
        document_store=document_store, embedding_model=os.getcwd() + "/kbQA/bert-german-model",
        gpu=True, model_format="transformers")

    if POPULATE_DOCUMENT_STORE:
        doc_dir = os.getcwd() + "/kbQA/data/test"
        dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_text, split_paragraphs=True)
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

        docs_to_index = df.to_dict(orient="records")
        document_store.write_documents(docs_to_index)


    question = "Wie viele haben Angst um ihren Job?"
    question = question.lower()  # auch hier wieder: Kleinschreibung zwingend notwendig!

    # Wir können aktuell keinen Reader verwenden, da diese scheinbar QA fine tuning voraussetzen
    # Der Retriever holt anhand der embeddings die besten Treffer ran.
    # get_answers() ohne reader nicht verwendbar
    finder = Finder(reader=None, retriever=retriever)
    prediction = finder.get_answers_via_similar_questions(question, top_k_retriever=2)
    print_answers(prediction, details="all")


if __name__ == '__main__':
    main()
