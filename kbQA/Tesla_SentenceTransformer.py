# -*- coding: utf-8 -*-
import os
import traceback
from typing import List

from somajo.somajo import SoMaJo

from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.utils import convert_files_to_dicts
from haystack.reader.transformers import TransformersReader
from haystack.retriever.dense import EmbeddingRetriever
from haystack.utils import print_answers

data_path = "./kbQA/data/tesla"
reader_model_name_full = "mrm8488/bert-multi-cased-finetuned-xquadv1"
reader_model_name = reader_model_name_full.split("/")[1]
retriever_model_name_full = "distiluse-base-multilingual-cased"
retriever_model_type = "sentence_transformers"
somajo = SoMaJo("de_CMC", split_camel_case=False, split_sentences=True)

POPULATE_DOCUMENT_STORE = False


def sentence_segmentation(text: str) -> List[str]:
    if not text.strip():
        return []

    tokenized_sentences = somajo.tokenize_text([text])

    sents = []
    for token_sent in tokenized_sentences:
        sent = []
        for token in token_sent:
            word = token.text

            if token.original_spelling:
                word = token.original_spelling
            if token.space_after:
                word = word + " "
            sent.append(word)
        sent = "".join(word for word in sent)
        sent = sent.strip()
        if not sent.endswith("\n"):
            sent = sent + "\n"
        sents.append(sent)
    return sents


def clean_text(path_name: str, text: str) -> str:
    topic = path_name.rstrip(".txt").replace("_", " ")

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    lines = text.split("\n")
    out = []
    for line in lines:
        sent_segs = sentence_segmentation(line)
        if not sent_segs:
            continue
        sent_segs = [f"{topic}: {sent}" for sent in sent_segs]
        out.extend(sent_segs)

    text = "\n".join(out)
    return text


def main():
    # 512 weil sentence transformer diese Dimension zurückgibt.
    document_store = ElasticsearchDocumentStore(host="localhost", username="", password="", index="document",
                                                embedding_dim=512, embedding_field="embedding")

    if POPULATE_DOCUMENT_STORE:
        dicts = convert_files_to_dicts(dir_path=data_path, clean_func=clean_text, split_paragraphs=True)
        print("files to dicts done.")
        print("first 10 dicts:", dicts[0:10])
        document_store.write_documents(dicts)
        print("documents to store written.")

        retriever = EmbeddingRetriever(document_store=document_store,
                                       embedding_model=retriever_model_name_full,
                                       model_format=retriever_model_type,
                                       gpu=False)
        document_store.update_embeddings(retriever)
        print("embeddings to documents in store written.")
        exit(0)

    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model=retriever_model_name_full,
                                   model_format=retriever_model_type,
                                   gpu=False)

    reader = TransformersReader(model="./kbQA/" + reader_model_name,
                                tokenizer="./kbQA/" + reader_model_name,
                                use_gpu=-1)

    finder = Finder(retriever=retriever, reader=reader)

    while True:
        try:
            q = input("Enter:").strip()
            if q == "!q":
                exit(0)

            candidate_docs = finder.retriever.retrieve(query=q, filters=None, top_k=10)
            for doc in candidate_docs:
                print("doc id:", doc.id)
                print("doc meta name:", doc.meta["name"])
                print("doc text:", doc.text)
                print("doc query score:", doc.query_score)
                print("")

            prediction = finder.get_answers(question=q, top_k_retriever=10, top_k_reader=5)
            print_answers(prediction, details="medium")
        except Exception as e:
            traceback.print_exc()
            print(f"exception: {e}")

if __name__ == '__main__':
    main()
