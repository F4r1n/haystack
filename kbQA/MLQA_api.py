# -*- coding: utf-8 -*-
import os
import traceback
import logging
import subprocess
import urllib
import json
import time
from typing import List

from somajo.somajo import SoMaJo

from haystack import Finder
from haystack.database.elasticsearch import ElasticsearchDocumentStore
from haystack.indexing.utils import convert_files_to_dicts
from haystack.reader.transformers import TransformersReader
from haystack.retriever.dense import EmbeddingRetriever

from flask import Flask, json, request

api = Flask(__name__)


@api.route('/answer', methods=['GET'])
def get_answer():
    question = request.args.get('q')
    candidate_docs = finder.retriever.retrieve(
        query=question, filters=None, top_k=5)
    answers = []
    for doc in candidate_docs:
        answers.append(
            {"id": doc.id, "metaName": doc.meta["name"], "text": doc.text, "score": doc.query_score})
    return json.dumps(answers)


data_path = "./kbQA/data/MLQA_V1"
reader_model_name_full = "mrm8488/bert-multi-cased-finetuned-xquadv1"
reader_model_name = reader_model_name_full.split("/")[1]
retriever_model_name_full = "distiluse-base-multilingual-cased"
retriever_model_type = "sentence_transformers"
somajo = SoMaJo("de_CMC", split_camel_case=False, split_sentences=True)


def sentence_segmentation(text: str) -> List[str]:
    # Aufteilen der Texte in Sätze
    if not text.strip():
        return []
    tokenized_sentences = somajo.tokenize_text([text])
    sents = []
    # Somajo generetates tokens. We need sentences instead. Thus we concatenate
    # the tokens back to sentences and use somajo as a sentence splitter
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
    # unifies the text
    # clean the name of the file to add it to the sentence later on.
    topic = path_name.rstrip(".txt").replace("_", " ")
    # remove double newlines
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    lines = text.split("\n")
    out = []
    for line in lines:
        # split  line in sentences
        sent_segs = sentence_segmentation(line)
        if not sent_segs:
            continue
        # add the path name to the sentence. This is used to preserve the context
        # of the sentence and bridge anaphora resolution
        sent_segs = [f"{topic}: {sent}" for sent in sent_segs]
        out.extend(sent_segs)

    text = "\n".join(out)
    return text


if __name__ == '__main__':
    while True:
        try:
            # 512 dimensions because that is what the sentnce transformer returns
            document_store = ElasticsearchDocumentStore(host="elasticsearch", username="",
                                                        password="", index="document",
                                                        embedding_dim=512,
                                                        embedding_field="embedding")
            break
        except:
            time.sleep(15)

    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model=retriever_model_name_full,
                                   model_format=retriever_model_type,
                                   gpu=False)
    if document_store.get_document_count() < 1:
        dicts = convert_files_to_dicts(
            dir_path=data_path, clean_func=clean_text, split_paragraphs=True)

        logging.info("files to dicts done.")
        # write dicts containing the texts to the database
        document_store.write_documents(dicts)
        logging.info("documents to store written.")
        # generate embeddings for each text and add it to the databse entry
        document_store.update_embeddings(retriever)
        logging.info("embeddings to documents in store written.")

    finder = Finder(retriever=retriever, reader=None)

    api.run(host='0.0.0.0', port=8000, debug=True)
