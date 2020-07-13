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
from haystack.utils import print_answers

data_path = "./kbQA/data/MLQA_V1"
reader_model_name_full = "mrm8488/bert-multi-cased-finetuned-xquadv1"
reader_model_name = reader_model_name_full.split("/")[1]
retriever_model_name_full = "distiluse-base-multilingual-cased"
retriever_model_type = "sentence_transformers"
somajo = SoMaJo("de_CMC", split_camel_case=False, split_sentences=True)

POPULATE_DOCUMENT_STORE = False
# Set the value to true wehn running for the first time. POPULATE_DOCUMENT_STORE is
# set to true automatically when this is true
LAUNCH_ELASTICSEARCH = False
# Test run messuring times and quality of answers
TEST = True


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


def main():
    # fetch model files if not present. not hosted in git repo
    # model_exists = os.path.isfile(
    #     './kbQA/bert-multi-cased-finetuned-xquadv1/pytorch_model.bin')
    # if not model_exists:
    #     logging.info("Starting model download (about 700MB) ...")
    #     urllib.request.urlretrieve(
    #         "https://cdn.huggingface.co/mrm8488/bert-multi-cased-finetuned-xquadv1/pytorch_model.bin",
    #         "./kbQA/bert-multi-cased-finetuned-xquadv1/pytorch_model.bin")
    #     logging.info("model successfully downloaded")
    # start Elasticsearch
    if LAUNCH_ELASTICSEARCH:
        logging.info("Starting Elasticsearch ...")
        status = subprocess.call(
            'docker run -d -p 9200:9200 -e "discovery.type=single-node" --name "MLQA2" elasticsearch:7.6.2',
            shell=True
        )
        if status.returncode:
            raise Exception("Failed to launch Elasticsearch. If you want to "
                            "connect to an existing Elasticsearch instance"
                            "then set LAUNCH_ELASTICSEARCH in the script to False.")
        time.sleep(15)

    # 512 dimensions because that is what the sentnce transformer returns
    document_store = ElasticsearchDocumentStore(host="localhost", username="",
                                                password="", index="document",
                                                embedding_dim=512,
                                                embedding_field="embedding")

    # load docs in database
    if LAUNCH_ELASTICSEARCH or POPULATE_DOCUMENT_STORE:
        dicts = convert_files_to_dicts(
            dir_path=data_path, clean_func=clean_text, split_paragraphs=True)

        logging.info("files to dicts done.")
        # write dicts containing the texts to the database
        document_store.write_documents(dicts)
        logging.info("documents to store written.")

        retriever = EmbeddingRetriever(document_store=document_store,
                                       embedding_model=retriever_model_name_full,
                                       model_format=retriever_model_type,
                                       gpu=False)
        # generate embeddings for each text and add it to the databse entry
        document_store.update_embeddings(retriever)
        logging.info("embeddings to documents in store written.")

    retriever = EmbeddingRetriever(document_store=document_store,
                                   embedding_model=retriever_model_name_full,
                                   model_format=retriever_model_type,
                                   gpu=False)

    # reader wont be used in the retrieval because results take longer and the quality is worse
    # still has to be initialized
    # reader = TransformersReader(model="./kbQA/" + reader_model_name,
    #                             tokenizer="./kbQA/" + reader_model_name,
    #                             use_gpu=-1)
    finder = Finder(retriever=retriever, reader=None)

    if TEST:
        try:
            with open("./kbQA/Test.json", encoding="utf-8") as file:
                times = []
                results = []
                failed = []
                # each line has multiple paragraphs and embeddings, read file line
                # by line
                for line in enumerate(file):
                    # load the json string of the current line as a a python object
                    data = json.loads(line[1])
                    q = data["question"]
                    # fetch results from db
                    start_time = time.process_time()
                    candidate_docs = finder.retriever.retrieve(
                        query=q, filters=None, top_k=5)
                    end_time = time.process_time()
                    times.append(end_time-start_time)
                    answered = False
                    for doc in candidate_docs:
                        if data["answer"] in doc.text:
                            answered = True
                            results.append(True)
                            break
                    if not answered:
                        answers = []
                        for doc in candidate_docs:
                            answers.append(doc.text)
                        failed.append(
                            {"q": q, "correct": data["answer"], "a": answers})
                total = 0
                for zeit in times:
                    total = total + zeit
                logging.info("Average time per request: %f",
                             total / len(times))
                logging.info("Questions answered correctly: %d/%d (%f)",
                             len(results), len(times), len(results)/len(times))
                logging.debug("Failed questions:")
                for fail in failed:
                    logging.debug("Question: %s", fail["q"])
                    logging.debug("Correct Answer: %s", fail["correct"])
                    for answer in fail["a"]:
                        logging.debug(answer)

        except Exception as e:
            traceback.print_exc()
            logging.error(f"exception: {e}")
    else:
        # loop until Keyboard-Interrupt event ctrl+c or "!q" input
        while True:
            try:
                # Eread input from console input
                q = input("Enter:").strip()
                # input "!q" to stop execution
                if q == "!q":
                    exit(0)
                # fetch results from db
                candidate_docs = finder.retriever.retrieve(
                    query=q, filters=None, top_k=5)
                for doc in candidate_docs:
                    logging.info("doc id: %s", doc.id)
                    logging.info("doc meta name: %s", doc.meta["name"])
                    logging.info("doc text: %s", doc.text)
                    logging.info("doc query score: %s", doc.query_score)
                    logging.info("")
                # not used
                # prediction = finder.get_answers(
                #     question=q, top_k_retriever=10, top_k_reader=5)
                # print_answers(prediction, details="medium")
            except Exception as e:
                traceback.print_exc()
                logging.error(f"exception: {e}")


if __name__ == '__main__':
    main()
