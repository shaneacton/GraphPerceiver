from __future__ import annotations

import pickle
from os.path import exists, join
from typing import List

import nlp
from tqdm import tqdm

from Data import DATA_FOLDER
from Config.options import use_special_ents, use_detected_ents, embedder_name
from Code.wikipoint import Wikipoint


def load_unprocessed_dataset(dataset_name, version_name, split):
    """loads the original, unprocessed version of the given dataset"""
    remaining_tries = 100
    dataset = None
    e = None
    while remaining_tries > 0:
        """load dataset from online"""
        try:
            dataset = nlp.load_dataset(path=dataset_name, split=split, name=version_name)
            break  # loaded successfully
        except Exception as e:
            remaining_tries -= 1  # retry
            if remaining_tries == 0:
                print("failed to load datasets though network")
                raise e

    return dataset


def get_wikipoints(tokenizer, split=nlp.Split.TRAIN) -> List[Wikipoint]:
    global has_loaded

    file_name = "wikihop_" + split._name
    file_name += ("_det" if use_detected_ents else "") + ("_spec" if use_special_ents else "")
    file_name += "_" + embedder_name.replace("/", "-")
    file_name += ".data"
    data_path = join(DATA_FOLDER, file_name)

    if exists(data_path):  # has been processed before
        print("loading preprocessed wikihop", file_name)
        filehandler = open(data_path, 'rb')
        processed_examples = pickle.load(filehandler)
        print("loaded", len(processed_examples), "graphs")
        filehandler.close()
        return processed_examples

    print("loading wikihop unprocessed")
    data = list(load_unprocessed_dataset("qangaroo", "wikihop", split))
    print("num examples:", len(data))

    print("processing wikihop", file_name)
    print("tokenising text for bert")
    processed_examples = [Wikipoint(ex, tokeniser=tokenizer) for ex in tqdm(data)]
    print("creating graphs")

    print("saving", len(processed_examples), "graphs")
    save_binary_data(processed_examples, join(DATA_FOLDER, file_name))
    return processed_examples


def save_binary_data(data, path):
    filehandler = open(path, 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()