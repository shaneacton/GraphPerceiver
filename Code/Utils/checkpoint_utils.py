import json
import os
import pickle
from os.path import join, exists

from Checkpoint import CHECKPOINT_FOLDER


def get_folder_path(run_name):
    path = join(CHECKPOINT_FOLDER, run_name)
    if not exists(path):
        os.mkdir(path)
    return path


def load_json_data(path):
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print("error loading json data at:", path)
        raise e
    return data


def save_json_data(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def save_binary_data(data, path):
    filehandler = open(path, 'wb')
    pickle.dump(data, filehandler)
    filehandler.close()


def load_binary_data(path):
    filehandler = open(path, 'rb')
    data = pickle.load(filehandler)
    filehandler.close()
    return data