import json
from os.path import join
from typing import Dict

import torch
from munch import Munch

from Config import CONFIG_FOLDER

"""EMBEDDING options"""
# embedder_name = "google/bigbird-roberta-base"  # "prajjwal1/bert-" for configurable sized bert
embedder_name = "prajjwal1/bert-"  # "prajjwal1/bert-" for configurable sized bert

bert_fine_tune_layers = ["embeddings", "pooler", "LayerNorm", ""]  # "" for all
bert_size = "small"  # tiny | mini | small | medium


"""GRAPH ENCODING options"""
use_detected_ents = False
use_special_ents = True
use_span_embeddings = False  # 34 when on 37 off

"""TRAINING options"""
num_epochs = 10
weight_decay = 0.00001
print_loss_every = 500
max_examples = -1  # -1 for off
device = torch.device("cuda:0")

_model_conf = Munch(json.load(open(join(CONFIG_FOLDER, "model_params.json"))))
def model_conf() -> Dict:
    return _model_conf


def set_model_conf(conf) -> Dict:
    global _model_conf
    _model_conf = conf


print("model conf:", model_conf())