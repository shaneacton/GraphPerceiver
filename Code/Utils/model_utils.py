from os.path import join, exists

import torch

from Code.Training.performance import Performance
from Code.Utils.checkpoint_utils import get_folder_path
from Config.options import device, model_conf


def get_model(run_name):
    if exists(checkpoint_path(run_name)):
        return load_checkpoint(run_name)
    print("starting new model")
    from Code.Model.mhqa_model import MHQA

    mhqa = MHQA().to(device)
    optim = torch.optim.SGD([c for c in mhqa.parameters() if c.requires_grad], lr=model_conf().lr)
    return mhqa, optim, Performance()


def save_checkpoint(run_name, model, optim, performance):
    save_model(run_name, model, optim)


def load_checkpoint(run_name):
    print("loading model checkpoint")
    model, optim = load_model(run_name)
    return model, optim, Performance()


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(run_name, model, optimizer):
    model_save_data = {"model": model, "optimizer_state_dict": optimizer.state_dict()}
    torch.save(model_save_data, checkpoint_path(run_name))


def load_model(run_name):
    checkpoint = torch.load(checkpoint_path(run_name), map_location=device)
    model = checkpoint["model"].to(device)
    optim = torch.optim.SGD([c for c in model.parameters() if c.requires_grad], lr=model_conf().lr)
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optim


def checkpoint_path(run_name):
    path = join(get_folder_path(run_name), "model_checkpoint")
    return path