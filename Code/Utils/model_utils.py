from os.path import join, exists

import torch
from munch import Munch

from Code.Training.performance import Performance
from Code.Utils import wandb_utils
from Code.Utils.checkpoint_utils import get_folder_path, save_json_data, load_json_data, save_binary_data, \
    load_binary_data
from Config.options import device, model_conf, set_model_conf, use_custom_output_module, use_learned_latents


def get_model(run_name):
    if exists(model_path(run_name)):
        return load_checkpoint(run_name)
    print("starting new model")

    mhqa, optim = get_new_model()
    return mhqa, optim, Performance()


def load_checkpoint(run_name):
    print("loading model checkpoint")
    model, optim = load_model(run_name)
    performance = load_binary_data(model_performance_path(run_name))
    return model, optim, performance


def get_new_model():
    if use_learned_latents:
        from Code.Model.mhqa.pure_perceiver_mhqa import PerceiverIOMHQA
        mhqa = PerceiverIOMHQA()
    else:
        if use_custom_output_module:
            from Code.Model.mhqa.custom_output_graph_perceiver import CustomOutputMHQA
            mhqa = CustomOutputMHQA()
        else:
            from Code.Model.mhqa.graph_perceiver_mhqa import GraphPerceiverMHQA
            mhqa = GraphPerceiverMHQA()
    mhqa = mhqa.to(device)
    optim = torch.optim.SGD([c for c in mhqa.parameters() if c.requires_grad], lr=model_conf().lr)
    if model_conf().use_wandb:
        wandb_utils.new_run(model_conf().model_name)
    return mhqa, optim


def load_model(run_name):
    checkpoint = torch.load(model_path(run_name), map_location=device)
    model = checkpoint["model"].to(device)
    conf = load_json_data(model_conf_path(run_name))
    conf = Munch(conf)
    set_model_conf(conf)

    wandb_id = conf["wandb_id"]
    wandb_utils.continue_run(wandb_id, model_conf().model_name)

    optim = torch.optim.SGD([c for c in model.parameters() if c.requires_grad], lr=model_conf().lr)
    optim.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optim


def save_checkpoint(run_name, model, optim, performance):
    save_model(run_name, model, optim)
    save_json_data(model_conf(), model_conf_path(run_name))
    save_binary_data(performance, model_performance_path(run_name))


def num_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(run_name, model, optimizer):
    model_save_data = {"model": model, "optimizer_state_dict": optimizer.state_dict()}
    torch.save(model_save_data, model_path(run_name))


def model_performance_path(run_name):
    path = join(get_folder_path(run_name), "performance.data")
    return path


def model_conf_path(run_name):
    path = join(get_folder_path(run_name), "model_conf.json")
    return path


def model_path(run_name):
    path = join(get_folder_path(run_name), "model_checkpoint")
    return path