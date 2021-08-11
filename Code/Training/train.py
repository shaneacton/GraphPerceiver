import torch
from numpy import mean
from tqdm import tqdm

from Config.options import model_conf, max_examples, print_loss_every
from Code.Model.bert_embedder import TooManyTokens
from Code.Model.mhqa_model import MHQA
from Code.Training.eval import evaluate
from Code.Utils.dataset_utils import get_wikipoints
from Code.Utils.model_utils import num_params


def train_and_eval(mhqa: MHQA, optim=None):
    """trains the model for 1 epoch"""
    if optim is None:
        optim = torch.optim.SGD([c for c in mhqa.parameters() if c.requires_grad], lr=model_conf().lr)

    wikipoints = get_wikipoints(mhqa.bert.tokenizer)
    print("created mhqa with", num_params(mhqa), "params")

    losses = []
    for i, w in tqdm(enumerate(wikipoints)):
        optim.zero_grad()
        if i >= max_examples != -1:
            break

        try:
            loss, pred_ans = mhqa(w)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        except TooManyTokens as e:
            # print(e)
            pass

        if i % print_loss_every == 0 and i > 0:
            print("loss:", mean(losses[-print_loss_every:]))

    valid_acc = evaluate(mhqa)
    return valid_acc