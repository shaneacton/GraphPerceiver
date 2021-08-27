import random

from numpy import mean

from Code.Model.bert_embedder import TooManyTokens
from Code.Training.eval import evaluate
from Code.Utils.dataset_utils import get_wikipoints
from Code.Utils.model_utils import num_params, save_checkpoint, get_model
from Config.options import max_examples, print_loss_every, num_epochs


def train_and_eval(run_name="checkpoint"):
    """trains the model for 1 epoch"""
    mhqa, optim, performance = get_model(run_name)

    wikipoints = get_wikipoints(mhqa.bert.tokenizer)
    print("created mhqa with", num_params(mhqa), "params")

    for e in range(num_epochs):
        if mhqa.last_epoch > e:
            continue
        random.Random(e).shuffle(wikipoints)
        valid_acc = train_epoch(mhqa, optim, wikipoints, e, run_name)
        print("epoch", e, "validation acc:", valid_acc)

    return valid_acc


def train_epoch(mhqa, optim, wikipoints, epoch, run_name):
    losses = []
    mhqa.last_epoch = epoch

    for i, w in enumerate(wikipoints):
        optim.zero_grad()
        if i >= max_examples != -1:
            break
        if mhqa.last_epoch == epoch and mhqa.last_i > i:
            continue

        try:
            loss, pred_ans = mhqa(w)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        except TooManyTokens as e:
            # print(e)
            pass

        mhqa.last_i = i

        if i % print_loss_every == 0 and i > 0:
            print("e:", epoch, "i:", i, "loss:", mean(losses[-print_loss_every:]))
            save_checkpoint(run_name, mhqa, optim, None)
    mhqa.last_i = 0
    mhqa.last_epoch = epoch + 1

    valid_acc = evaluate(mhqa)
    return valid_acc
