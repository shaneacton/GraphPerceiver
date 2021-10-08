import random

from numpy import mean
from tqdm import tqdm

from Code.Model.bert_embedder import TooManyTokens
from Code.Model.mhqa.mhqa_model import MHQAModel
from Code.Training.eval import evaluate
from Code.Utils.dataset_utils import get_wikipoints
from Code.Utils.model_utils import num_params, save_checkpoint, get_model
from Code.Utils.wandb_utils import wandb_run
from Config.options import max_examples, print_loss_every, num_epochs, model_conf, bert_freeze_epochs

num_training_examples = -1


def train_and_eval():
    """trains the model for 1 epoch"""
    mhqa, optim, performance = get_model(model_conf().model_name)
    print(mhqa)
    if model_conf().use_wandb:
        try:
            wandb_run().watch(mhqa)
        except:
            pass

    wikipoints = get_wikipoints(mhqa.bert.tokenizer)
    print("training mhqa with", num_params(mhqa), "params")
    global num_training_examples
    num_training_examples = len(wikipoints)

    for e in range(num_epochs):
        if mhqa.last_epoch > e:
            continue
        random.Random(e).shuffle(wikipoints)
        valid_acc = train_epoch(mhqa, optim, wikipoints, e, model_conf().model_name, performance)
        performance.log_valid_acc(valid_acc)

    return valid_acc


def train_epoch(mhqa: MHQAModel, optim, wikipoints, epoch, run_name, performance):
    losses = []
    mhqa.last_epoch = epoch
    if epoch < bert_freeze_epochs:
        mhqa.set_bert_trainable(False)
    else:
        mhqa.set_bert_trainable(True)

    for i, w in tqdm(enumerate(wikipoints)):
        optim.zero_grad()
        if i >= max_examples != -1:
            break
        if mhqa.last_epoch == epoch and mhqa.last_i >= i:
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
            e = epoch + i/num_training_examples
            performance.log_loss(mean(losses[-print_loss_every:]), e)
            save_checkpoint(run_name, mhqa, optim, performance)
    mhqa.last_i = 0
    mhqa.last_epoch = epoch + 1

    valid_acc = evaluate(mhqa)
    return valid_acc
