import torch
from nlp import tqdm
from numpy import mean

from Model.bert_embedder import TooManyTokens
from Training.eval import evaluate
from config import device, print_loss_every, lr, weight_decay, max_examples
from Utils.dataset_utils import get_wikipoints
from Model.mhqa_model import MHQA
from Utils.model_utils import num_params

mhqa = MHQA().to(device)
optim = torch.optim.SGD([c for c in mhqa.parameters() if c.requires_grad], lr=lr)

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


