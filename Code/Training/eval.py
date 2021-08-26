import nlp
import torch
from numpy import mean
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from Code.Model.bert_embedder import TooManyTokens
from Code.Model.mhqa_model import MHQA
from Code.Utils.dataset_utils import get_wikipoints
from Code.Utils.eval_utils import get_acc_and_f1
from Config.options import max_examples

_test = None


def get_test(tokeniser: PreTrainedTokenizerBase):
    global _test
    if _test is None:
        _test = get_wikipoints(tokeniser, split=nlp.Split.VALIDATION)
        print("num valid ex:", len(_test))
    return _test


def evaluate(model: MHQA):
    test_set = get_test(model.bert.tokenizer)
    answers = []
    predictions = []
    chances = []

    model.eval()

    with torch.no_grad():
        for i, ex in enumerate(test_set):
            if i >= max_examples != -1:
                break

            try:
                _, pred_ans = model(ex)
            except TooManyTokens as e:
                continue

            answers.append([ex.answer])
            predictions.append(pred_ans)
            chances.append(1./len(ex.candidates))

    model.last_example = -1

    valid_acc = get_acc_and_f1(answers, predictions)['exact_match']
    print("eval completed. Validation acc:", valid_acc, "chance:", mean(chances))
    return valid_acc
