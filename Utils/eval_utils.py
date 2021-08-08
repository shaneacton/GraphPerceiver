import re
import string
from collections import Counter
from typing import List

import nlp

from Utils.dataset_utils import load_unprocessed_dataset


def normalize_answer(s: str):
    """
        from patil-suraj : longformer_qa_training
        Lower text and remove punctuation, articles and extra whitespace.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth:str):
    """
        from patil-suraj : longformer_qa_training

    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction: str, ground_truth: str):
    """
        from patil-suraj : longformer_qa_training
    """
    em = (normalize_answer(prediction) == normalize_answer(ground_truth))
    return em


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]):
    """
        from patil-suraj : longformer_qa_training
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def get_acc_and_f1(gold_answers: List[List[str]], predictions: List[str]):
    """
        from patil-suraj : longformer_qa_training
    """
    f1 = exact_match = total = 0
    if isinstance(gold_answers[0], str):
        gold_answers = [[g] for g in gold_answers]

    if len(predictions) == 0:
        return {'exact_match': 0, 'f1': 0}
    for ground_truths, prediction in zip(gold_answers, predictions):
        if not prediction:
            continue
        total += 1
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}