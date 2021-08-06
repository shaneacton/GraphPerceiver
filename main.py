from config import device
from dataset_utils import get_wikipoints
from mhqa_model import MHQA
from model_utils import num_params

mhqa = MHQA().to(device)
wikipoints = get_wikipoints(mhqa.bert.tokenizer)
print("created mhqa with", num_params(mhqa), "params")

for w in wikipoints:
    print(w)
    mhqa(w)
    break

