from Config.options import device
from Model.mhqa_model import MHQA
from Training.train import train_and_eval

mhqa = MHQA().to(device)

train_and_eval(mhqa)
