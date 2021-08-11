import os
import sys
from os.path import join

dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path_1 = os.path.split(os.path.split(dir_path)[0])[0]
sys.path.append(dir_path_1)
sys.path.append(join(dir_path_1, 'Code'))
sys.path.append(join(dir_path_1, 'Config'))
sys.path.append(join(dir_path_1, 'Checkpoint'))
sys.path.append(join(dir_path_1, 'Data'))

from Config.options import device
from Code.Model.mhqa_model import MHQA
from Code.Training.train import train_and_eval

mhqa = MHQA().to(device)

train_and_eval(mhqa)
