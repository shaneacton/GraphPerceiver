import argparse
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

from Config.options import set_config_name

parser = argparse.ArgumentParser()  # error: <Process(Process-2, initial)>
parser.add_argument('--config_name', '-c', help='Whether or not to run the debug configs - y/n', default="model_params")
args = parser.parse_args()
set_config_name(args.config_name)


from Code.Training.train import train_and_eval
train_and_eval()
