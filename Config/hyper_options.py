import joblib
import optuna

from Config.options import model_conf, device
from Code.Model import MHQA
from Code.Training import train_and_eval

study = optuna.create_study()


def objective(trial):
    """we run the model with"""
    joblib.dump(study, 'study.pkl')
    print(type(trial))
    num_summariser_layers = trial.suggest_int("num_summariser_layers", 1, 5)
    model_conf().num_summariser_layers = num_summariser_layers
    mhqa = MHQA().to(device)
    valid_acc = train_and_eval(mhqa)
    cost = -valid_acc
    return cost


study.optimize(objective, timeout=60*60*10)



