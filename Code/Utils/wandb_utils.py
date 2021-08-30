from Config.options import model_conf

try:
    import wandb
except Exception as e:
    print("wandb error:", e)
    model_conf()["use_wandb"] = False


_wandb_run = None


def wandb_run():
    if _wandb_run is None:
        raise Exception("no wandb run")
    return _wandb_run


def new_run(model_name):
    global _wandb_run
    id = wandb.util.generate_id()
    print("starting new wandb run. ID:", id)
    model_conf()["wandb_id"] = id
    try:
        _wandb_run = wandb.init(project="gnn_thesis", entity="shaneacton", config=model_conf(), resume=True, name=model_name, id=id)
    except Exception as e:
        print("cannot init wandb session. turning off wandb logging")
        print(e)
        model_conf()["use_wandb"] = False
        model_conf()["wandb_id"] = -1
        return None

    return _wandb_run


def continue_run(id, model_name):
    if id == -1:
        raise Exception("cannot continue wandb run. no valid  run id")
    global _wandb_run
    _wandb_run = wandb.init(project="gnn_thesis", entity="shaneacton", config=model_conf(), resume=True, name=model_name, id=id)
    print("continuing wandb run, id=", id)
    return _wandb_run