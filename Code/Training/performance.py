from Code.Utils.wandb_utils import wandb_run


class Performance:

    def __init__(self):
        self.losses = []
        self.valid_accs = []
        self.num_losses_per_epoch = -1

    def log_loss(self, loss, e):
        print("e:", e, "loss:", loss)
        self.losses.append(loss)
        wandb_run().log({"loss": loss, "epoch": e})

    def log_valid_acc(self, acc):
        e = len(self.valid_accs)
        print("epoch", e, "validation acc:", acc)
        wandb_run().log({"valid_acc": acc, "epoch": e + 1})
        self.valid_accs.append(acc)
        if self.num_losses_per_epoch == -1:
            self.num_losses_per_epoch = len(self.losses)

