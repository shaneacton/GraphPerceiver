
class Performance:

    def __init__(self):
        self.losses = []
        self.valid_accs = []
        self.num_losses_per_epoch = -1

    def log_loss(self, loss):
        self.losses.append(loss)

    def log_valid_acc(self, acc):
        self.valid_accs.append(acc)
        if self.num_losses_per_epoch == -1:
            self.num_losses_per_epoch = len(self.losses)
