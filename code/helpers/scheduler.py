import torch
from torch.optim.lr_scheduler import _LRScheduler


class LinearDecayLR(_LRScheduler):
    """Linear decay scheduler.
    Args:
        optimizer (torch.optim.Optimizer): Optimizer.
        lr_i (float, optional): Initial learning rate.
        lr_f (float, optional): Final learning rate.
        s_n (float, optional): number of steps.
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 lr_i: float = 5e-4,
                 lr_f: float = 5e-5,
                 s_n: float = 1e6) -> None:

        self.rate = (lr_i - lr_f) / s_n
        self.lr_i = lr_i
        self.lr_f = lr_f

        super().__init__(optimizer)

    def get_lr(self):
        return [
            max(
                self.lr_i - self.rate * self.last_epoch, self.lr_f)
            for base_lr in self.base_lrs
        ]
