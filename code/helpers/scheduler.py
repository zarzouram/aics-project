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
            max(self.lr_i - self.rate * self.last_epoch, self.lr_f)
            for base_lr in self.base_lrs
        ]


class SigmaAnnealer:
    """Annealer for Loss function in the genration network.
    Args:
        init (float): Initial value.
        final (float): Final value.
        constant (float): Constant value for pre-train
        steps (int): Number of annealing steps.
        pretrain (int): Number of pre-training steps.
    """
    def __init__(self, init: float, final: float, constant: float, steps: int,
                 pretrain: int) -> None:

        # based on
        # https://github.com/rnagumo/gqnlib/blob/96bd8499f90c00b29817f71e6380bc622ce78479/gqnlib/scheduler.py#L66

        if steps < pretrain:
            steps += pretrain

        self.init = init
        self.rate = (init - final) / (steps - pretrain)
        self.final = final
        self.constant = constant
        self.pretrain = pretrain

        # Current time step
        self.t = 0

    def __iter__(self):
        return self

    def __next__(self) -> float:
        self.t += 1

        if self.t <= self.pretrain:
            value = self.constant
        else:
            value = max(self.init - self.rate * self.t, self.final)

        # Return sigma
        return value
