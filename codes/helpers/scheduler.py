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
                 lr_init: float = 5e-4,
                 lr_final: float = 5e-5,
                 step_num: float = 1e6) -> None:

        self.rate = (lr_init - lr_final) / step_num
        self.lr_i = lr_init
        self.lr_f = lr_final

        super().__init__(optimizer)

    def get_lr(self):
        return [
            max(self.lr_i - self.rate * self.last_epoch, self.lr_f)
            for base_lr in self.base_lrs
        ]


class VarAnnealer:
    """Annealer for Loss function in the genration network.
    Args:
        init (float): Initial value.
        final (float): Final value.
        constant (float): Constant value for pre-train
        steps (int): Number of annealing steps.
        pretrain (int): Number of pre-training steps.
    """
    def __init__(self,
                 init: float = 1.0,
                 final: float = 2.0,
                 constant: float = 1.0,
                 steps: int = 700000,
                 pretrain: int = 30500) -> None:

        if steps < pretrain:
            steps += pretrain

        self.init = init
        self.rate = (init - final) / steps
        self.final = final
        self.constant = constant
        self.pretrain = pretrain

        self.scale = 0

        # Current step
        self.t = 0

    def __iter__(self):
        return self

    def __next__(self) -> float:

        if self.t <= self.pretrain:
            value = self.constant
        else:
            value = max(self.init - self.rate * self.t, self.final)

        self.t += 1

        self.scale = value
        # Return var scale
        return value
