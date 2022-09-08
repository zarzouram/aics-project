import torch
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np


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
                 step_num: float = 1e6,
                 step_interv: int = 500) -> None:

        self.rate = (lr_init - lr_final) / step_num
        self.lr_i = lr_init
        self.lr_f = lr_final
        self.step_interv = step_interv
        self.lr_factor_ = lr_init

        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch % self.step_interv == 0:
            lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        else:
            lr_factor = self.lr_factor_
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = max(self.lr_i - self.rate * epoch, self.lr_f)
        self.lr_factor_ = lr_factor
        return lr_factor


class XfmrWarmupScheduler(_LRScheduler):

    def __init__(
        self,
        optimizer,
        warmup: int = 4000,
        lr_init: float = 5e-4,
        lr_final: float = 5e-5,
        cos_step: float = 20000,
        step_num: float = 1e6,
        step_interv: int = 500
    ):

        self.warmup = warmup
        self.max_step = step_num
        self.cos_step = cos_step
        self.rate = (lr_init - lr_final) / step_num
        self.lr_i = lr_init
        self.lr_f = lr_final
        self.lr_factor_ = lr_init
        self.step_interv = step_interv

        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        self.lr_factor_ = lr_factor
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup:
            lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.cos_step))
            lr_factor *= epoch * 1.0 / self.warmup
        elif epoch <= self.cos_step and epoch > self.warmup:
            lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.cos_step))
        else:
            if epoch % self.step_interv == 0:
                lr_factor = max(self.lr_i - self.rate * epoch, self.lr_f)
            else:
                lr_factor = self.lr_factor_
        return lr_factor
