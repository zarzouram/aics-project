from typing import List, Optional
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

import math

import torch
import torch.nn as nn
from torch import Tensor

from torch.utils.tensorboard import SummaryWriter

from codes.utils.utils import seed_everything
from codes.helpers.tracking import TrackMetrics


class Trainer():

    def __init__(self,
                 model: torch.nn.Module,
                 optims: List[torch.optim.Optimizer],
                 schedulers: List[torch.optim.lr_scheduler._LRScheduler],
                 train_iter: torch.utils.data.DataLoader,
                 val_iter: torch.utils.data.DataLoader,
                 device: torch.device,
                 save_path: str,
                 log_dir: str,
                 seed: int,
                 total_steps: int = int(1e6),
                 val_interv: int = 499,
                 device_ids: Optional[List[int]] = None,
                 pretrain: Optional[str] = None,
                 freeze_gen: Optional[int] = None):

        if device_ids is not None:
            self.model = nn.DataParallel(model, device_ids=device_ids)
            self.dp = True
        else:
            self.dp = False
        self.model = model.to(device)

        self.optims = optims
        self.schedulers = schedulers
        self.device = device
        self.train_iter = train_iter
        self.val_iter = val_iter

        self.pretrain = pretrain
        self.freeze_gen = freeze_gen

        # Other parameters
        self.global_step = 0  # number of steps counter
        self.seed = seed
        self.in_train = True
        self.total_steps = total_steps
        self.val_interval = val_interv
        self.best_loss = math.inf

        # losses
        self.tracking = TrackMetrics()
        self.best_model = False

        # Tensorboard
        self.logger = SummaryWriter(log_dir=f"{log_dir}/logs")
        loss_logger = SummaryWriter(log_dir=f"{log_dir}/loss")
        nll_logger = SummaryWriter(log_dir=f"{log_dir}/nll")
        kl_logger = SummaryWriter(log_dir=f"{log_dir}/kl")
        std_logger = SummaryWriter(log_dir=f"{log_dir}/std")
        self.imgs_logger = SummaryWriter(log_dir=f"{log_dir}/imgs")
        self.writers = {
            "loss": loss_logger,
            "nll": nll_logger,
            "kl": kl_logger,
            "std": std_logger
        }

        self.save_path = save_path  # path to save best model

        # progress bar data
        self.trainpb = tqdm(total=self.total_steps,
                            unit="Step",
                            initial=self.global_step)

        self._state_dict = {
            "global_step": self.global_step,
            "seed": self.seed,
            "best_loss": self.best_loss,
            "best_model": self.best_model
        }

    def run(self):
        seed_everything(self.seed)

        while self.in_train:
            if self.pretrain == "caption_encoder":
                self.train_language_encoder()
            else:
                self.train()
                if self.global_step % self.val_interval == 0:
                    self.eval()
                    self.best_model = bool(
                        self.tracking.last_metric() < self.best_loss)
                    if self.best_model:
                        self.best_loss = self.tracking.last_metric()

                self.save_checkpoint()

    def resume(self, state_path):
        self.load_states(state_path)
        self.run()

    # def get_output(
    #         self, output: Union[Tuple[Tensor],
    #                             Tuple[Tuple[Tensor]]]) -> Tuple[Tensor]:
    #     # when using nn.DataParallel the output is a list of tensors of length
    #     # equal to number of devices used.
    #     # Stack the images and average the losses
    #     if self.dp:
    #         output = [torch.vstack(o) for o in zip(*output)]
    #         output = [o.mean() if o.dim() == 2 else o for o in output]

    #     if self.pretrain == "draw":
    #         img_const, kl, nll, img_std = output
    #     else:
    #         img_const, kl, nll, img_std, _ = output

    #     return img_const, kl, nll, img_std

    def step(self, batch: List[Tensor], train: bool):
        with torch.set_grad_enabled(train):
            output = self.model([b.to(self.device) for b in batch])
            img_const, kl, nll, img_std = output
            loss = nll + kl
            if train:
                loss.backward()
                for optim in self.optims:
                    optim.step()

        # track losses and generated images
        phase = "train" if train else "val"
        track_dict = {
            "loss": loss.item(),
            "kl": kl.item(),
            "nll": nll.item(),
            "std": img_std.item()
        }
        self.tracking.add_running(track_dict, phase=phase)

        if not train:
            images = {
                "img_gt": batch[0].to(torch.device("cpu")),
                "img_gn": img_const.to(torch.device("cpu"))
            }
            self.tracking.add_images(images)

    def train(self):
        self.trainpb.set_description("Training ...")
        self.model.train()

        with tqdm(self.train_iter, leave=False, unit="image") as batch_trainpb:
            for batch in batch_trainpb:
                for optim in self.optims:
                    optim.zero_grad()
                self.step(batch, train=True)

                self.global_step += 1
                self.trainpb.update(1)

        if self.global_step % self.val_interval == 0:
            for scheduler in self.schedulers:
                scheduler.step(epoch=self.global_step)

        self.tracking.update("train")
        self.plot("train")

    def eval(self) -> Tensor:
        self.trainpb.set_description("Eval ...")
        self.model.eval()
        with tqdm(self.val_iter, leave=False, unit="image") as valpb:
            for batch in valpb:
                self.step(batch, train=False)

        grid = self.tracking.track_best_images()
        self.tracking.update("val")
        self.plot("val")
        self.imgs_logger.add_image("Reconstruction", grid, self.global_step)

    def plot(self, phase: str):
        metrics = self.tracking.metrics[phase]
        for n, vs in metrics.items():
            self.writers[n].add_scalars(n, {phase: vs[-1]}, self.global_step)

    def states(self):
        return self._state_dict

    def load_states(self, state: dict):
        for name, value in state.items():
            self._state_dict[name] = value

        self.trainpb.initial = self.global_step

    def save_checkpoint(self):
        model_state = self.model.state_dict()
        optimizer_state = [optim.state_dict() for optim in self.optims]
        scheduler_state = [scd.state_dict() for scd in self.schedulers]
        tracking_state = self.tracking.states()
        trainer_states = self.states()

        filename = "checkpoint"
        if self.pretrain == "draw":
            filename += "_draw"
        if self.best_model:
            filename += "_best"

        save_path = str(self.save_path / f"{filename}.pt")
        state_dict = {
            "model_state": model_state,
            "optims_state": optimizer_state,
            "schedulers_state": scheduler_state,
            "tracking_state": tracking_state,
            "trainer_states": trainer_states
        }

        torch.save(state_dict, save_path)

        if self.best_model:
            self.best_model = False

    def load_checkpoint(self, state_path):
        state = torch.load(state_path)
        self.model.load_state_dict(state["model_state"])
        for optim_state, optim in zip(state["optims_state"], self.optims):
            optim.load_state_dict(optim_state)
        for scd_state, scd in zip(state["schedulers_state"], self.schedulers):
            scd.load_state_dict(scd_state)
        self.tracking.load_states(state["tracking_state"])
        self.load_states(state["trainer_states"])

    def train_language_encoder(self):
        raise NotImplementedError
