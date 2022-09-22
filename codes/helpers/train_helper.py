from typing import List, Optional
from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm

import math

import torch
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
                 lang_enc_train_steps: int = 20000,
                 sigmas_const: List[float] = [2.0, 0.7, 2e5],
                 val_interv: int = 499,
                 device_ids: Optional[List[int]] = None,
                 pretrain: Optional[str] = None,
                 freeze_gen: Optional[int] = None):

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

        # Pixel-variance annealing parameters
        self.n = lang_enc_train_steps
        self.sigma_f, self.sigma_i, num_steps = sigmas_const
        self.sigma_rate = (self.sigma_i - self.sigma_f) / num_steps
        self.sigma_annealing()

        # losses
        self.tracking = TrackMetrics()
        self.best_model = False

        # Tensorboard
        loss_logger = SummaryWriter(log_dir=f"{log_dir}/loss")
        nll_logger = SummaryWriter(log_dir=f"{log_dir}/nll")
        kl_logger = SummaryWriter(log_dir=f"{log_dir}/kl")
        std_logger = SummaryWriter(log_dir=f"{log_dir}/std")
        p_error_logger = SummaryWriter(log_dir=f"{log_dir}/p_error")
        self.imgs_logger = SummaryWriter(log_dir=f"{log_dir}/imgs")
        self.writers = {
            "loss": loss_logger,
            "nll": nll_logger,
            "kl": kl_logger,
            "std": std_logger,
            "p_error": p_error_logger,
        }
        self.lrs_logger = [
            SummaryWriter(log_dir=f"{log_dir}/lrs") for _ in self.optims
        ]

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

    def unfreez_draw(self):
        self.model.freeze_draw(False)

        draw_parms = {
            "params": self.model.target_viewpoint_encoder.parameters()
        }
        self.optims[-1].add_param_group(draw_parms)
        draw_parms = {"params": self.model.gen_model.parameters()}
        self.optims[-1].add_param_group(draw_parms)

    def sigma_annealing(self):
        # Pixel-variance annealing
        self.sigma = max(self.sigma_i - self.sigma_rate * self.global_step,
                         self.sigma_f)

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
                self.sigma_annealing()
                if self.global_step == self.freeze_gen:
                    self.unfreez_draw()

    def resume(self, state_path):
        self.load_states(state_path)
        self.run()

    def step(self, batch: List[Tensor], train: bool):
        phase = "train" if train else "val"
        with torch.set_grad_enabled(train):
            output = self.model([b.to(self.device) for b in batch], self.sigma)
            img_const, kl, nll, p_error = output
            loss = nll + kl
            if train:
                loss.backward()
                for optim in self.optims:
                    optim.step()
                for scheduler in self.schedulers:
                    scheduler.step()

        # track losses and generated images
        track_dict = {
            "loss": loss.item(),
            "kl": kl.item(),
            "nll": nll.item(),
            "std": self.sigma,
            "p_error": p_error,
        }
        self.tracking.add_running(track_dict, phase=phase)

        if not train:
            images = {
                "img_gt": batch[0].detach().to(torch.device("cpu")),
                "img_gn": img_const.detach().to(torch.device("cpu"))
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
        if phase == "train":
            lrs = [optim.param_groups[0]["lr"] for optim in self.optims]
            for i, (lr, lr_logger) in enumerate(zip(lrs, self.lrs_logger)):
                lr_logger.add_scalar(f"optim_{i}",
                                     lr,
                                     self.global_step,
                                     new_style=True)

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
