from tqdm import tqdm
# from tqdm import tqdm_notebook as tqdm
from datetime import datetime

import torch

from utils.pytorch_modelsize import SizeEstimator

import warnings

warnings.filterwarnings("ignore",
                        message="Please also save or load the state ")


class Trainer():
    def __init__(self,
                 model,
                 device,
                 epoch_interval: int,
                 save_path: str,
                 total_steps_num: int = 2000000,
                 check_model_size: bool = False):

        if check_model_size:
            se = SizeEstimator(model)
            se.get_parameter_sizes()
            se.calc_param_bits()
            sz = se.param_bits
            self.model_memory_gb = (sz / 8) / (1024**3)
        else:
            self.model_memory_gb = None

        # Other parameters
        self.global_steps = 0  # number of steps counter
        self.local_steps = 0
        self.val_steps = 0
        self.total_norm = 0

        # number of samples to train every (epoch)
        self.epoch_finished = False
        self.epoch = 0
        self.epoch_intv = epoch_interval

        # losses
        self.train_loss = 0
        self.epoch_loss = 0
        self.val_loss = 0
        self.kl_train = 0
        self.lx_train = 0
        self.kl_val = 0
        self.lx_val = 0
        self.best_loss = 1e6  # best valid_tn loss

        self.save_path = save_path  # path to save best mpdel

        self.in_train = True
        self.end = total_steps_num

        # progress bar data
        self.trainpb = tqdm(total=self.end, unit="GlobalStep")
        self.postfix = {"epoch loss": 0.0, "test loss": 0.0}
        self.trainpb.set_description(f"Global Step {self.global_steps}")

    def step(self, model, optimizer, scheduler, batch, vs, check_grad=False):
        # train model
        model.train()
        optimizer.zero_grad()
        # forward
        _, loss, (kl, lx) = model(batch, vs)
        # backward and update
        loss.backward()

        self.total_norm = 0
        if check_grad:
            parameters = list(
                filter(lambda p: p.grad is not None, model.parameters()))
            counter = 0
            for p in parameters:
                param_norm = p.grad.data.norm(2)
                self.total_norm += param_norm.item()**2
                counter += 1
            self.total_norm = self.total_norm**(1. / 2)
            self.total_norm = self.total_norm / counter

        # torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()

        self.train_loss = loss.item()
        self.epoch_loss += self.train_loss
        self.kl_train += kl
        self.lx_train += lx

    def eval(self, model, val_batch, vs):
        # validate
        model.eval()
        with tqdm(val_batch, leave=False, unit="valbatch") as valpb:
            for val_mini_batch in valpb:

                with torch.no_grad():
                    _, loss, (kl, lx) = model(val_mini_batch, vs)
                    self.val_loss += loss.item()
                    self.kl_val += kl
                    self.lx_val += lx

                self.val_steps += 1

                # update progress bars
                valpb.set_postfix({"test loss": loss.item()})
                valpb.update(1)

        return

    def save_checkpoint(self,
                        model,
                        optimizer,
                        scheduler,
                        var_scale,
                        best_model=True):
        model_state = model.state_dict()
        optimizer_state = optimizer.state_dict()
        scheduler_state = scheduler.state_dict()

        time_tag = str(datetime.now().strftime("%d-%m-%Hh%M"))
        if best_model:
            modelname = f"slim_{time_tag}_{self.global_steps}.pt"
        else:
            modelname = "reg_save_slim.pt"
        save_path = f"{self.save_path}{modelname}"
        state_dict = {
            "steps": self.global_steps,
            "loss": self.best_loss,
            "epoch": self.epoch,
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "scheduler_state_dict": scheduler_state,
            "var_scale": var_scale
        }

        torch.save(state_dict, save_path)
